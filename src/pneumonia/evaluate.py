"""Held-out evaluation: models + baselines, val-tuned and default thresholds, CIs, figures.

The official test folder is scored once per checkpoint. Test predictions are cached next to
a fingerprint of the checkpoint; re-running `pneumonia evaluate` reuses them, and every
actual scoring of the test split is appended to results/predictions/test_access_log.csv.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import plots
from .config import load_config, run_dir
from .data import make_loader, read_manifests, resolve_root
from .indexing import KAGGLE_SLUG
from .metrics import (
    BASELINES,
    baseline_scores,
    binary_metrics,
    bootstrap_ci,
    choose_threshold,
    operating_curve,
)
from .model import build_model
from .train import predict
from .utils import get_device, git_commit, library_versions, utc_now

RESULT_COLS = [
    "model",
    "kind",
    "split",
    "threshold_rule",
    "threshold",
    "n",
    "n_pos",
    "n_neg",
    "recall",
    "specificity",
    "fp",
    "precision",
    "npv",
    "f2",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "tp",
    "tn",
    "fn",
    "note",
]


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_checkpoint(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ck["config"]["model"]
    model = build_model(m["arch"], pretrained=False, dropout=float(m["dropout"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def score_split(
    cfg,
    model,
    df,
    root,
    device,
    pred_dir: Path,
    name: str,
    split: str,
    fingerprint: str,
    rescore: bool = False,
) -> pd.DataFrame:
    pred_path = pred_dir / f"{name}_{split}.csv"
    meta_path = pred_dir / f"{name}_{split}.meta.json"
    if pred_path.exists() and meta_path.exists() and not rescore:
        meta = json.loads(meta_path.read_text())
        if meta.get("checkpoint_sha256_16") == fingerprint:
            print(f"[evaluate] reusing cached {split} predictions for {name} ({pred_path})")
            return pd.read_csv(pred_path)
    loader = make_loader(cfg, df, root, train=False, shuffle=False)
    s, y, i, _ = predict(model, loader, device)
    out = df.iloc[i][["path", "class_name", "label", "patient_id", "subtype"]].copy()
    out["score"] = s
    pred_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    meta_path.write_text(json.dumps({"checkpoint_sha256_16": fingerprint, "scored_at": utc_now()}))
    if split == "test":
        log = pred_dir / "test_access_log.csv"
        row = pd.DataFrame(
            [
                {
                    "model": name,
                    "checkpoint_sha256_16": fingerprint,
                    "scored_at": utc_now(),
                    "n": len(out),
                }
            ]
        )
        row.to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"[evaluate] scored the held-out TEST split for {name} (logged in {log})")
    return out


def _row(model, kind, split, rule, y, s, thr, note=""):
    m = binary_metrics(y, s, thr)
    return {"model": model, "kind": kind, "split": split, "threshold_rule": rule, **m, "note": note}


def _blank_row(model, kind, split, rule, note):
    r = {c: np.nan for c in RESULT_COLS}
    r.update({"model": model, "kind": kind, "split": split, "threshold_rule": rule, "note": note})
    return r


def _image_stats(path: Path) -> dict:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        w, h = im.size
    return {
        "mean_intensity": float(a.mean()),
        "std_intensity": float(a.std()),
        "width": w,
        "height": h,
    }


def error_analysis(test_pred: pd.DataFrame, thr: float, root: Path, out_dir: Path) -> dict:
    df = test_pred.copy()
    df["pred"] = (df["score"] >= thr).astype(int)
    pos = df[df["label"] == 1]
    fn = pos[pos["pred"] == 0].sort_values("score")
    tp = pos[pos["pred"] == 1]
    stats = pd.DataFrame([_image_stats(root / p) for p in pos["path"]], index=pos.index)
    pos = pos.join(stats)
    fn = fn.join(stats)
    fn[["path", "subtype", "score", "mean_intensity", "std_intensity", "width", "height"]].to_csv(
        out_dir / "false_negatives.csv", index=False
    )

    def share(d, k):
        return float((d["subtype"] == k).mean()) if len(d) else float("nan")

    res = {
        "n_false_negatives": int(len(fn)),
        "n_positives": int(len(pos)),
        "fn_subtype_counts": fn["subtype"].value_counts().to_dict(),
        "positives_subtype_counts": pos["subtype"].value_counts().to_dict(),
        "viral_share_fn": share(fn, "virus"),
        "viral_share_all_positives": share(pos, "virus"),
        "mean_intensity_fn": float(fn["mean_intensity"].mean()) if len(fn) else float("nan"),
        "mean_intensity_tp": float(pos.loc[tp.index, "mean_intensity"].mean())
        if len(tp)
        else float("nan"),
        "contrast_std_fn": float(fn["std_intensity"].mean()) if len(fn) else float("nan"),
        "contrast_std_tp": float(pos.loc[tp.index, "std_intensity"].mean())
        if len(tp)
        else float("nan"),
        "fn_median_score": float(fn["score"].median()) if len(fn) else float("nan"),
        "fn_lowest_scores": fn["score"].head(5).round(4).tolist(),
    }
    # FN rate per subtype: are viral pneumonias missed more often?
    res["miss_rate_by_subtype"] = {
        k: float((g["pred"] == 0).mean()) for k, g in df[df["label"] == 1].groupby("subtype")
    }
    return res


def _nan_to_none(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return o


def evaluate(config_paths: list[str], device: str | None = None, rescore: bool = False) -> Path:
    cfgs = [load_config(p) for p in config_paths]
    primary = cfgs[0]
    ev = primary["eval"]
    results_dir = Path(primary["paths"]["results_dir"])
    fig_dir = results_dir / "figures"
    pred_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device(device or primary["train"].get("device", "auto"))
    root = resolve_root(primary)
    splits = read_manifests(primary["data"]["splits_dir"])
    target = float(primary["selection"]["target_recall"])
    sens_targets = [float(t) for t in primary["selection"]["sensitivity_targets"]]
    seed = int(primary["experiment"]["seed"])
    rule_tuned = f"val-tuned (recall>={target:.2f})"
    rule_default = "default 0.5"

    rows, models_meta, curves_test, preds = [], {}, {}, {}

    # ---------------------------------------------------------------- baselines
    yv, yt = splits["val"]["label"].to_numpy(), splits["test"]["label"].to_numpy()
    for b in BASELINES:
        sv = baseline_scores(b, len(yv), seed)
        st = baseline_scores(b, len(yt), seed + 1)
        if b == "random score":
            thr = choose_threshold(yv, sv, target)["threshold"]
            for split, y, s in (("val", yv, sv), ("test", yt, st)):
                rows.append(_row(b, "baseline", split, rule_tuned, y, s, thr))
                rows.append(_row(b, "baseline", split, rule_default, y, s, 0.5))
        else:  # constant predictors: the threshold is irrelevant
            for split, y, s in (("val", yv, sv), ("test", yt, st)):
                rows.append(_row(b, "baseline", split, "constant", y, s, 0.5))

    # ---------------------------------------------------------------- trained models
    for cfg in cfgs:
        name = cfg["experiment"]["name"]
        ckpt = run_dir(cfg) / "best.pt"
        if not ckpt.exists():
            note = (
                f"not run: no checkpoint at {ckpt.as_posix()}; fill with `pneumonia train "
                f"--config {cfg['_config_path']}` then re-run evaluate"
            )
            for split in ("val", "test"):
                for rule in (rule_tuned, rule_default):
                    rows.append(_blank_row(name, "model", split, rule, note))
            models_meta[name] = {"status": "not run", "note": note}
            continue
        model, ck = load_checkpoint(ckpt, dev)
        fp = _fingerprint(ckpt)
        pv = score_split(cfg, model, splits["val"], root, dev, pred_dir, name, "val", fp, rescore)
        pt = score_split(cfg, model, splits["test"], root, dev, pred_dir, name, "test", fp, rescore)
        yv_m, sv_m = pv["label"].to_numpy(), pv["score"].to_numpy()
        yt_m, st_m = pt["label"].to_numpy(), pt["score"].to_numpy()
        op = choose_threshold(yv_m, sv_m, target)
        thr = op["threshold"]
        for split, y, s in (("val", yv_m, sv_m), ("test", yt_m, st_m)):
            rows.append(_row(name, "model", split, rule_tuned, y, s, thr))
            rows.append(_row(name, "model", split, rule_default, y, s, 0.5))
        sens = []
        for t in sens_targets:
            o = choose_threshold(yv_m, sv_m, t)
            mt = binary_metrics(yt_m, st_m, o["threshold"])
            sens.append(
                {
                    "target_recall": t,
                    "threshold": o["threshold"],
                    "val_recall": o["recall"],
                    "val_specificity": o["specificity"],
                    "test_recall": mt["recall"],
                    "test_specificity": mt["specificity"],
                    "test_fp": mt["fp"],
                    "test_fn": mt["fn"],
                }
            )
        ci_tuned = bootstrap_ci(
            yt_m,
            st_m >= thr,
            n_boot=int(ev["bootstrap_n"]),
            alpha=1 - float(ev["ci_level"]),
            seed=seed,
        )
        ci_default = bootstrap_ci(
            yt_m,
            st_m >= 0.5,
            n_boot=int(ev["bootstrap_n"]),
            alpha=1 - float(ev["ci_level"]),
            seed=seed,
        )
        mt = binary_metrics(yt_m, st_m, thr)
        sentence = (
            f"{name} at the val-tuned threshold {thr:.3f} missed {mt['fn']} of "
            f"{mt['n_pos']} pneumonia cases (recall {mt['recall']:.3f}) and raised "
            f"{mt['fp']} false alarms on {mt['n_neg']} normal films "
            f"(specificity {mt['specificity']:.3f})."
        )
        print("[evaluate] " + sentence)
        models_meta[name] = {
            "status": "evaluated",
            "arch": cfg["model"]["arch"],
            "pretrained": cfg["model"]["pretrained"],
            "best_epoch": ck.get("epoch"),
            "checkpoint_sha256_16": fp,
            "val_operating_point": op,
            "shipped_threshold": thr,
            "test_at_shipped": mt,
            "test_ci_at_shipped": ci_tuned,
            "test_ci_at_0.5": ci_default,
            "recall_target_sensitivity": sens,
            "summary_sentence": sentence,
        }
        curves_test[name] = (yt_m, st_m)
        preds[name] = (pv, pt, op)
        hist = run_dir(cfg) / "history.csv"
        if hist.exists():
            plots.training_curves(
                pd.read_csv(hist), fig_dir / f"training_curves_{name}.png", title=name
            )

    res = pd.DataFrame(rows)[RESULT_COLS]
    res.to_csv(results_dir / "results.csv", index=False)

    # ---------------------------------------------------------------- figures (primary model)
    pname = primary["experiment"]["name"]
    ea = None
    if pname in preds:
        pv, pt, op = preds[pname]
        thr = op["threshold"]
        mt = models_meta[pname]["test_at_shipped"]
        plots.roc(curves_test, fig_dir / "roc.png")
        plots.precision_recall(curves_test, fig_dir / "precision_recall.png")
        plots.confusion(
            mt, fig_dir / "confusion_matrix.png", title=f"{pname}, test @ thr {thr:.3f}"
        )
        plots.operating_curve(
            operating_curve(pv["label"], pv["score"]),
            op,
            fig_dir / "threshold_operating_curve.png",
            test_point=mt,
        )
        plots.score_histogram(
            pt["label"],
            pt["score"],
            thr,
            fig_dir / "score_histogram.png",
            title=f"{pname}: test scores by true class",
        )
        hist = run_dir(primary) / "history.csv"
        if hist.exists():
            plots.training_curves(pd.read_csv(hist), fig_dir / "training_curves.png", title=pname)
        ea = error_analysis(pt, thr, root, results_dir)

    split_summary = (
        json.loads((Path(primary["data"]["splits_dir"]) / "split_summary.json").read_text())
        if (Path(primary["data"]["splits_dir"]) / "split_summary.json").exists()
        else None
    )
    metrics = {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "versions": library_versions(),
        "seed": seed,
        "target_recall": target,
        "primary_model": pname,
        "dataset": {"kaggle": KAGGLE_SLUG, "root": str(root), "splits": split_summary},
        "models": models_meta,
        "error_analysis": ea,
        "test_access_log": (
            pd.read_csv(pred_dir / "test_access_log.csv").to_dict("records")
            if (pred_dir / "test_access_log.csv").exists()
            else []
        ),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(_nan_to_none(metrics), f, indent=2, default=str)
    print(f"[evaluate] wrote {results_dir / 'results.csv'} and {results_dir / 'metrics.json'}")
    return results_dir

"""`pneumonia prepare`: index + audit the dataset, patient-grouped split, manifests, cache,
class-balance figure and a metadata-only shortcut probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import plots
from .data import (
    assert_no_patient_leakage,
    build_cache,
    cache_dir_for,
    cache_path,
    make_splits,
    split_summary,
    write_manifests,
)
from .indexing import (
    DOWNLOAD_CMD,
    KAGGLE_SLUG,
    audit_images,
    build_index,
    count_table,
    find_dataset_root,
)
from .utils import utc_now


def _border_features(img_path: Path, band: float = 0.06) -> dict:
    with Image.open(img_path) as im:
        a = np.asarray(im.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
    b = int(224 * band)
    border = np.concatenate([a[:b].ravel(), a[-b:].ravel(), a[:, :b].ravel(), a[:, -b:].ravel()])
    return {"border_mean": float(border.mean()), "border_std": float(border.std())}


def shortcut_probe(splits: dict, sizes: pd.DataFrame, root: Path, cache_dir: Path | None) -> dict:
    """Can the class be predicted WITHOUT looking at the lungs?

    Fits a logistic regression on acquisition metadata only (original width, height, aspect,
    intensity statistics of the outer image border) on train and scores val. A high AUROC
    means the two classes came through visibly different pipelines - the shortcut a CNN can
    learn instead of parenchyma - and raises the bar for the Grad-CAM audit.
    """
    feats = {}
    for name in ("train", "val"):
        df = splits[name].merge(sizes[["path", "width", "height", "aspect"]], on="path")
        rows = []
        for rel in df["path"]:
            p = cache_path(cache_dir, rel) if cache_dir else root / rel
            rows.append(_border_features(p if p.exists() else root / rel))
        feats[name] = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out = {}
    groups = {
        "size_only": ["width", "height", "aspect"],
        "border_intensity_only": ["border_mean", "border_std"],
        "size_and_border": [
            "width",
            "height",
            "aspect",
            "border_mean",
            "border_std",
        ],
    }
    for gname, cols in groups.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(feats["train"][cols], feats["train"]["label"])
        s = clf.predict_proba(feats["val"][cols])[:, 1]
        out[gname] = round(float(roc_auc_score(feats["val"]["label"], s)), 4)
    out["note"] = (
        "Val AUROC of a logistic regression on non-anatomical metadata only. "
        "0.5 = no shortcut signal."
    )
    return out


def prepare(cfg: dict, skip_audit: bool = False, skip_cache: bool = False) -> dict:
    d = cfg["data"]
    root = find_dataset_root(d["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = results_dir / "figures"
    audit_dir = results_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] dataset root resolved to: {root}")

    index = build_index(root)
    official = count_table(index)
    print("\n[prepare] Official Kaggle folders (images / distinct patients):")
    print(official.to_string())

    unparsed = index["patient_id"].str.startswith("UNPARSED-").sum()
    print(
        f"[prepare] {len(index)} images, {index['patient_id'].nunique()} distinct patients, "
        f"{unparsed} unparsed file names"
    )

    sizes = None
    if not skip_audit:
        print("[prepare] opening every image (decode check + size audit)...")
        sizes = audit_images(index, root).merge(index[["path", "class_name"]], on="path")
        sizes.to_csv(audit_dir / "image_sizes.csv", index=False)
        size_summary = (
            sizes.groupby("class_name")[["width", "height", "aspect"]]
            .describe(percentiles=[0.05, 0.5, 0.95])
            .round(2)
        )
        print(size_summary.T.to_string())
        size_summary.to_csv(audit_dir / "image_size_summary.csv")
        print(
            f"[prepare] all {len(sizes)} images decode; modes: "
            f"{sizes['mode'].value_counts().to_dict()}"
        )
        plots.image_sizes(sizes, fig_dir / "image_sizes.png")

    splits, info = make_splits(
        index,
        float(d["val_fraction"]),
        int(cfg["experiment"]["seed"]),
        bool(d.get("exclude_test_patients_from_trainval", True)),
    )
    assert_no_patient_leakage(splits)
    write_manifests(splits, d["splits_dir"])
    summ = split_summary(splits)
    print("\n[prepare] Patient-grouped splits (official test untouched):")
    print(
        summ.pivot(index="split", columns="class_name", values=["images", "patients"])
        .reindex(["train", "val", "test"])
        .to_string()
    )
    print(
        f"[prepare] test/train+val shared patient ids: {info['test_trainval_shared_patient_ids']}"
        f" -> dropped {info['dropped_trainval_images_for_test_overlap']} train/val images"
    )

    all_splits = pd.concat(splits.values())
    plots.class_balance(official, count_table(all_splits, "split"), fig_dir / "class_balance.png")

    cdir = cache_dir_for(cfg)
    if cdir is not None and not skip_cache:
        n = build_cache(index, root, cdir, int(d["image_size"]))
        print(f"[prepare] resized cache at {cdir} ({n} new files)")

    probe = shortcut_probe(splits, sizes, root, cdir) if sizes is not None else None
    if probe:
        print(f"[prepare] metadata-only shortcut probe (val AUROC): {probe}")

    summary = {
        "generated_at": utc_now(),
        "kaggle_dataset": KAGGLE_SLUG,
        "download_cmd": DOWNLOAD_CMD,
        "dataset_root": str(root),
        "n_images": len(index),
        "n_patients": int(index["patient_id"].nunique()),
        "n_unparsed_names": int(unparsed),
        "official_counts": official.reset_index().to_dict("records"),
        "split_counts": summ.to_dict("records"),
        "split_info": info,
        "shortcut_probe_val_auroc": probe,
    }
    Path(d["splits_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(d["splits_dir"]) / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(audit_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[prepare] manifests written to {d['splits_dir']}/split_{{train,val,test}}.csv")
    return summary

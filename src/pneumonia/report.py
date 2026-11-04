"""`pneumonia report`: render results.csv / metrics.json into the markdown docs.

Replaces the blocks between `<!-- AUTO:key -->` and `<!-- /AUTO:key -->` in README.md,
report.md and docs/model_card.md, so every number in the docs comes from code.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

DOCS = ("README.md", "report.md", "docs/model_card.md")
DASH = "—"


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    return f"{x:.{nd}f}"


def _i(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    return str(int(x))


def _ci(ci: dict | None, key: str, point) -> str:
    if not ci or key not in ci or ci[key].get("lo") is None:
        return _f(point)
    c = ci[key]
    return f"{_f(point)} [{_f(c['lo'])}, {_f(c['hi'])}]"


def results_table(res: pd.DataFrame, metrics: dict, split: str = "test") -> str:
    models = metrics.get("models", {}) if metrics else {}
    d = res[res["split"] == split]
    hdr = (
        "| Model | Threshold | Recall [95% CI] | Specificity [95% CI] | False alarms (FP) | "
        "Missed (FN) | PPV | NPV | F2 | Bal. acc. | AUROC | AUPRC |"
    )
    lines = [hdr, "|" + "---|" * 12]
    notes = []
    for _, r in d.iterrows():
        ci = None
        mm = models.get(r["model"], {})
        # the bootstrap CIs in metrics.json are computed on the test split only
        if r["kind"] == "model" and mm.get("status") == "evaluated" and split == "test":
            ci = (
                mm["test_ci_at_shipped"]
                if r["threshold_rule"].startswith("val-tuned")
                else mm.get("test_ci_at_0.5")
            )
        thr = (
            r["threshold_rule"]
            if pd.isna(r["threshold"]) or r["threshold_rule"] == "constant"
            else f"{r['threshold_rule']}: {r['threshold']:.3f}"
        )
        name = f"**{r['model']}**" if r["kind"] == "model" else f"*{r['model']}* (baseline)"
        if isinstance(r.get("note"), str) and r["note"]:
            notes.append(f"{r['model']}: {r['note']}")
            name += " ¹"
        lines.append(
            f"| {name} | {thr} | {_ci(ci, 'recall', r['recall'])} | "
            f"{_ci(ci, 'specificity', r['specificity'])} | {_i(r['fp'])} | {_i(r['fn'])} | "
            f"{_f(r['precision'])} | {_f(r['npv'])} | {_f(r['f2'])} | "
            f"{_f(r['balanced_accuracy'])} | {_f(r['auroc'])} | {_f(r['auprc'])} |"
        )
    n = d["n"].dropna()
    if len(n):
        r0 = d.dropna(subset=["n"]).iloc[0]
        lines.append("")
        lines.append(
            f"{split.capitalize()} split: {int(r0['n'])} images ({int(r0['n_pos'])} "
            f"PNEUMONIA, {int(r0['n_neg'])} NORMAL). NPV is undefined (—) when the "
            "model never predicts NORMAL."
        )
    for nt in dict.fromkeys(notes):
        lines.append(f"\n¹ {nt}")
    return "\n".join(lines)


def sensitivity_table(metrics: dict) -> str:
    p = metrics["primary_model"]
    mm = metrics["models"].get(p, {})
    if mm.get("status") != "evaluated":
        return f"_{p} not evaluated yet - run `pneumonia train` and `pneumonia evaluate`._"
    lines = [
        "| Recall target (val) | Threshold | Val recall | Val specificity | Test recall | "
        "Test specificity | Test FP | Test FN |",
        "|" + "---|" * 8,
    ]
    for s in mm["recall_target_sensitivity"]:
        lines.append(
            f"| {s['target_recall']:.2f} | {_f(s['threshold'])} | {_f(s['val_recall'])} | "
            f"{_f(s['val_specificity'])} | {_f(s['test_recall'])} | "
            f"{_f(s['test_specificity'])} | {_i(s['test_fp'])} | {_i(s['test_fn'])} |"
        )
    return "\n".join(lines)


def split_table(summary: dict | None) -> str:
    if not summary:
        return "_Run `pneumonia prepare` to generate the split table._"
    df = pd.DataFrame(summary["split_counts"])
    lines = [
        "| Split | NORMAL images (patients) | PNEUMONIA images (patients) | Total images | "
        "Distinct patients |",
        "|---|---|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        g = df[df["split"] == s].set_index("class_name")
        lines.append(
            f"| {s} | {g.loc['NORMAL', 'images']} ({g.loc['NORMAL', 'patients']}) | "
            f"{g.loc['PNEUMONIA', 'images']} ({g.loc['PNEUMONIA', 'patients']}) | "
            f"{g.loc['ALL', 'images']} | {g.loc['ALL', 'patients']} |"
        )
    info = summary["split_info"]
    lines.append("")
    lines.append(
        f"Pooled official train+val re-split by patient with `StratifiedGroupKFold` "
        f"(seed {info['seed']}, val fraction {info['val_fraction']}); official test "
        f"untouched. Patient ids shared between the test folder and train/val: "
        f"{info['test_trainval_shared_patient_ids']}; train/val images dropped to keep "
        f"the test patients unseen: {info['dropped_trainval_images_for_test_overlap']}. "
        f"Unparsed file names: {summary['n_unparsed_names']}."
    )
    if summary.get("shortcut_probe_val_auroc"):
        p = summary["shortcut_probe_val_auroc"]
        lines.append("")
        lines.append(
            f"Metadata-only shortcut probe (logistic regression, val AUROC; 0.5 = no "
            f"signal): image size only **{p['size_only']}**, border intensity only "
            f"**{p['border_intensity_only']}**, both **{p['size_and_border']}**."
        )
    return "\n".join(lines)


def headline(metrics: dict) -> str:
    p = metrics["primary_model"]
    mm = metrics["models"].get(p, {})
    if mm.get("status") != "evaluated":
        return "_Headline pending: no trained checkpoint was evaluated yet._"
    t = mm["test_at_shipped"]
    ci = mm["test_ci_at_shipped"]
    return (
        f"**{p}**, threshold {mm['shipped_threshold']:.3f} chosen on val for recall >= "
        f"{metrics['target_recall']:.2f}: test recall **{t['recall']:.3f}** "
        f"[{ci['recall']['lo']:.3f}, {ci['recall']['hi']:.3f}] at specificity "
        f"**{t['specificity']:.3f}** [{ci['specificity']['lo']:.3f}, "
        f"{ci['specificity']['hi']:.3f}] - missed **{t['fn']} of {t['n_pos']}** pneumonia "
        f"cases, **{t['fp']} false alarms** on {t['n_neg']} normal films. The always-PNEUMONIA "
        f"baseline scores recall 1.000 at specificity 0.000 ({t['n_neg']} false alarms)."
    )


def resume_line(metrics: dict) -> str:
    p = metrics["primary_model"]
    mm = metrics["models"].get(p, {})
    if mm.get("status") != "evaluated":
        return "_Fill after the run: recall [X]% at specificity [Y]%._"
    t = mm["test_at_shipped"]
    return (
        f"> Fine-tuned an ImageNet-pretrained DenseNet121 to detect pneumonia in 5.8k chest "
        f"radiographs under patient-grouped splits, tuning the decision threshold for "
        f"{100 * t['recall']:.1f}% recall at {100 * t['specificity']:.1f}% specificity on the "
        f"held-out test set against an always-positive baseline (recall 100%, specificity "
        f"0%), and used Grad-CAM to audit whether the model attends to lung fields rather "
        f"than acquisition artefacts."
    )


def error_block(metrics: dict) -> str:
    ea = metrics.get("error_analysis")
    if not ea:
        return "_Pending evaluation._"
    mr = ", ".join(f"{k} {v:.3f}" for k, v in ea["miss_rate_by_subtype"].items())
    return (
        f"- False negatives: **{ea['n_false_negatives']} of {ea['n_positives']}** test "
        f"pneumonias. Subtypes among misses: {ea['fn_subtype_counts']} (all positives: "
        f"{ea['positives_subtype_counts']}).\n"
        f"- Miss rate by subtype: {mr}. Viral share among misses "
        f"{_f(ea['viral_share_fn'])} vs {_f(ea['viral_share_all_positives'])} among all "
        f"positives.\n"
        f"- Exposure: mean pixel intensity of misses {_f(ea['mean_intensity_fn'])} vs "
        f"{_f(ea['mean_intensity_tp'])} for true positives; contrast (pixel std) "
        f"{_f(ea['contrast_std_fn'])} vs {_f(ea['contrast_std_tp'])}.\n"
        f"- Lowest-scored misses (the confidently wrong ones): "
        f"{[round(x, 4) for x in ea['fn_lowest_scores']]}. "
        f"Full list: `results/false_negatives.csv`."
    )


def gradcam_block(results_dir: Path) -> str:
    sp, ap = results_dir / "gradcam_summary.json", results_dir / "gradcam_audit.csv"
    if not sp.exists():
        return "_Run `pneumonia gradcam --per-case 3` to produce the panel and audit sheet._"
    s = json.loads(sp.read_text())
    a = pd.read_csv(ap).fillna("")
    cats = ["lung", "outside", "mixed", "empty"]
    lines = [
        "| Case type | Maps | Auto: off-anatomy? | Auto: empty | By eye: lung | By eye: outside "
        "| By eye: mixed | By eye: empty | Not reviewed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def row(label, g):
        ml = g["manual_location"].str.lower().str.strip()
        counts = " | ".join(str((ml == c).sum()) for c in cats)
        return (
            f"| {label} | {len(g)} | {(g['auto_flag'] == 'off-anatomy?').sum()} | "
            f"{(g['auto_flag'] == 'empty').sum()} | {counts} | {(~ml.isin(cats)).sum()} |"
        )

    order = ["True positives", "False negatives", "False positives", "True negatives"]
    for ct in order + [c for c in a["case_type"].unique() if c not in order]:
        lines.append(row(ct, a[a["case_type"] == ct]))
    lines.append(row("**All**", a))
    lines.append("")
    lines.append(
        f"Auto-flag off-anatomy? = CAM peak in the outer {int(100 * 0.12)}% band, or >60% of CAM "
        f"mass there (a spatially uniform map puts {s['uniform_map_border_mass']:.0%} there); "
        "empty = no positive PNEUMONIA evidence anywhere (expected for confident negatives). "
        f"Mean border mass over the sampled maps: {s['mean_border_mass']:.3f}. Manual "
        "columns come from `results/gradcam_audit.csv`, filled by eye."
    )
    return "\n".join(lines)


def versions_block(metrics: dict) -> str:
    v = metrics.get("versions", {})
    return (
        f"Generated {metrics.get('generated_at')} (git {metrics.get('git_commit')}), seed "
        f"{metrics.get('seed')}; "
        + ", ".join(
            f"{k} {v[k]}"
            for k in ("python", "torch", "torchvision", "numpy", "scikit-learn")
            if k in v
        )
        + (f"; GPU {v['gpu']}" if "gpu" in v else "; CPU only")
        + "."
    )


def _inject(text: str, key: str, body: str) -> str:
    pat = re.compile(rf"(<!-- AUTO:{key} -->)(.*?)(<!-- /AUTO:{key} -->)", re.S)
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def build_report(
    results_dir: str | Path = "results",
    splits_dir: str | Path = "splits",
    repo_root: str | Path = ".",
) -> dict[str, str]:
    results_dir, repo_root = Path(results_dir), Path(repo_root)
    blocks: dict[str, str] = {}
    summ_p = Path(splits_dir) / "split_summary.json"
    blocks["split_table"] = split_table(json.loads(summ_p.read_text()) if summ_p.exists() else None)
    mp, rp = results_dir / "metrics.json", results_dir / "results.csv"
    if mp.exists() and rp.exists():
        metrics, res = json.loads(mp.read_text()), pd.read_csv(rp)
        blocks["results_table"] = results_table(res, metrics, "test")
        blocks["val_table"] = results_table(res, metrics, "val")
        blocks["sensitivity_table"] = sensitivity_table(metrics)
        blocks["headline"] = headline(metrics)
        blocks["resume"] = resume_line(metrics)
        blocks["error_analysis"] = error_block(metrics)
        blocks["versions"] = versions_block(metrics)
    blocks["gradcam_audit"] = gradcam_block(results_dir)
    for doc in DOCS:
        p = repo_root / doc
        if not p.exists():
            continue
        t = p.read_text()
        for k, b in blocks.items():
            t = _inject(t, k, b)
        p.write_text(t)
        print(f"[report] updated {p}")
    (results_dir / "tables.md").parent.mkdir(parents=True, exist_ok=True)
    (results_dir / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items()))
    return blocks

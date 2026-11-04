"""Publication-style figures (labelled axes, readable at column width, 200 dpi PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = {"PNEUMONIA": PALETTE[0], "NORMAL": PALETTE[1]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASELINE_GREY = "#8a8984"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def class_balance(official_counts, split_counts, path) -> Path:
    """Two panels: official Kaggle folders vs our patient-grouped re-split.

    Each *_counts is a DataFrame indexed by split with images_NORMAL/images_PNEUMONIA and
    patients_NORMAL/patients_PNEUMONIA columns.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (title, t) in zip(
        axes,
        [
            ("Official Kaggle folders", official_counts),
            ("Patient-grouped splits (used)", split_counts),
        ],
    ):
        order = [s for s in ("train", "val", "test") if s in t.index]
        x = np.arange(len(order))
        w = 0.38
        for k, cls in enumerate(("NORMAL", "PNEUMONIA")):
            vals = [int(t.loc[s, f"images_{cls}"]) for s in order]
            pats = [int(t.loc[s, f"patients_{cls}"]) for s in order]
            bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=CLASS_COLORS[cls], label=cls)
            for b, v, p in zip(bars, vals, pats):
                ax.annotate(
                    f"{v}\n({p} pts)",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK2,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        ax.set_xticks(x, order)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Images")
    axes[0].legend(loc="upper right")
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, ymax * 1.18)
    return _save(fig, path)


def image_sizes(sizes_df, path) -> Path:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for cls, c in CLASS_COLORS.items():
        d = sizes_df[sizes_df["class_name"] == cls]
        axes[0].scatter(d["width"], d["height"], s=4, alpha=0.35, color=c, label=cls, lw=0)
        axes[1].hist(d["aspect"], bins=50, alpha=0.6, color=c, label=cls)
    axes[0].set(xlabel="Width (px)", ylabel="Height (px)", title="Original image size")
    axes[1].set(xlabel="Aspect ratio (w / h)", ylabel="Images", title="Aspect ratio by class")
    axes[0].legend(markerscale=4)
    return _save(fig, path)


def training_curves(history, path, title: str = "") -> Path:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    e = history["epoch"]
    axes[0].plot(e, history["train_loss"], color=PALETTE[0], label="train")
    axes[0].plot(e, history["val_loss"], color=PALETTE[1], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Loss")
    axes[1].plot(e, history["val_auroc"], color=PALETTE[0], label="AUROC")
    axes[1].plot(e, history["val_auprc"], color=PALETTE[2], label="AUPRC")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Val threshold-free")
    axes[2].plot(e, history["val_specificity"], color=PALETTE[0], label="specificity @ target")
    axes[2].plot(e, history["val_recall"], color=PALETTE[1], ls="--", label="recall (constrained)")
    axes[2].set(xlabel="Epoch", ylabel="Rate", title="Val operating point (selection metric)")
    best = history.loc[history["val_specificity"].idxmax(), "epoch"] if len(history) else None
    p2 = history[history["phase"] == "phase2_full"]
    for ax in axes:
        if len(p2) and p2["epoch"].min() > 1:
            ax.axvline(p2["epoch"].min() - 0.5, color=BASELINE_GREY, lw=1, ls=":")
        ax.legend(fontsize=8)
    if best is not None:
        axes[2].axvline(best, color=INK2, lw=1)
        axes[2].annotate(
            "selected",
            (best, 0.02),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
        )
    if title:
        fig.suptitle(title, y=1.03)
    return _save(fig, path)


def roc(curves: dict, path) -> Path:
    """curves: name -> (y_true, scores)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=BASELINE_GREY, lw=1, ls="--", label="chance (AUROC 0.50)")
    for k, (name, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        from sklearn.metrics import roc_auc_score

        ax.plot(
            fpr, tpr, color=PALETTE[k % len(PALETTE)], label=f"{name} ({roc_auc_score(y, s):.3f})"
        )
    ax.set(
        xlabel="False positive rate (1 - specificity)",
        ylabel="Recall (sensitivity)",
        title="ROC - test set",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def precision_recall(curves: dict, path) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    prevalence = None
    for k, (name, (y, s)) in enumerate(curves.items()):
        p, r, _ = precision_recall_curve(y, s)
        from sklearn.metrics import average_precision_score

        ax.plot(
            r,
            p,
            color=PALETTE[k % len(PALETTE)],
            label=f"{name} ({average_precision_score(y, s):.3f})",
        )
        prevalence = float(np.mean(y))
    if prevalence is not None:
        ax.axhline(
            prevalence,
            color=BASELINE_GREY,
            lw=1,
            ls="--",
            label=f"prevalence = always PNEUMONIA ({prevalence:.3f})",
        )
    ax.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision (PPV)",
        title="Precision-recall - test",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
    )
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, path)


def confusion(c: dict, path, title: str) -> Path:
    setup_style()
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.25)
    ax.grid(False)
    names = [["TN", "FP (false alarm)"], ["FN (missed)", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{m[i, j]}\n{names[i][j]}",
                ha="center",
                va="center",
                fontsize=10,
                color=INK,
                fontweight="bold" if (i, j) == (1, 0) else "normal",
            )
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    return _save(fig, path)


def operating_curve(curve: dict, chosen: dict, path, test_point: dict | None = None) -> Path:
    """Recall / specificity / precision vs threshold (val), shipped threshold marked."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    t = curve["threshold"]
    ax.plot(t, curve["recall"], color=PALETTE[0], label="recall (val)")
    ax.plot(t, curve["specificity"], color=PALETTE[1], label="specificity (val)")
    ax.plot(t, curve["precision"], color=PALETTE[2], label="precision (val)")
    thr = chosen["threshold"]
    ax.axvline(thr, color=INK, lw=1)
    ax.axhline(chosen["target_recall"], color=BASELINE_GREY, lw=1, ls=":")
    ax.annotate(
        f"shipped threshold {thr:.3f}\nval recall {chosen['recall']:.3f}, "
        f"spec {chosen['specificity']:.3f}",
        (thr, 0.05),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    ax.text(
        0.99,
        chosen["target_recall"],
        f"recall target {chosen['target_recall']:.2f}",
        ha="right",
        va="top",
        fontsize=8,
        color=INK2,
        transform=ax.get_yaxis_transform(),
    )
    if test_point is not None:
        ax.plot(
            [thr],
            [test_point["recall"]],
            "o",
            ms=8,
            mfc="white",
            mec=PALETTE[0],
            mew=2,
            label=f"test recall {test_point['recall']:.3f}",
        )
        ax.plot(
            [thr],
            [test_point["specificity"]],
            "o",
            ms=8,
            mfc="white",
            mec=PALETTE[1],
            mew=2,
            label=f"test specificity {test_point['specificity']:.3f}",
        )
    ax.set(
        xlabel="Decision threshold on P(PNEUMONIA)",
        ylabel="Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Operating curve",
    )
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def score_histogram(y, s, threshold: float, path, title="Test scores by true class") -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bins = np.linspace(0, 1, 41)
    for cls, lab in (("NORMAL", 0), ("PNEUMONIA", 1)):
        ax.hist(
            np.asarray(s)[np.asarray(y) == lab],
            bins=bins,
            alpha=0.6,
            color=CLASS_COLORS[cls],
            label=f"true {cls}",
        )
    ax.axvline(threshold, color=INK, lw=1.2)
    ax.annotate(
        f"threshold {threshold:.3f}",
        (threshold, ax.get_ylim()[1] * 0.92),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set(xlabel="Predicted P(PNEUMONIA)", ylabel="Images (log scale)", title=title)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper center")
    return _save(fig, path)


def gradcam_panel(groups: dict, path, title: str) -> Path:
    """groups: case_type -> list of dicts(image HxW [0,1], cam HxW [0,1], prob, caption).

    Layout: case-type blocks in a 2x2 grid; inside each block, original on top and
    the Grad-CAM overlay beneath, one column per case.
    """
    setup_style()
    types = list(groups)
    per = max(len(v) for v in groups.values()) if groups else 1
    nrows_blocks = (len(types) + 1) // 2
    fig = plt.figure(figsize=(2.05 * per * 2 + 0.4, 4.6 * nrows_blocks))
    outer = fig.add_gridspec(nrows_blocks, 2, wspace=0.08, hspace=0.32)
    for b, ct in enumerate(types):
        inner = outer[b // 2, b % 2].subgridspec(2, per, wspace=0.04, hspace=0.04)
        cases = groups[ct]
        for j in range(per):
            ax_o, ax_c = fig.add_subplot(inner[0, j]), fig.add_subplot(inner[1, j])
            for ax in (ax_o, ax_c):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if j >= len(cases):
                ax_o.set_visible(False)
                ax_c.set_visible(False)
                if j == 0:  # an empty case type is a finding, not a blank: say so
                    ax_o.set_visible(True)
                    ax_o.axis("off")
                    ax_o.text(
                        0.0,
                        0.5,
                        "none at this threshold",
                        fontsize=10,
                        color=INK2,
                        transform=ax_o.transAxes,
                        va="center",
                    )
                continue
            c = cases[j]
            ax_o.imshow(c["image"], cmap="gray", vmin=0, vmax=1)
            ax_c.imshow(c["image"], cmap="gray", vmin=0, vmax=1)
            ax_c.imshow(c["cam"], cmap="jet", alpha=0.42, vmin=0, vmax=1)
            ax_o.set_title(f"p={c['prob']:.3f}", fontsize=9, pad=3)
            ax_c.set_xlabel(c.get("caption", ""), fontsize=7, color=INK2, labelpad=2)
            if j == 0:
                ax_o.set_ylabel("original", fontsize=8, color=INK2)
                ax_c.set_ylabel("Grad-CAM", fontsize=8, color=INK2)
        # block title
        bbox = outer[b // 2, b % 2].get_position(fig)
        n = len(groups[ct])
        fig.text(
            bbox.x0,
            bbox.y1 + 0.045 / nrows_blocks,
            f"{ct} (n shown = {n})",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.suptitle(title, fontsize=12, y=1.0 + 0.04 / nrows_blocks)
    return _save(fig, path)

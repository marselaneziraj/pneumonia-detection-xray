"""Threshold-dependent and threshold-free metrics, threshold selection and bootstrap CIs.

Convention: label 1 = PNEUMONIA (positive), predict positive iff score >= threshold.
Undefined ratios (0/0) are returned as NaN, never silently as 0 or 1.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def recall(c: dict) -> float:
    return _div(c["tp"], c["tp"] + c["fn"])


def specificity(c: dict) -> float:
    return _div(c["tn"], c["tn"] + c["fp"])


def precision(c: dict) -> float:
    return _div(c["tp"], c["tp"] + c["fp"])


def npv(c: dict) -> float:
    return _div(c["tn"], c["tn"] + c["fn"])


def fbeta(c: dict, beta: float = 2.0) -> float:
    """F-beta from counts: (1+b^2) TP / ((1+b^2) TP + b^2 FN + FP). beta=2 weights recall 4x."""
    b2 = beta * beta
    return _div((1 + b2) * c["tp"], (1 + b2) * c["tp"] + b2 * c["fn"] + c["fp"])


def balanced_accuracy(c: dict) -> float:
    r, s = recall(c), specificity(c)
    return (r + s) / 2 if not (math.isnan(r) or math.isnan(s)) else float("nan")


def threshold_free(y_true, scores) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def binary_metrics(y_true, scores, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    c = confusion_counts(y_true, scores >= threshold)
    out = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "recall": recall(c),
        "specificity": specificity(c),
        "precision": precision(c),
        "npv": npv(c),
        "f2": fbeta(c, 2.0),
        "balanced_accuracy": balanced_accuracy(c),
        **threshold_free(y_true, scores),
        **c,
    }
    return out


def choose_threshold(y_true, scores, target_recall: float) -> dict[str, float]:
    """Highest-specificity threshold subject to recall >= target_recall.

    Candidates are the distinct scores (predict positive iff score >= t). Ties in
    specificity break towards the higher threshold. The returned threshold is moved to
    the midpoint between the chosen score and the next lower distinct score: on the
    tuning data the predictions are identical, but a test score that lands just below
    the chosen val score is not lost to a knife-edge.
    A feasible threshold always exists (the minimum score gives recall 1.0).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        raise ValueError("choose_threshold needs at least one positive example")
    # Recall only rises and specificity only falls as the threshold drops, so scanning the
    # distinct scores from high to low, the first one meeting the recall target is the
    # highest-specificity feasible threshold (and the highest threshold among ties).
    cand = np.unique(scores)[::-1]
    pos = y_true == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    best = None
    for i, t in enumerate(cand):
        pred = scores >= t
        r = (pred & pos).sum() / n_pos
        if r + 1e-12 < target_recall:
            continue
        lower = cand[i + 1] if i + 1 < len(cand) else None
        best = {
            "raw_threshold": float(t),
            "threshold": float((t + lower) / 2) if lower is not None else float(t),
            "recall": float(r),
            "specificity": float(((~pred) & neg).sum() / n_neg) if n_neg else float("nan"),
        }
        break
    assert best is not None  # the lowest score always gives recall 1.0
    best["target_recall"] = float(target_recall)
    return best


def operating_curve(y_true, scores, n: int = 501) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    ts = np.linspace(0, 1, n)
    rec, spec, prec = [], [], []
    for t in ts:
        c = confusion_counts(y_true, scores >= t)
        rec.append(recall(c))
        spec.append(specificity(c))
        prec.append(precision(c))
    return {
        "threshold": ts,
        "recall": np.array(rec),
        "specificity": np.array(spec),
        "precision": np.array(prec),
    }


def bootstrap_ci(
    y_true,
    y_pred,
    metrics=("recall", "specificity", "precision", "npv"),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs at a fixed threshold, resampling images with replacement.

    Resampling is stratified by class so every replicate keeps the observed prevalence and
    recall/specificity are always defined.
    """
    fns = {
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f2": fbeta,
        "balanced_accuracy": balanced_accuracy,
    }
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))])
        c = confusion_counts(y_true[idx], y_pred[idx])
        for m in metrics:
            draws[m].append(fns[m](c))
    point = confusion_counts(y_true, y_pred)
    out = {}
    for m in metrics:
        arr = np.array(draws[m], dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if len(arr) else (np.nan, np.nan)
        out[m] = {"point": fns[m](point), "lo": float(lo), "hi": float(hi)}
    out["_meta"] = {"n_boot": n_boot, "alpha": alpha, "seed": seed, "stratified": True}
    return out


# ----------------------------------------------------------------------------- baselines
def baseline_scores(name: str, n: int, seed: int = 0) -> np.ndarray:
    if name == "always PNEUMONIA":
        return np.ones(n)
    if name == "always NORMAL":
        return np.zeros(n)
    if name == "random score":
        return np.random.default_rng(seed).uniform(size=n)
    raise ValueError(name)


BASELINES = ("always PNEUMONIA", "always NORMAL", "random score")

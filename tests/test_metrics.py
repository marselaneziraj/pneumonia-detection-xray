import math

import numpy as np
import pytest

from pneumonia.metrics import (
    balanced_accuracy,
    baseline_scores,
    binary_metrics,
    bootstrap_ci,
    choose_threshold,
    confusion_counts,
    fbeta,
    npv,
    precision,
    recall,
    specificity,
)


def test_confusion_hand_computed():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    p = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    c = confusion_counts(y, p)
    assert c == {"tp": 4, "fp": 2, "tn": 3, "fn": 1}
    assert recall(c) == pytest.approx(4 / 5)
    assert specificity(c) == pytest.approx(3 / 5)
    assert precision(c) == pytest.approx(4 / 6)
    assert npv(c) == pytest.approx(3 / 4)
    assert balanced_accuracy(c) == pytest.approx(0.7)


def test_fbeta_hand_computed():
    c = {"tp": 4, "fp": 2, "tn": 3, "fn": 1}
    # F2 = 5*TP / (5*TP + 4*FN + FP) = 20 / (20 + 4 + 2)
    assert fbeta(c, 2.0) == pytest.approx(20 / 26)
    # F1 = 2PR/(P+R) with P=2/3, R=4/5
    p, r = 2 / 3, 4 / 5
    assert fbeta(c, 1.0) == pytest.approx(2 * p * r / (p + r))
    # F2 equals the general formula (1+b^2)PR/(b^2 P + R)
    assert fbeta(c, 2.0) == pytest.approx(5 * p * r / (4 * p + r))


def test_always_pneumonia_baseline_is_recall_1_spec_0():
    y = np.array([1] * 390 + [0] * 234)  # official test prevalence
    m = binary_metrics(y, baseline_scores("always PNEUMONIA", len(y)), 0.5)
    assert m["recall"] == 1.0 and m["specificity"] == 0.0
    assert m["fp"] == 234 and m["fn"] == 0
    assert m["precision"] == pytest.approx(390 / 624)
    assert math.isnan(m["npv"])  # never predicts NORMAL: NPV undefined, not 0 or 1
    assert m["auroc"] == pytest.approx(0.5)


def test_always_normal_baseline():
    y = np.array([1, 1, 0, 0, 0])
    m = binary_metrics(y, baseline_scores("always NORMAL", 5), 0.5)
    assert m["recall"] == 0.0 and m["specificity"] == 1.0


def _brute_force(y, s, target):
    best = None
    for t in np.unique(s):
        c = confusion_counts(y, s >= t)
        if recall(c) >= target and (best is None or specificity(c) > best[1]):
            best = (t, specificity(c))
    return best


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("target", [0.9, 0.95, 0.98, 0.99, 1.0])
def test_choose_threshold_meets_target_and_maximises_specificity(seed, target):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=300) < 0.7).astype(int)
    s = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1).round(3)  # ties included
    op = choose_threshold(y, s, target)
    c = confusion_counts(y, s >= op["threshold"])
    assert recall(c) >= target
    assert specificity(c) == pytest.approx(op["specificity"])
    assert op["specificity"] == pytest.approx(_brute_force(y, s, target)[1])


def test_choose_threshold_midpoint_keeps_tuning_predictions():
    y = np.array([0, 0, 1, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    op = choose_threshold(y, s, 1.0)
    assert op["raw_threshold"] == 0.3 and op["threshold"] == pytest.approx(0.25)
    assert op["specificity"] == pytest.approx(2 / 3)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=624) < 0.625).astype(int)
    p = np.where(y == 1, rng.uniform(size=624) < 0.97, rng.uniform(size=624) < 0.3).astype(int)
    ci = bootstrap_ci(y, p, n_boot=500, seed=0)
    for k in ("recall", "specificity"):
        assert ci[k]["lo"] <= ci[k]["point"] <= ci[k]["hi"]
        assert ci[k]["hi"] - ci[k]["lo"] < 0.2

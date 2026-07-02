"""Tests for per-subgroup fairness metrics (sentinel.fairness.metrics)."""

import numpy as np

from sentinel.fairness import metrics


def test_confusion_at_threshold_hand_computed():
    #            prob:  0.9   0.4   0.6   0.1
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    c = metrics.confusion_at_threshold(y, p, 0.5)
    # flagged (>=0.5): idx0 (y=1 -> TP), idx2 (y=0 -> FP). not flagged: idx1 (TN), idx3 (FN).
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1, 1, 1, 1)
    assert c["tpr"] == 0.5  # 1 / (1 TP + 1 FN)
    assert c["fpr"] == 0.5  # 1 / (1 FP + 1 TN)
    assert c["selection_rate"] == 0.5  # 2 flagged of 4


def test_is_powered_boundary_inclusive():
    assert metrics.is_powered({"n": 100, "positives": 30}) is True
    assert metrics.is_powered({"n": 99, "positives": 30}) is False
    assert metrics.is_powered({"n": 100, "positives": 29}) is False


def test_subgroup_metrics_single_class_returns_none_auroc():
    y = np.zeros(50, dtype=int)  # single class -> AUROC/AUPRC undefined
    p = np.linspace(0.01, 0.5, 50)
    m = metrics.subgroup_metrics(y, p, 0.2)
    assert m["auroc"] is None and m["auprc"] is None
    assert m["positives"] == 0
    assert m["brier"] is not None  # Brier still defined
    assert m["ece"] is not None


def test_subgroup_metrics_two_class_has_auroc():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 400)
    p = np.clip(0.3 + 0.3 * (y - 0.3) + rng.normal(0, 0.1, 400), 0, 1)
    m = metrics.subgroup_metrics(y, p, 0.3)
    assert 0.5 <= m["auroc"] <= 1.0


def test_pairwise_and_four_fifths():
    a = {"tpr": 0.6, "fpr": 0.2}
    b = {"tpr": 0.5, "fpr": 0.0}
    assert abs(metrics.pairwise_difference(a, b, "tpr") - 0.1) < 1e-12
    assert abs(metrics.four_fifths_ratio(a, b, "tpr") - 1.2) < 1e-12
    assert np.isnan(metrics.four_fifths_ratio(a, b, "fpr"))  # zero denominator -> nan
    assert metrics.pairwise_difference({"auroc": None}, a, "auroc") is None

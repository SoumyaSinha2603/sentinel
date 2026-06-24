"""Tests for the evaluation metrics layer (synthetic arrays only — no model fits)."""

import numpy as np

from sentinel.evaluation import metrics


def test_perfect_and_random_auroc():
    # Perfectly separated: all negatives score below all positives.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert metrics.discrimination_metrics(y_true, y_prob_perfect)["auroc"] == 1.0

    # Random scores -> AUROC near 0.5.
    rng = np.random.default_rng(0)
    y_rand = rng.integers(0, 2, size=2000)
    p_rand = rng.random(2000)
    auroc = metrics.discrimination_metrics(y_rand, p_rand)["auroc"]
    assert abs(auroc - 0.5) < 0.05


def test_brier_and_ece_bounds():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=1000)
    p = rng.random(1000)
    disc = metrics.discrimination_metrics(y, p)
    cal = metrics.calibration_metrics(y, p)
    assert 0.0 <= disc["brier"] <= 1.0
    assert 0.0 <= cal["ece"] <= 1.0

    # A well-calibrated toy case: predicted prob equals true class probability.
    # Group A (p=0.2, base rate 0.2), group B (p=0.8, base rate 0.8).
    n = 5000
    y_cal = np.concatenate([rng.random(n) < 0.2, rng.random(n) < 0.8]).astype(int)
    p_cal = np.concatenate([np.full(n, 0.2), np.full(n, 0.8)])
    ece = metrics.calibration_metrics(y_cal, p_cal)["ece"]
    assert ece < 0.05


def test_summarize_keys():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.3, 0.6, 0.4, 0.8])
    out = metrics.summarize(y, p)
    for key in ("auroc", "auprc", "brier", "ece", "reliability", "n", "positives", "prevalence"):
        assert key in out


def test_constant_baseline_auroc_half():
    rng = np.random.default_rng(2)
    prevalence = 0.11
    y = (rng.random(5000) < prevalence).astype(int)
    p = np.full(len(y), y.mean())  # constant predictor = base rate
    out = metrics.summarize(y, p)
    assert out["auroc"] == 0.5
    # AUPRC of a constant predictor equals the prevalence.
    assert abs(out["auprc"] - out["prevalence"]) < 0.02

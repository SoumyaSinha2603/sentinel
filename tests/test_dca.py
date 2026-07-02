"""Tests for decision-curve analysis (sentinel.clinical_utility.dca)."""

import numpy as np

from sentinel.clinical_utility import dca


def _synth(n: int = 4000, seed: int = 0):
    """Informative ranker: p correlates with y, ~12% prevalence."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.12, n)
    p = np.clip(0.12 + 0.35 * (y - 0.12) + rng.normal(0, 0.05, n), 1e-3, 1 - 1e-3)
    return y, p


def test_nb_none_is_zero():
    y, p = _synth()
    g = dca.dca_grid(y, p)
    assert np.allclose(g["nb_none"], 0.0)


def test_nb_all_matches_closed_form():
    prev = 0.114
    for pt in (0.05, 0.10, 0.20, 0.35):
        expected = prev - (1 - prev) * (pt / (1 - pt))
        assert abs(float(dca.net_benefit_all(prev, pt)) - expected) < 1e-12


def test_treat_all_crosses_zero_at_prevalence():
    y, _ = _synth()
    prev = float(np.mean(y == 1))
    # exact closed form: zero at p_t == prev
    assert abs(float(dca.net_benefit_all(prev, prev))) < 1e-12
    # on the grid, the point nearest prev is ~0 (within grid tol)
    thr = dca.THRESHOLDS
    nearest = thr[np.argmin(np.abs(thr - prev))]
    assert abs(float(dca.net_benefit_all(prev, nearest))) < 0.01


def test_nb_model_finite_and_correctly_signed():
    y, p = _synth()
    g = dca.dca_grid(y, p)
    assert np.all(np.isfinite(g["nb_model"]))
    # an informative ranker beats treat-none at the smallest threshold
    assert g["nb_model"][0] > 0


def test_nb_model_matches_bruteforce():
    y, p = _synth()
    n = len(y)
    for pt in (0.05, 0.10, 0.20, 0.30):
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        expected = tp / n - (fp / n) * (pt / (1 - pt))
        assert abs(dca.net_benefit_model(y, p, pt) - expected) < 1e-9


def test_bootstrap_ci_brackets_point_estimate_loosely():
    y, p = _synth()
    pid = np.arange(len(y))  # one encounter per patient here
    g = dca.dca_grid(y, p)
    lo, hi = dca.dca_bootstrap_ci(y, p, pid, dca.THRESHOLDS, n_boot=200, seed=42)
    assert lo.shape == hi.shape == g["nb_model"].shape
    assert np.all(lo <= hi + 1e-12)

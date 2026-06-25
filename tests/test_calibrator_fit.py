"""Tests for the calibrator fit/apply/gate logic (sentinel.calibration.fit_calibrators).

Covers the two contractual properties — ECE drops post-calibration, and the manifest's
portable form round-trips against the fitted sklearn object — plus the gate and the
pre-registered selection rule. These are unit tests over synthetic scores: they do NOT
load @phase1 or touch MLflow.
"""

import numpy as np

from sentinel.calibration import fit_calibrators as fc
from sentinel.evaluation import metrics


def _miscalibrated(n: int = 5000, seed: int = 0):
    """Well-ranked but systematically OVERconfident scores (positive native ECE)."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.02, 0.40, n)
    y = rng.binomial(1, true_p)
    # p**0.4 > p for p in (0,1): scores sit above the true probability -> miscalibrated.
    scores = np.clip(true_p**0.4, 1e-4, 1 - 1e-4)
    return scores, y


def test_assert_env_returns_versions():
    v = fc.assert_env()
    assert v["sklearn"] == fc.REQUIRED_SKLEARN
    assert "lightgbm" in v


def test_isotonic_reduces_ece():
    scores, y = _miscalibrated()
    iso = fc.fit_isotonic(scores, y)
    before = metrics.calibration_metrics(y, scores)["ece"]
    after = metrics.calibration_metrics(y, fc.apply_calibrator("isotonic", iso, scores))["ece"]
    assert after < before


def test_platt_does_not_worsen_ece():
    scores, y = _miscalibrated()
    platt = fc.fit_platt(scores, y)
    before = metrics.calibration_metrics(y, scores)["ece"]
    after = metrics.calibration_metrics(y, fc.apply_calibrator("platt", platt, scores))["ece"]
    assert after <= before + 1e-9


def test_isotonic_portable_round_trip():
    scores, y = _miscalibrated()
    iso = fc.fit_isotonic(scores, y)
    portable = fc.isotonic_portable(iso)
    obj_pred = fc.apply_calibrator("isotonic", iso, scores)
    port_pred = fc.apply_portable("isotonic", portable, scores)
    assert np.allclose(obj_pred, port_pred, atol=1e-9)


def test_platt_portable_round_trip():
    scores, y = _miscalibrated()
    platt = fc.fit_platt(scores, y)
    portable = fc.platt_portable(platt)
    obj_pred = fc.apply_calibrator("platt", platt, scores)
    port_pred = fc.apply_portable("platt", portable, scores)
    assert np.allclose(obj_pred, port_pred, atol=1e-9)


def test_gate_pass_on_identical_distributions():
    rng = np.random.default_rng(1)
    d = rng.random(2000)
    gate = fc.score_gate(d, d.copy())
    assert gate["pass"]
    assert len(gate["decile_deltas"]) == 9
    assert gate["ks"] <= fc.GATE_KS_MAX


def test_gate_fail_on_shifted_distribution():
    rng = np.random.default_rng(2)
    d = rng.random(2000)
    gate = fc.score_gate(d, np.clip(d + 0.3, 0.0, 1.0))
    assert not gate["pass"]


def test_selection_prefers_lower_ece():
    summaries = {
        "isotonic": {"ece": 0.02, "brier": 0.10},
        "platt": {"ece": 0.05, "brier": 0.10},
    }
    monotone = {"isotonic": True, "platt": True}
    winner, rejected = fc.select_winner(summaries, monotone)
    assert winner == "isotonic"
    assert rejected == []


def test_selection_rejects_non_monotone_even_if_lower_ece():
    summaries = {
        "isotonic": {"ece": 0.02, "brier": 0.10},
        "platt": {"ece": 0.05, "brier": 0.10},
    }
    monotone = {"isotonic": False, "platt": True}
    winner, rejected = fc.select_winner(summaries, monotone)
    assert winner == "platt"
    assert "isotonic" in rejected

"""Tests for worklist ranking + the calibrated path (sentinel.clinical_utility)."""

import json

import joblib
import numpy as np
import pytest

from sentinel.clinical_utility import calibrated, ranking
from sentinel.clinical_utility.evaluate import OPERATING_POINTS_PATH


def _synth(n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.12, n)
    p_raw = np.clip(0.12 + 0.35 * (y - 0.12) + rng.normal(0, 0.05, n), 1e-4, 1 - 1e-4)
    return y, p_raw


def test_precision_equals_lift_times_prevalence():
    y, p = _synth()
    prev = float(np.mean(y == 1))
    order = ranking.worklist_order(p)
    for row in ranking.precision_recall_at_k(y, order):
        assert abs(row["precision"] - row["lift"] * prev) < 1e-9


def test_recall_monotone_non_decreasing_in_k():
    y, p = _synth()
    order = ranking.worklist_order(p)
    recalls = [row["recall"] for row in ranking.precision_recall_at_k(y, order)]
    assert all(recalls[i + 1] >= recalls[i] - 1e-12 for i in range(len(recalls) - 1))


def test_calibration_invariance_strictly_monotone():
    y, p_raw = _synth()
    p_cal = np.sqrt(p_raw)  # strictly monotone -> identical ranking
    assert ranking.precision_recall_invariant(y, p_cal, p_raw) < 1e-9


def test_calibration_invariance_with_ties():
    """Isotonic-like flats create ties; raw tiebreak keeps the worklist identical."""
    y, p_raw = _synth()
    p_cal = np.clip(np.round(p_raw * 5) / 5, 0.0, 1.0)  # heavy ties (flat regions)
    assert len(np.unique(p_cal)) < len(np.unique(p_raw))  # ties really exist
    assert ranking.precision_recall_invariant(y, p_cal, p_raw) < 1e-9


def test_implied_threshold_maps_back_to_k():
    y, p_raw = _synth()
    p_cal = np.sqrt(p_raw)
    order = ranking.worklist_order(p_cal, p_raw)
    n = len(y)
    for k in ranking.BUDGETS:
        thr = ranking.implied_threshold(p_cal, order, k)
        nf = int(round(k * n))
        # thr is exactly the k-th ranked calibrated probability
        assert p_cal[order][nf - 1] == thr
        # flagging p_cal >= thr selects at least the top-k
        assert int(np.sum(p_cal >= thr)) >= nf


def test_portable_calibrator_matches_committed_joblib():
    """get_calibrated_proba's portable map == the committed joblib (within 1e-12)."""
    iso = joblib.load(calibrated.CALIBRATOR_JOBLIB)
    manifest = calibrated.load_manifest()
    probe = np.linspace(0.0, 1.0, 1001)
    obj_pred = np.asarray(iso.predict(probe), dtype=float)
    portable_pred = calibrated.apply_calibrator_portable(manifest, probe)
    assert np.max(np.abs(obj_pred - portable_pred)) < 1e-12


def test_operating_points_json_round_trips():
    if not OPERATING_POINTS_PATH.exists():
        pytest.skip("operating_points.json not generated yet (run evaluate first)")
    data = json.loads(OPERATING_POINTS_PATH.read_text(encoding="utf-8"))
    # round-trips losslessly
    assert json.loads(json.dumps(data)) == data
    # required provenance + structure
    for key in (
        "operating_points",
        "k_grid_curve",
        "dca_grid",
        "prevalence",
        "s_eval_patient_sha256",
        "seed",
        "sklearn_version",
        "lightgbm_version",
        "git_commit",
        "surface_reuse_note",
    ):
        assert key in data
    budgets = [op["budget"] for op in data["operating_points"]]
    assert budgets == [0.05, 0.10, 0.20]
    # higher budget -> lower implied probability threshold (worklist reaches deeper)
    thr = [op["implied_threshold"] for op in data["operating_points"]]
    assert thr[0] >= thr[1] >= thr[2]

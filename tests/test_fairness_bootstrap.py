"""Tests for the patient-grouped fairness bootstrap (sentinel.fairness.bootstrap)."""

import numpy as np

from sentinel.fairness import bootstrap


def _synth(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, n)
    p = np.clip(0.2 + 0.4 * (y - 0.2) + rng.normal(0, 0.1, n), 1e-3, 1 - 1e-3)
    labels = np.where(np.arange(n) % 2 == 0, "A", "B")
    patient_ids = np.arange(n)  # one encounter per patient
    return y, p, patient_ids, labels


def test_determinism_same_seed_identical_ci():
    y, p, pid, labels = _synth()
    a = bootstrap.bootstrap_subgroup_metrics(y, p, pid, labels, 0.2, n_boot=200, seed=42)
    b = bootstrap.bootstrap_subgroup_metrics(y, p, pid, labels, 0.2, n_boot=200, seed=42)
    assert a == b


def test_pairwise_uses_same_resamples_as_subgroup_and_is_paired():
    y, p, pid, labels = _synth()
    # Same seed/n_boot in both entrypoints -> identical patient draws (module guarantee).
    diff = bootstrap.bootstrap_pairwise_diff(y, p, pid, labels, "A", "B", 0.2, n_boot=200, seed=42)
    for key in ("auroc", "tpr", "fpr", "ece"):
        lo, hi = diff[key]["ci"]
        assert lo <= hi
        assert diff[key]["n_eff"] <= 200


def test_nan_guard_drops_degenerate_draws_and_reports_effective_count():
    # Group B is single-class (all y=0) -> AUROC undefined every draw -> dropped.
    n = 600
    rng = np.random.default_rng(1)
    y = np.zeros(n, dtype=int)
    y[:300] = rng.binomial(1, 0.3, 300)  # only group A has positives
    p = np.clip(rng.random(n), 1e-3, 1 - 1e-3)
    labels = np.where(np.arange(n) < 300, "A", "B")
    pid = np.arange(n)
    res = bootstrap.bootstrap_subgroup_metrics(y, p, pid, labels, 0.3, n_boot=100, seed=42)
    assert res["B"]["n_eff"]["auroc"] == 0  # all draws degenerate -> dropped
    assert np.isnan(res["B"]["ci"]["auroc"][0])
    assert res["A"]["n_eff"]["auroc"] > 0  # group A stays estimable
    # selection_rate is always defined (no positives needed).
    assert res["B"]["n_eff"]["selection_rate"] == 100

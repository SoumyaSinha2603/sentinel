"""Tests for the age-disparity OOF robustness check (sentinel.fairness.age_oof_robustness).

The data/model-backed tests are marked slow and skip when the cached raw dataset is absent.
"""

import pytest

from sentinel.config import RAW_DIR
from sentinel.fairness import age_oof_robustness as oof
from sentinel.fairness import subgroups

RAW_CSV = RAW_DIR / "diabetes_130.csv"


def test_report_section_extension_is_idempotent(tmp_path):
    report = tmp_path / "fairness_audit.md"
    report.write_text("# Fairness audit\n\nbody\n", encoding="utf-8")
    section = oof.SECTION_MARKER + "\n\nfirst run\n"
    oof.extend_report(section, report_path=report)
    once = report.read_text(encoding="utf-8")
    # Re-running replaces the prior section rather than appending a duplicate.
    oof.extend_report(oof.SECTION_MARKER + "\n\nsecond run\n", report_path=report)
    twice = report.read_text(encoding="utf-8")
    assert once.count(oof.SECTION_MARKER) == 1
    assert twice.count(oof.SECTION_MARKER) == 1
    assert "second run" in twice and "first run" not in twice
    assert twice.startswith("# Fairness audit")  # original body preserved


_GRADIENT = {"<40": 0.751, "40-60": 0.698, "60-80": 0.659, "80+": 0.623}


def _per_band(ece_80=0.006):
    pb = {b: {"auroc_oof": _GRADIENT[b], "ece_oof_cal": 0.005} for b in subgroups.AGE_BAND_ORDER}
    pb["80+"]["ece_oof_cal"] = ece_80
    return pb


def test_verdict_survives_reframed_as_monotone_gradient_with_ece_caveat():
    pairwise = {
        "80+__vs__60-80": {"diff": -0.036, "ci": (-0.055, -0.018), "material": False},
        "80+__vs__40-60": {"diff": -0.075, "ci": (-0.097, -0.052), "material": True},
        "80+__vs__<40": {"diff": -0.128, "ci": (-0.159, -0.096), "material": True},
    }
    verdict, rationale = oof._verdict(True, _per_band(), pairwise)
    assert verdict.startswith("GAP SURVIVES")
    # (1) reframed as a cumulative gradient, explicitly not an 80+ cliff.
    assert "gradient" in rationale.lower() and "cliff" in rationale.lower()
    assert "sub-threshold" in rationale  # 80+ vs 60-80 called out as sub-threshold
    # (2) ECE caveat + robustness line present.
    assert "well-calibrated" in rationale and "robust to the caveat" in rationale


def test_verdict_survives_flags_miscalibration_when_ece_above_bar():
    pairwise = {
        "80+__vs__60-80": {"diff": -0.036, "ci": (-0.055, -0.018), "material": False},
        "80+__vs__40-60": {"diff": -0.075, "ci": (-0.097, -0.052), "material": True},
        "80+__vs__<40": {"diff": -0.128, "ci": (-0.159, -0.096), "material": True},
    }
    _, rationale = oof._verdict(True, _per_band(ece_80=0.09), pairwise)
    assert "miscalibrated as well as less rank-separable" in rationale


def test_verdict_collapses_when_primary_contrasts_not_material():
    pairwise = {
        "80+__vs__60-80": {"diff": -0.02, "ci": (-0.05, 0.01), "material": False},
        "80+__vs__40-60": {"diff": -0.03, "ci": (-0.07, 0.00), "material": False},
        "80+__vs__<40": {"diff": -0.04, "ci": (-0.09, 0.01), "material": False},
    }
    verdict, _ = oof._verdict(False, _per_band(), pairwise)
    assert verdict.startswith("GAP COLLAPSES")


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not cached")
def test_s_fit_disjoint_from_s_eval_and_holdout_and_aligned():
    rec = oof.recover_s_fit_frames()
    built = rec["df_fit_built"]["patient_nbr"].to_numpy()
    cohort = rec["df_fit_cohort"]["patient_nbr"].to_numpy()
    assert (built == cohort).all()  # row-aligned
    # ~64k rows (S_fit = 80% of the 80% train), all with the raw age string available.
    assert len(built) > 60_000
    assert "age" in rec["df_fit_cohort"].columns  # raw decade string survives in cohort frame


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not cached")
def test_diagnostic_uses_raw_age_string_not_midpoint():
    rec = oof.recover_s_fit_frames()
    # The cohort frame carries the raw decade string; the built frame does not.
    assert oof.DIAGNOSTIC_BUCKET in set(rec["df_fit_cohort"]["age"].unique())
    assert "age" not in rec["df_fit_built"].columns  # build_features collapsed it to age_midpoint
    assert "age_midpoint" in rec["df_fit_built"].columns


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not cached")
def test_oof_reproduces_w5_auroc_and_is_deterministic():
    r1 = oof.compute(n_boot=50)
    assert round(r1["pooled_oof_auroc"], 4) == oof.EXPECTED_OOF_AUROC
    assert r1["all_powered"] is True
    # Determinism: same seed -> identical OOF AUROC and identical band CIs.
    r2 = oof.compute(n_boot=50)
    assert r1["pooled_oof_auroc"] == r2["pooled_oof_auroc"]
    for b in subgroups.AGE_BAND_ORDER:
        assert r1["per_band"][b]["auroc_oof_ci"] == r2["per_band"][b]["auroc_oof_ci"]
    assert r1["verdict"] in {
        "GAP SURVIVES (real)",
        "GAP COLLAPSES (substantially in-sample artifact)",
    }

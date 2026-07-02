"""Tests for the fairness audit entrypoint (sentinel.fairness.audit).

Pure verdict-logic tests always run. The data/model-backed integration tests are marked
slow and skip cleanly when the cached raw dataset or the local MLflow registry is absent.
"""

import copy

import pytest

from sentinel.clinical_utility import calibrated
from sentinel.config import RAW_DIR, ROOT
from sentinel.fairness import audit

RAW_CSV = RAW_DIR / "diabetes_130.csv"
MLRUNS = ROOT / "mlruns"


def _attr(overall, tier):
    return {"tier": tier, "verdict": {"overall": overall}}


def test_payer_material_cannot_drive_project_verdict():
    per_attribute = {
        "race": _attr("PASS", "primary"),
        "gender": _attr("PASS", "primary"),
        "age": _attr("FLAG (monitor)", "primary"),
        "payer_code": _attr("MATERIAL DISPARITY", "exploratory"),
    }
    # payer_code is exploratory -> excluded; worst PRIMARY is age's FLAG.
    assert audit.project_verdict(per_attribute) == "FLAG (monitor)"


def test_primary_material_drives_project_verdict():
    per_attribute = {
        "race": _attr("MATERIAL DISPARITY", "primary"),
        "gender": _attr("PASS", "primary"),
    }
    assert audit.project_verdict(per_attribute) == "MATERIAL DISPARITY"


def test_classify_pair_metric_material_needs_both_conditions():
    # CI excludes 0 AND |diff| > practical threshold (0.05 for auroc) -> material.
    assert (
        audit._classify_pair_metric({"ci_excludes_zero": True, "diff": 0.08}, "auroc") == "material"
    )
    # Significant but sub-threshold -> flag, not material.
    assert audit._classify_pair_metric({"ci_excludes_zero": True, "diff": 0.02}, "auroc") == "flag"
    # Not significant -> pass regardless of magnitude.
    assert audit._classify_pair_metric({"ci_excludes_zero": False, "diff": 0.20}, "auroc") == "pass"


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not cached")
def test_recover_s_eval_matches_prereg_and_excludes_holdout():
    manifest = calibrated.load_manifest()
    rec = audit.recover_s_eval_frames(manifest)
    # Reproduced S_eval matches the pre-registered surface (prereg §0 provenance).
    assert rec["s_eval_hash"] == audit.S_EVAL_EXPECTED_HASH
    assert len(rec["df_eval_built"]) == 15_734
    assert len(rec["df_eval_cohort"]) == 15_734
    # Built (scored) frame and raw (sliced) frame are row-aligned by patient id.
    assert (
        rec["df_eval_built"]["patient_nbr"].to_numpy()
        == rec["df_eval_cohort"]["patient_nbr"].to_numpy()
    ).all()


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not cached")
def test_hash_guard_fires_on_tampered_manifest():
    manifest = copy.deepcopy(calibrated.load_manifest())
    manifest["split_hashes"]["S_eval"] = "0" * 64  # tamper
    with pytest.raises(AssertionError, match="hash drift"):
        audit.recover_s_eval_frames(manifest)


@pytest.mark.slow
@pytest.mark.skipif(
    not (RAW_CSV.exists() and MLRUNS.exists()), reason="raw dataset or MLflow registry absent"
)
def test_audit_runs_end_to_end_small_bootstrap():
    r = audit.compute(n_boot=50)
    assert r["s_eval_hash"] == audit.S_EVAL_EXPECTED_HASH
    assert set(r["per_attribute"]) == set(audit.ATTRIBUTES)
    assert r["project_verdict"] in {"PASS", "FLAG (monitor)", "MATERIAL DISPARITY"}
    # payer_code stays exploratory in the emitted structure.
    assert r["per_attribute"]["payer_code"]["tier"] == "exploratory"
    # Race is powered only for Caucasian vs AfricanAmerican (prereg §2.1 / §3).
    assert set(r["per_attribute"]["race"]["powered_labels"]) == {"Caucasian", "AfricanAmerican"}

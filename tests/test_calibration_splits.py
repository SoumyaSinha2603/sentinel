"""Tests for the Phase-2 calibration surfaces (sentinel.calibration.calibration_splits).

Verifies patient-disjointness across {S_train, S_cal, S_eval, holdout}, the S_fit = S_train
∪ S_cal identity, full row coverage, and reproducible patient-set hashes.
"""

import numpy as np
import pandas as pd

from sentinel.calibration.calibration_splits import make_calibration_splits
from sentinel.evaluation.splits import GROUP_COL, make_holdout_split
from sentinel.features.build import MEDICATION_COLUMNS, build_features


def _toy_cohort(n: int = 400) -> pd.DataFrame:
    """Cohort-schema frame with enough patients for grouped sub-splitting."""
    rng = np.random.default_rng(0)
    base = {
        "encounter_id": list(range(1, n + 1)),
        "patient_nbr": [i // 2 + 1 for i in range(n)],
        "race": (["Caucasian", "AfricanAmerican", "Hispanic"] * n)[:n],
        "gender": (["Male", "Female"] * n)[:n],
        "age": (["[0-10)", "[70-80)", "[50-60)", "[90-100)"] * n)[:n],
        "weight": [np.nan] * n,
        "admission_type_id": ([1, 2, 3] * n)[:n],
        "discharge_disposition_id": ([1, 2, 3] * n)[:n],
        "admission_source_id": ([7, 1, 4] * n)[:n],
        "time_in_hospital": rng.integers(1, 14, n),
        "payer_code": [np.nan] * n,
        "medical_specialty": (["Cardiology", "InternalMedicine"] * n)[:n],
        "num_lab_procedures": rng.integers(1, 80, n),
        "num_procedures": rng.integers(0, 6, n),
        "num_medications": rng.integers(1, 30, n),
        "number_outpatient": rng.integers(0, 3, n),
        "number_emergency": rng.integers(0, 3, n),
        "number_inpatient": rng.integers(0, 3, n),
        "diag_1": (["250.83", "428", "486", "V45"] * n)[:n],
        "diag_2": (["715", "156", "?", "578"] * n)[:n],
        "diag_3": (["800", "250", "460", "788"] * n)[:n],
        "number_diagnoses": rng.integers(1, 9, n),
        "max_glu_serum": [np.nan] * n,
        "A1Cresult": (["None", ">7", ">8", "Norm"] * n)[:n],
        "change": (["Ch", "No"] * n)[:n],
        "diabetesMed": (["Yes", "No"] * n)[:n],
        "readmitted": (["<30", ">30", "NO"] * n)[:n],
    }
    for med in MEDICATION_COLUMNS:
        base[med] = (["No", "Steady", "Up", "Down"] * n)[:n]
    return build_features(pd.DataFrame(base))


def _train_frame(df: pd.DataFrame) -> pd.DataFrame:
    train_idx, _ = make_holdout_split(df)
    return df.iloc[train_idx].reset_index(drop=True)


def test_surfaces_patient_disjoint_and_holdout_excluded():
    df = _toy_cohort()
    train_idx, holdout_idx = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    sp = make_calibration_splits(df_train)
    pat = sp["patients"]

    assert pat["S_train"].isdisjoint(pat["S_cal"])
    assert pat["S_train"].isdisjoint(pat["S_eval"])
    assert pat["S_cal"].isdisjoint(pat["S_eval"])
    assert pat["S_fit"] == pat["S_train"] | pat["S_cal"]

    holdout = {str(p) for p in df[GROUP_COL].to_numpy()[holdout_idx]}
    for name in ("S_train", "S_cal", "S_eval", "S_fit"):
        assert holdout.isdisjoint(pat[name])


def test_row_coverage_exact():
    df_train = _train_frame(_toy_cohort())
    sp = make_calibration_splits(df_train)
    idx = sp["indices"]
    assert len(idx["S_train"]) + len(idx["S_cal"]) == len(idx["S_fit"])
    assert len(idx["S_fit"]) + len(idx["S_eval"]) == len(df_train)


def test_hashes_deterministic_and_64_hex():
    df = _toy_cohort()
    a = make_calibration_splits(_train_frame(df))
    b = make_calibration_splits(_train_frame(df))
    for name in ("S_train", "S_cal", "S_eval", "S_fit"):
        ha = a["stats"][name]["patient_sha256"]
        hb = b["stats"][name]["patient_sha256"]
        assert ha == hb
        assert len(ha) == 64
    # distinct surfaces have distinct patient sets -> distinct hashes
    assert a["stats"]["S_train"]["patient_sha256"] != a["stats"]["S_cal"]["patient_sha256"]


def test_prevalence_in_unit_interval():
    sp = make_calibration_splits(_train_frame(_toy_cohort()))
    for name in ("S_train", "S_cal", "S_eval", "S_fit"):
        assert 0.0 <= sp["stats"][name]["prevalence"] <= 1.0

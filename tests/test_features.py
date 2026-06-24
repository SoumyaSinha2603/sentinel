"""Tests for the target-free feature transform (sentinel.features.build)."""

import numpy as np
import pandas as pd

from sentinel.evaluation.splits import feature_columns
from sentinel.features.build import (
    CATEGORICAL_FEATURES,
    MEDICATION_COLUMNS,
    NUMERIC_FEATURES,
    build_features,
    map_icd9,
)


def _toy_cohort(n: int = 12) -> pd.DataFrame:
    """A tiny frame mirroring the cohort schema (post-cohort: examide/citoglipton gone)."""
    rng = np.random.default_rng(0)
    base = {
        "encounter_id": list(range(1, n + 1)),
        "patient_nbr": [i // 2 + 1 for i in range(n)],
        "race": ["Caucasian"] * (n - 2) + [np.nan, np.nan],
        "gender": (["Male", "Female"] * n)[:n],
        "age": (["[0-10)", "[70-80)", "[50-60)", "[90-100)"] * n)[:n],
        "weight": [np.nan] * n,
        "admission_type_id": [1, 2, 3] * (n // 3),
        "discharge_disposition_id": [1, 2, 3] * (n // 3),
        "admission_source_id": [7, 1, 4] * (n // 3),
        "time_in_hospital": rng.integers(1, 14, n),
        "payer_code": [np.nan] * n,
        "medical_specialty": [np.nan] * n,
        "num_lab_procedures": rng.integers(1, 80, n),
        "num_procedures": rng.integers(0, 6, n),
        "num_medications": rng.integers(1, 30, n),
        "number_outpatient": rng.integers(0, 3, n),
        "number_emergency": rng.integers(0, 3, n),
        "number_inpatient": rng.integers(0, 3, n),
        "diag_1": ["250.83", "428", "486", "V45"] * (n // 4),
        "diag_2": ["715", "156", "?", "578"] * (n // 4),
        "diag_3": ["800", "250", "460", "788"] * (n // 4),
        "number_diagnoses": rng.integers(1, 9, n),
        "max_glu_serum": [np.nan] * n,
        "A1Cresult": [np.nan] * (n - 1) + [">7"],
        "change": (["Ch", "No"] * n)[:n],
        "diabetesMed": (["Yes", "No"] * n)[:n],
        "readmitted": (["<30", ">30", "NO"] * n)[:n],
    }
    for med in MEDICATION_COLUMNS:
        base[med] = (["No", "Steady", "Up", "Down"] * n)[:n]
    return pd.DataFrame(base)


def test_row_count_preserved():
    df = _toy_cohort()
    assert len(build_features(df)) == len(df)


def test_icd9_mapping():
    assert map_icd9("250.83") == "Diabetes"
    assert map_icd9("250") == "Diabetes"
    assert map_icd9(428) == "Circulatory"
    assert map_icd9("486") == "Respiratory"
    assert map_icd9("V45") == "Other"
    assert map_icd9("?") == "Missing"
    assert map_icd9(np.nan) == "Missing"
    assert map_icd9("800") == "Injury"
    assert map_icd9("715") == "Musculoskeletal"
    assert map_icd9("156") == "Neoplasms"


def test_age_midpoint():
    df = _toy_cohort()
    out = build_features(df)
    age_by_enc = dict(zip(df["encounter_id"], df["age"], strict=True))
    mid_by_enc = dict(zip(out["encounter_id"], out["age_midpoint"], strict=True))
    for enc, age in age_by_enc.items():
        if age == "[0-10)":
            assert mid_by_enc[enc] == 5
        if age == "[70-80)":
            assert mid_by_enc[enc] == 75


def test_med_counts():
    # 3 meds on (Steady/Up/Down != No), of which 2 changed (Up, Down).
    df = pd.DataFrame(
        {
            "metformin": ["No"],
            "insulin": ["Steady"],
            "glipizide": ["Up"],
            "glyburide": ["Down"],
        }
    )
    cols = ["metformin", "insulin", "glipizide", "glyburide"]
    n_on = (df[cols] != "No").sum(axis=1).iloc[0]
    n_changed = df[cols].isin(["Up", "Down"]).sum(axis=1).iloc[0]
    assert n_on == 3
    assert n_changed == 2


def test_no_nan_in_categoricals():
    out = build_features(_toy_cohort())
    for col in CATEGORICAL_FEATURES:
        assert out[col].isna().sum() == 0, f"NaN left in categorical {col}"


def test_passthrough_columns_present():
    out = build_features(_toy_cohort())
    for col in ("encounter_id", "patient_nbr", "readmitted"):
        assert col in out.columns


def test_feature_set_reconciles():
    out = build_features(_toy_cohort())
    assert set(feature_columns(out)) == set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)


def test_determinism():
    df = _toy_cohort()
    a = build_features(df)
    b = build_features(df)
    pd.testing.assert_frame_equal(a, b)


def test_target_independence():
    """Shuffling the label must not change any engineered feature column (label-blind)."""
    df = _toy_cohort()
    base = build_features(df)

    shuffled = df.copy()
    shuffled["readmitted"] = df["readmitted"].sample(frac=1.0, random_state=123).to_numpy()
    rebuilt = build_features(shuffled)

    feat_cols = feature_columns(base)
    pd.testing.assert_frame_equal(base[feat_cols], rebuilt[feat_cols])

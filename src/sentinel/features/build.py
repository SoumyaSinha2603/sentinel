"""
Feature engineering for Sentinel — deterministic, TARGET-FREE transform.

Chains off the locked cohort (`sentinel.data.cohort.build_cohort`) and turns it into the
modeling feature frame. This module is **pure and label-blind**: it never references the
``readmitted`` target to construct a feature, preserves row count exactly (eligibility
already happened in cohort.py), and uses no RNG. The raw identifiers and target are passed
through untouched — the frozen split harness builds ``y`` from ``readmitted``.

Transforms (see `build_features`):
  - ICD-9 diagnosis codes -> Strack disease groups (raw diag_1/2/3 dropped).
  - age bucket string -> numeric midpoint (raw age dropped).
  - medication-activity counts over the surviving medication columns.
  - total_prior_visits = outpatient + emergency + inpatient.
  - missingness -> explicit categories (weight dropped at 96.9% missing).
  - coded id columns cast to nominal categoricals (they are codes, not magnitudes).
"""

from __future__ import annotations

import pandas as pd

from sentinel.data.cohort import build_cohort
from sentinel.data.load import fetch_raw
from sentinel.evaluation.splits import IDENTIFIER_COLS, TARGET_COL, feature_columns

# Surviving medication columns = the original Diabetes-130 medication set MINUS the two
# zero-variance meds (examide, citoglipton) already dropped by cohort.build_cohort. Kept
# in full as categoricals — we let the model prune, we do not hand-drop near-constant meds.
MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Age bucket -> numeric midpoint.
AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

# Coded id columns: integer-coded categories, not magnitudes -> nominal categoricals.
CODED_CATEGORICAL_IDS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    *MEDICATION_COLUMNS,
]

NUMERIC_FEATURES = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "n_meds_on",
    "n_meds_changed",
    "total_prior_visits",
]


def map_icd9(code) -> str:
    """Map a single ICD-9 diagnosis code to a Strack disease group.

    Pure and label-blind. NaN / "?" -> "Missing"; V- and E-codes -> "Other"; numeric
    codes routed by the Strack et al. 2014 ranges. Non-parseable codes fall back to
    "Other" rather than raising.
    """
    if pd.isna(code) or code == "?":
        return "Missing"
    s = str(code)
    if s.startswith("V") or s.startswith("E"):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if int(v) == 250:  # 250.xx
        return "Diabetes"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure, target-free feature transform on a cohort frame. Row count is preserved.

    Identifiers (``encounter_id``, ``patient_nbr``) and the raw ``readmitted`` target are
    passed through unchanged. Raises ``AssertionError`` if the resulting feature set does
    not match ``CATEGORICAL_FEATURES | NUMERIC_FEATURES`` exactly (so this module and the
    frozen harness can never disagree on what a feature is).
    """
    n_in = len(df)
    out = df.copy()

    # 1. ICD-9 diagnosis grouping (drop raw diag codes).
    for raw, grouped in (
        ("diag_1", "diag_1_group"),
        ("diag_2", "diag_2_group"),
        ("diag_3", "diag_3_group"),
    ):
        out[grouped] = out[raw].map(map_icd9)
    out = out.drop(columns=["diag_1", "diag_2", "diag_3"])

    # 2. Age bucket -> numeric midpoint (drop raw age).
    out["age_midpoint"] = out["age"].map(AGE_MIDPOINTS)
    out = out.drop(columns=["age"])

    # 3. Medication-activity counts over the surviving medication columns.
    meds = out[MEDICATION_COLUMNS]
    out["n_meds_on"] = (meds != "No").sum(axis=1).astype(int)
    out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(int)

    # 4. Total prior utilization.
    out["total_prior_visits"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )

    # 5. Missingness -> explicit categories.
    out = out.drop(columns=["weight"])  # 96.9% missing
    for col in ("medical_specialty", "payer_code", "race"):
        out[col] = out[col].fillna("Missing")
    out["gender"] = out["gender"].replace("Unknown/Invalid", "Missing")
    for col in ("max_glu_serum", "A1Cresult"):
        out[col] = out[col].fillna("None")

    # 6. Cast nominal categoricals to pandas 'category' dtype (after all fillna).
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")

    # Rigor: row count preserved and the feature set matches the declared lists exactly.
    assert len(out) == n_in, f"row count changed: {n_in} -> {len(out)}"
    produced = set(feature_columns(out))
    declared = set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
    if produced != declared:
        raise AssertionError(
            "feature set mismatch between build_features and declared lists:\n"
            f"  only produced (unexpected): {sorted(produced - declared)}\n"
            f"  only declared (missing):    {sorted(declared - produced)}"
        )
    # Identifiers and raw target survive untouched for the harness.
    for col in (*IDENTIFIER_COLS, TARGET_COL):
        assert col in out.columns, f"required pass-through column dropped: {col}"

    return out


def load_and_build() -> pd.DataFrame:
    """Convenience: raw -> cohort -> engineered features."""
    return build_features(build_cohort(fetch_raw()))

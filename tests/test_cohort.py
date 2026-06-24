"""Tests for the cohort builder (eligibility + variance filters)."""

import pandas as pd

from sentinel.data.cohort import DEATH_HOSPICE_DISPOSITIONS, build_cohort


def _toy_frame() -> pd.DataFrame:
    """Small frame: repeat patients, some death/hospice rows, a constant + near-constant col."""
    return pd.DataFrame(
        {
            "encounter_id": [10, 5, 20, 30, 40, 50],
            "patient_nbr": [1, 1, 2, 2, 3, 3],
            "discharge_disposition_id": [1, 11, 3, 13, 1, 6],  # 11, 13 are death/hospice
            "constant_col": ["No", "No", "No", "No", "No", "No"],  # zero variance
            "near_constant_col": ["No", "No", "No", "No", "No", "Yes"],  # 5/6 = ~83%
            "feature_a": [1, 2, 3, 4, 5, 6],
            "readmitted": ["<30", "<30", "NO", ">30", "NO", "<30"],
        }
    )


def test_death_hospice_removed():
    df = _toy_frame()
    cohort = build_cohort(df)
    remaining = set(cohort["discharge_disposition_id"])
    assert remaining.isdisjoint(DEATH_HOSPICE_DISPOSITIONS)
    # the two flagged rows (ids 11, 13) are gone
    assert len(cohort) == len(df) - 2


def test_zero_variance_dropped():
    df = _toy_frame()
    cohort = build_cohort(df)
    # exactly-one-value column dropped...
    assert "constant_col" not in cohort.columns
    # ...but a merely-near-constant column survives
    assert "near_constant_col" in cohort.columns
    # identifiers and target always survive
    assert "encounter_id" in cohort.columns
    assert "patient_nbr" in cohort.columns
    assert "readmitted" in cohort.columns


def test_first_encounter_mode():
    df = _toy_frame()
    cohort = build_cohort(df, first_encounter_only=True)
    # each patient appears exactly once
    assert cohort["patient_nbr"].is_unique
    # and it is the minimum encounter_id among that patient's *eligible* encounters.
    # patient 1: enc 5 is death/hospice (removed) -> earliest eligible is enc 10.
    # patient 2: enc 30 is death/hospice (removed) -> earliest eligible is enc 20.
    # patient 3: both eligible -> earliest is enc 40.
    kept = dict(zip(cohort["patient_nbr"], cohort["encounter_id"], strict=True))
    assert kept == {1: 10, 2: 20, 3: 40}


def test_cohort_is_pure():
    df = _toy_frame()
    before = df.copy(deep=True)
    _ = build_cohort(df)
    pd.testing.assert_frame_equal(df, before)

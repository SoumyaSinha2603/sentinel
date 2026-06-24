"""Tests for the locked evaluation harness (patient-grouped splits)."""

import numpy as np
import pandas as pd

from sentinel.evaluation.splits import (
    feature_columns,
    make_binary_target,
    make_holdout_split,
)


def _toy_frame(n_patients: int = 60, encounters_per_patient: int = 3) -> pd.DataFrame:
    """A small frame with repeat encounters per patient and a mixed target."""
    rng = np.random.default_rng(0)
    rows = []
    enc = 1
    for pid in range(1, n_patients + 1):
        for _ in range(encounters_per_patient):
            rows.append(
                {
                    "encounter_id": enc,
                    "patient_nbr": pid,
                    "feature_a": rng.normal(),
                    "readmitted": rng.choice(["<30", ">30", "NO"]),
                }
            )
            enc += 1
    return pd.DataFrame(rows)


def test_target_binarization():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
    y = make_binary_target(df)
    assert list(y) == [1, 0, 0, 1]
    # purity: input is untouched
    assert list(df["readmitted"]) == ["<30", ">30", "NO", "<30"]


def test_split_no_patient_overlap():
    df = _toy_frame()
    train_idx, test_idx = make_holdout_split(df)
    train_patients = set(df["patient_nbr"].iloc[train_idx])
    test_patients = set(df["patient_nbr"].iloc[test_idx])
    assert train_patients.isdisjoint(test_patients)


def test_split_determinism():
    df = _toy_frame()
    train_a, test_a = make_holdout_split(df)
    train_b, test_b = make_holdout_split(df)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)


def test_identifiers_not_in_features():
    df = _toy_frame()
    feats = feature_columns(df)
    assert "encounter_id" not in feats
    assert "patient_nbr" not in feats
    assert "readmitted" not in feats
    assert "feature_a" in feats

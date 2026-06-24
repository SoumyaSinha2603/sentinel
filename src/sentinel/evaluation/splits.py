"""
Locked evaluation harness for Sentinel — patient-grouped splits.

Every model in the project is judged through *exactly* these splits: same grouped
split, same seed, same folds. This guarantees that a patient never appears in both
train and test (a naive row-wise split would leak ~30k repeat encounters across the
boundary). See ``CLAUDE.md`` — this module is **frozen once approved**; do not add
per-model bespoke splitting.

Locked configuration lives here as module constants. ``encounter_id`` and
``patient_nbr`` are identifiers and must NEVER be used as model features; the target
binarisation (``readmitted == "<30"`` → 1) is likewise locked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# --- Locked configuration (do not change without explicit sign-off) -----------------
SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
GROUP_COL = "patient_nbr"
TARGET_COL = "readmitted"
POSITIVE_LABEL = "<30"

# Identifiers — present in the raw frame but excluded from every feature set.
IDENTIFIER_COLS = ("encounter_id", "patient_nbr")


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    """Binarise the readmission target: 1 where ``readmitted == "<30"``, else 0.

    Pure: does not mutate ``df``. ``">30"`` and ``"NO"`` both map to 0 — this is a
    *30-day* readmission product (locked decision).
    """
    return (df[TARGET_COL] == POSITIVE_LABEL).astype(int)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the modelable columns: every column except identifiers and the target.

    This is the single source of truth for "what may a model see". Identifiers
    (``encounter_id``, ``patient_nbr``) and the raw ``readmitted`` target are excluded.
    """
    excluded = set(IDENTIFIER_COLS) | {TARGET_COL}
    return [c for c in df.columns if c not in excluded]


def assert_no_group_overlap(df: pd.DataFrame, idx_a: np.ndarray, idx_b: np.ndarray) -> None:
    """Raise ``AssertionError`` if any ``patient_nbr`` appears in both index sets.

    Makes the patient-disjointness guarantee structural rather than optional.
    """
    groups = df[GROUP_COL].to_numpy()
    patients_a = set(groups[idx_a])
    patients_b = set(groups[idx_b])
    overlap = patients_a & patients_b
    if overlap:
        raise AssertionError(
            f"patient-level leakage: {len(overlap)} patient(s) appear in both index "
            f"sets (e.g. {sorted(overlap)[:5]})"
        )


def make_holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Outer train/test split grouped by ``patient_nbr``.

    Uses ``GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)``.
    Returns integer positional indices ``(train_idx, test_idx)`` and verifies
    patient-disjointness before returning.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL]))
    assert_no_group_overlap(df, train_idx, test_idx)
    return train_idx, test_idx


def make_cv_folds(df_train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified, patient-grouped CV folds within the training portion.

    Uses ``StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)``
    stratified on the binary target and grouped by ``patient_nbr``. Returns a
    materialised list of ``(train_idx, val_idx)`` positional-index pairs (deterministic).
    """
    y = make_binary_target(df_train)
    groups = df_train[GROUP_COL]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(df_train, y, groups=groups))

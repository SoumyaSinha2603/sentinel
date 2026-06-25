"""Phase-2 calibration surfaces — carved from INSIDE the 80% train.

The Phase-1 holdout is spent (CLAUDE.md), so calibration needs its own evaluation
surface that has never fit anything. This helper does **not** touch the frozen harness
(``sentinel.evaluation.splits`` stays untouched); it only re-uses its locked primitives
(``SEED``, ``GROUP_COL``, ``make_binary_target``, ``assert_no_group_overlap``).

All splits are patient-grouped (``GroupShuffleSplit``, group=``patient_nbr``, seed=42) so a
patient never straddles two surfaces. From the 80% train:

    S_eval   = 20% of the 80%   -> FROZEN Phase-2 eval surface (never fits anything)
    S_fit    = remaining 80%    -> calibrator-fitting region
      S_cal   = 25% of S_fit    -> fallback single-model calibrator-fit surface
      S_train = 75% of S_fit    -> fallback cal-booster training surface

Indices returned are positional **within the passed 80%-train frame** (assumed already
``reset_index(drop=True)``). Per-surface reproducibility stats — including a sha256 over
the sorted unique ``patient_nbr`` — are returned so the manifest can lock the partition.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from sentinel.evaluation.splits import (
    GROUP_COL,
    SEED,
    assert_no_group_overlap,
    make_binary_target,
)

# Fractions (locked alongside SEED for a reproducible partition).
S_EVAL_TEST_SIZE = 0.20  # S_eval is 20% of the 80% train
S_CAL_TEST_SIZE = 0.25  # S_cal is 25% of S_fit (S_train is the 75% remainder)

SURFACE_NAMES = ("S_train", "S_cal", "S_eval", "S_fit")


def _patient_hash(df: pd.DataFrame, idx: np.ndarray) -> str:
    """sha256 over the sorted unique ``patient_nbr`` of a surface (partition lock)."""
    patients = sorted({str(p) for p in df[GROUP_COL].to_numpy()[idx]})
    return hashlib.sha256(",".join(patients).encode("utf-8")).hexdigest()


def _surface_stats(df: pd.DataFrame, y: np.ndarray, idx: np.ndarray) -> dict:
    """n_patients, n_rows, prevalence, and patient sha256 for one surface."""
    return {
        "n_patients": int(len({str(p) for p in df[GROUP_COL].to_numpy()[idx]})),
        "n_rows": int(len(idx)),
        "prevalence": float(np.mean(y[idx])) if len(idx) else 0.0,
        "patient_sha256": _patient_hash(df, idx),
    }


def make_calibration_splits(df_train: pd.DataFrame) -> dict:
    """Partition the 80% train into S_eval / S_fit / (S_cal, S_train), patient-grouped.

    ``df_train`` MUST be the 80% training portion, ``reset_index(drop=True)``. Returns a
    dict with positional ``indices`` (within ``df_train``), ``patients`` (sets), and
    per-surface ``stats``. Pairwise patient-disjointness across {S_train, S_cal, S_eval}
    is asserted before returning (S_fit = S_train ∪ S_cal by construction).
    """
    y = make_binary_target(df_train).to_numpy()
    groups = df_train[GROUP_COL]

    # S_fit (80%) vs S_eval (20%) of the 80% train.
    outer = GroupShuffleSplit(n_splits=1, test_size=S_EVAL_TEST_SIZE, random_state=SEED)
    s_fit_idx, s_eval_idx = next(outer.split(df_train, groups=groups))

    # Within S_fit: S_train (75%) vs S_cal (25%).
    df_fit = df_train.iloc[s_fit_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=S_CAL_TEST_SIZE, random_state=SEED)
    train_local, cal_local = next(inner.split(df_fit, groups=df_fit[GROUP_COL]))
    s_train_idx = s_fit_idx[train_local]
    s_cal_idx = s_fit_idx[cal_local]

    indices = {
        "S_train": s_train_idx,
        "S_cal": s_cal_idx,
        "S_eval": s_eval_idx,
        "S_fit": s_fit_idx,
    }

    # Structural disjointness: the three leaf surfaces must share no patient.
    assert_no_group_overlap(df_train, s_train_idx, s_cal_idx)
    assert_no_group_overlap(df_train, s_train_idx, s_eval_idx)
    assert_no_group_overlap(df_train, s_cal_idx, s_eval_idx)

    patients = {
        name: {str(p) for p in df_train[GROUP_COL].to_numpy()[idx]} for name, idx in indices.items()
    }
    stats = {name: _surface_stats(df_train, y, idx) for name, idx in indices.items()}

    return {"indices": indices, "patients": patients, "stats": stats}

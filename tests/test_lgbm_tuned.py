"""Tests for the tuned LightGBM step (sentinel.models.lgbm_tuned)."""

import numpy as np
import pandas as pd

from sentinel.features.build import MEDICATION_COLUMNS, build_features
from sentinel.models import lgbm_tuned


def _toy_cohort(n: int = 240) -> pd.DataFrame:
    """A CV-friendly cohort-schema frame (enough patients for nested grouped CV)."""
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


def test_smoke_short_study_finite():
    df = _toy_cohort()
    tuned = lgbm_tuned.tune(df, n_trials=3, inner_splits=2)
    outer = lgbm_tuned.score_outer(df, tuned["refit_params"])
    # finite best params + metrics
    for k in ("num_leaves", "learning_rate", "reg_alpha", "reg_lambda"):
        assert np.isfinite(tuned["best_params"][k])
    for k in ("auroc", "auprc", "brier", "ece"):
        assert np.isfinite(outer["mean"][k])
    assert tuned["final_n_estimators"] >= 1


def test_determinism():
    df = _toy_cohort()
    a = lgbm_tuned.tune(df, n_trials=3, inner_splits=2)
    b = lgbm_tuned.tune(df, n_trials=3, inner_splits=2)
    assert a["best_params"] == b["best_params"]


def test_isolation_holdout_never_in_train():
    """Structural no-leak: holdout patients appear in no inner or outer TRAIN fold."""
    df = _toy_cohort()
    tuned = lgbm_tuned.tune(df, n_trials=2, inner_splits=2)
    outer = lgbm_tuned.score_outer(df, tuned["refit_params"])
    holdout = tuned["holdout_patients"]
    assert holdout  # non-empty
    assert holdout.isdisjoint(tuned["inner_train_patients"])
    assert holdout.isdisjoint(outer["outer_train_patients"])


def test_tripwire_thresholds():
    assert lgbm_tuned.verdict(0.73).startswith("TRIPWIRE")
    assert lgbm_tuned.verdict(0.68).startswith("GATE PASS")
    assert lgbm_tuned.verdict(0.640).startswith("MARGINAL")
    assert lgbm_tuned.verdict(0.60).startswith("GATE FAIL")

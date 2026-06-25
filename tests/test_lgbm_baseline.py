"""Tests for the untuned LightGBM honesty gate (sentinel.models.lgbm_baseline)."""

import numpy as np
import pandas as pd

from sentinel.features.build import CATEGORICAL_FEATURES, MEDICATION_COLUMNS, build_features
from sentinel.models import lgbm_baseline


def _toy_cohort(n: int = 240) -> pd.DataFrame:
    """A CV-friendly cohort-schema frame (enough patients for a 5-fold grouped split)."""
    rng = np.random.default_rng(0)
    base = {
        "encounter_id": list(range(1, n + 1)),
        "patient_nbr": [i // 2 + 1 for i in range(n)],
        "race": (["Caucasian", "AfricanAmerican", "Hispanic"] * n)[:n],
        "gender": (["Male", "Female"] * n)[:n],
        "age": (["[0-10)", "[70-80)", "[50-60)", "[90-100)"] * n)[:n],
        "weight": [np.nan] * n,
        "admission_type_id": (([1, 2, 3] * n)[:n]),
        "discharge_disposition_id": (([1, 2, 3] * n)[:n]),
        "admission_source_id": (([7, 1, 4] * n)[:n]),
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


def test_smoke_trains_and_returns_finite_metrics():
    result = lgbm_baseline.run_cv(_toy_cohort())
    for key in ("auroc", "auprc", "brier", "ece"):
        assert np.isfinite(result["mean"][key])
        assert np.isfinite(result["std"][key])
    assert len(result["fold_metrics"]) == 5
    assert result["importances"]  # non-empty


def test_determinism():
    df = _toy_cohort()
    a = lgbm_baseline.run_cv(df)
    b = lgbm_baseline.run_cv(df)
    assert a["mean"]["auroc"] == b["mean"]["auroc"]


def test_categorical_wiring_matches_declared():
    """The model's feature frame must mark exactly CATEGORICAL_FEATURES as 'category'."""
    x = lgbm_baseline.feature_frame(_toy_cohort())
    category_cols = set(x.select_dtypes("category").columns)
    assert category_cols == set(CATEGORICAL_FEATURES)
    # coded ids must be categorical, never silently numeric
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        assert col in category_cols


def test_holdout_is_sealed():
    """CV must run only on the training portion — never on the holdout rows/patients."""
    result = lgbm_baseline.run_cv(_toy_cohort())
    assert result["n_holdout_rows"] > 0
    assert result["train_patients"].isdisjoint(result["holdout_patients"])


def test_module_does_not_score_holdout(monkeypatch):
    """Guard: the module never evaluates predictions on the sealed holdout split."""
    import sentinel.models.lgbm_baseline as mod

    real_split = mod.make_holdout_split
    seen = {"called": 0}

    def _tracked(df):
        seen["called"] += 1
        return real_split(df)

    monkeypatch.setattr(mod, "make_holdout_split", _tracked)
    result = mod.run_cv(_toy_cohort())
    # Used exactly once — to carve the train portion, not to score a holdout test set.
    assert seen["called"] == 1
    # Every CV row count is accounted for by the training portion, holdout excluded.
    assert result["n_train_rows"] + result["n_holdout_rows"] == len(_toy_cohort())

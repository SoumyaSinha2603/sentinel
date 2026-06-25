"""Tests for the Phase-1 production-model registration (sentinel.models.register)."""

import numpy as np
import pandas as pd

from sentinel.features.build import (
    CATEGORICAL_FEATURES,
    MEDICATION_COLUMNS,
    NUMERIC_FEATURES,
    build_features,
)
from sentinel.models import register


def _toy_cohort(n: int = 240) -> pd.DataFrame:
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


def _tmp_uri(tmp_path) -> str:
    return (tmp_path / "mlruns").as_uri()


def test_smoke_register_and_reload(tmp_path):
    df = _toy_cohort()
    result = register.register_model(
        df, tracking_uri=_tmp_uri(tmp_path), registered_name="toy-model", attach_study=False
    )
    assert result["version"] == "1"
    assert result["alias"] == "phase1"
    assert result["reload_ok"] is True
    assert len(result["reload_sample_proba"]) == register.N_INPUT_EXAMPLE_ROWS
    assert all(np.isfinite(result["reload_sample_proba"]))


def test_feature_contract_signature(tmp_path):
    """The logged signature input columns must be exactly the engineered feature set."""
    import mlflow

    df = _toy_cohort()
    result = register.register_model(
        df, tracking_uri=_tmp_uri(tmp_path), registered_name="toy-model-2", attach_study=False
    )
    mlflow.set_tracking_uri(_tmp_uri(tmp_path))
    model_info = mlflow.models.get_model_info(result["model_uri"])
    input_cols = [s["name"] for s in model_info.signature.inputs.to_dict()]
    assert set(input_cols) == set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)


def test_determinism_predictions(tmp_path):
    """Two refits with seed=42 give identical predictions on a fixed sample."""
    df = _toy_cohort()
    m1, x1 = register.build_production_model(df)
    m2, _ = register.build_production_model(df)
    p1 = m1.predict_proba(x1.head(10))[:, 1]
    p2 = m2.predict_proba(x1.head(10))[:, 1]
    assert np.array_equal(p1, p2)

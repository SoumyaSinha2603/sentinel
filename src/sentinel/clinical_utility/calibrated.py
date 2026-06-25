"""The Phase-4 calibrated-probability loader path (dogfooded here in Phase 2).

This module deliberately mirrors what the Phase-4 pyfunc wrapper will do: load the
``@phase1`` booster, then apply the calibration map RECONSTRUCTED FROM THE COMMITTED
MANIFEST's portable form (isotonic knots via ``np.interp``, or a Platt sigmoid) — NOT the
pickled joblib. It is intentionally free of any training-code imports (no optuna/MLflow
training modules) so it reflects the real serving dependency surface.

``models/calibration/calibrator_manifest.json`` is the source of truth; ``mlruns/`` is not
load-bearing (mirrors the W5 pattern). Equivalence to the committed joblib is asserted at
point of use (re-confirming W5 CHECK 2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.config import MODELS_DIR, ROOT
from sentinel.models.lgbm_baseline import feature_frame

# Mirror of the registry coordinates (kept local so this module imports no training code).
MLFLOW_TRACKING_URI = (ROOT / "mlruns").as_uri()
REGISTERED_NAME = "sentinel-readmission"
PHASE1_ALIAS = "phase1"

MANIFEST_PATH = MODELS_DIR / "calibration" / "calibrator_manifest.json"
CALIBRATOR_JOBLIB = MODELS_DIR / "calibration" / "calibrator.joblib"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Read the committed calibrator manifest (provenance + portable reconstruction)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_calibrator_portable(manifest: dict, scores: np.ndarray) -> np.ndarray:
    """Apply the calibrator reconstructed from the manifest's portable form.

    isotonic -> ``np.interp`` over (x_thresholds, y_thresholds); np.interp clamps to the
    endpoints, matching ``IsotonicRegression(out_of_bounds="clip")``. platt -> sigmoid.
    """
    scores = np.asarray(scores, dtype=float)
    method = manifest["method"]
    p = manifest["portable"]
    if method == "isotonic":
        x = np.asarray(p["x_thresholds"], dtype=float)
        y = np.asarray(p["y_thresholds"], dtype=float)
        return np.interp(scores, x, y)
    if method == "platt":
        return 1.0 / (1.0 + np.exp(-(p["A"] * scores + p["B"])))
    raise ValueError(f"unknown calibration method: {method!r}")


def load_phase1_booster(tracking_uri: str | None = None):
    """Load the registered ``@phase1`` LightGBM booster (the deployment base)."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or MLFLOW_TRACKING_URI)
    return mlflow.lightgbm.load_model(f"models:/{REGISTERED_NAME}@{PHASE1_ALIAS}")


def raw_proba(df: pd.DataFrame, booster) -> np.ndarray:
    """Uncalibrated @phase1 probability of the positive class."""
    return booster.predict_proba(feature_frame(df))[:, 1]


def get_calibrated_proba(
    df: pd.DataFrame,
    *,
    booster=None,
    manifest: dict | None = None,
    tracking_uri: str | None = None,
) -> np.ndarray:
    """Calibrated positive-class probability via the exact Phase-4 loader path.

    Loads @phase1 (unless ``booster`` is supplied) and applies the manifest's portable
    calibration map to the raw scores. ``booster``/``manifest`` are injectable so callers
    that already hold them (and tests) avoid re-loading.
    """
    manifest = manifest or load_manifest()
    booster = booster if booster is not None else load_phase1_booster(tracking_uri)
    return apply_calibrator_portable(manifest, raw_proba(df, booster))

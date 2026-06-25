"""
Select, persist, and register the Phase-1 production model (Phase 1, Step 5).

Freezes the tuned LightGBM as a versioned, reloadable MLflow artifact under the
registered name ``sentinel-readmission`` with the alias ``phase1``, the locked holdout
number and the Optuna study attached as metadata.

This step makes **no new training decisions**: it refits the *locked* best params
(`lgbm_tuned.LOCKED_BEST_PARAMS`, the single source of truth) on the full 80% training
portion and persists the result. It NEVER re-scores the 20% holdout — the locked 0.677
is attached as recorded metadata, not recomputed. Phase 4's API loads
``models:/sentinel-readmission@phase1``.

Run:

    python -m sentinel.models.register
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.config import REPORTS_DIR
from sentinel.evaluation.splits import make_binary_target, make_holdout_split
from sentinel.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_and_build,
)
from sentinel.models.lgbm_baseline import feature_frame
from sentinel.models.lgbm_tuned import (
    LOCKED_BEST_PARAMS,
    LOCKED_HOLDOUT_METRICS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    _model,
    tune,
)

REGISTERED_NAME = "sentinel-readmission"
ALIAS = "phase1"
N_INPUT_EXAMPLE_ROWS = 5


def build_production_model(df: pd.DataFrame):
    """Refit the locked best params on the FULL 80% train portion (the production booster).

    Returns ``(model, x_train)`` — the fitted classifier and its training feature frame
    (used for the signature / input example). Deterministic (seed=42 inside ``_model``).
    """
    train_idx, _ = make_holdout_split(df)
    x_train = feature_frame(df.iloc[train_idx])
    y_train = make_binary_target(df).to_numpy()[train_idx]
    model = _model(LOCKED_BEST_PARAMS)
    model.fit(x_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    return model, x_train


def _verify_locked_params(df: pd.DataFrame) -> dict:
    """Re-derive the Optuna study (deterministic) to (a) attach trials and (b) prove the
    locked constant is faithful. Not a new decision — the search is reproducible."""
    tuned = tune(df)
    rederived = {
        k: tuned["best_params"][k]
        for k in ("num_leaves", "min_child_samples", "learning_rate", "reg_alpha", "reg_lambda")
    }
    locked = {k: LOCKED_BEST_PARAMS[k] for k in rederived}
    if rederived != locked or tuned["final_n_estimators"] != LOCKED_BEST_PARAMS["n_estimators"]:
        raise AssertionError(
            "re-derived best params do not match LOCKED_BEST_PARAMS:\n"
            f"  re-derived: {rederived}, n_estimators={tuned['final_n_estimators']}\n"
            f"  locked:     {locked}, n_estimators={LOCKED_BEST_PARAMS['n_estimators']}"
        )
    return tuned


def register_model(
    df: pd.DataFrame,
    *,
    tracking_uri: str | None = None,
    registered_name: str = REGISTERED_NAME,
    alias: str = ALIAS,
    attach_study: bool = True,
) -> dict:
    """Refit, log, register, alias, attach metadata, and verify reload-by-alias."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri or MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model, x_train = build_production_model(df)
    sample = x_train.head(N_INPUT_EXAMPLE_ROWS)
    signature = infer_signature(sample, model.predict_proba(sample))

    study = _verify_locked_params(df) if attach_study else None

    with mlflow.start_run(run_name="register_phase1") as run:
        mlflow.log_params({f"best_{k}": v for k, v in LOCKED_BEST_PARAMS.items()})
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)
        mlflow.set_tag("phase", "phase1-production")
        mlflow.set_tag(
            "holdout_provenance",
            "locked Phase-1 step-4 holdout (run lgbm_tuned); recorded, NOT recomputed here",
        )

        info = mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            holdout_path = tmp / "locked_holdout_metrics.json"
            holdout_path.write_text(
                json.dumps(
                    {
                        **LOCKED_HOLDOUT_METRICS,
                        "_note": "locked Phase-1 holdout (step 4); recorded, not recomputed in step 5",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            contract_path = tmp / "feature_contract.json"
            contract_path.write_text(
                json.dumps(
                    {
                        "categorical_features": CATEGORICAL_FEATURES,
                        "numeric_features": NUMERIC_FEATURES,
                        "n_features": len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(holdout_path))
            mlflow.log_artifact(str(contract_path))

            if study is not None:
                trials_path = tmp / "optuna_trials.csv"
                study["study"].trials_dataframe().to_csv(trials_path, index=False)
                mlflow.log_artifact(str(trials_path))

        run_id = run.info.run_id

    mv = mlflow.register_model(info.model_uri, registered_name)
    MlflowClient(tracking_uri).set_registered_model_alias(registered_name, alias, mv.version)

    # Reload BY ALIAS and verify — the exact path Phase 4's API will use.
    loaded = mlflow.lightgbm.load_model(f"models:/{registered_name}@{alias}")
    proba = loaded.predict_proba(sample)[:, 1]
    reload_ok = bool(np.all(np.isfinite(proba)) and proba.shape == (len(sample),))

    return {
        "registered_name": registered_name,
        "version": str(mv.version),
        "alias": alias,
        "model_uri": info.model_uri,
        "artifact_uri": mlflow.get_run(run_id).info.artifact_uri,
        "run_id": run_id,
        "reload_ok": reload_ok,
        "reload_sample_proba": proba.tolist(),
        "study_attached": study is not None,
    }


def _render_report(result: dict) -> str:
    m = LOCKED_HOLDOUT_METRICS
    lines = [
        "# Phase-1 Production Model — Registry",
        "",
        "> The tuned LightGBM (locked best params) refit on the full 80% train and frozen "
        "as a versioned MLflow artifact. No new training decisions; the 20% holdout was "
        "**not** re-scored — its metrics are attached as recorded metadata. Generated by "
        "`sentinel.models.register`.",
        "",
        "## Registered model",
        "",
        "| field | value |",
        "|---|---|",
        f"| registered name | `{result['registered_name']}` |",
        f"| version | {result['version']} |",
        f"| alias | `{result['alias']}` |",
        f"| model URI | `{result['model_uri']}` |",
        f"| artifact URI | `{result['artifact_uri']}` |",
        f"| run id | `{result['run_id']}` |",
        f"| Optuna study attached | {result['study_attached']} |",
        "",
        "## Locked holdout metrics (recorded, NOT recomputed)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| AUROC | {m['auroc']:.3f} |",
        f"| AUPRC | {m['auprc']:.3f} |",
        f"| Brier | {m['brier']:.3f} |",
        f"| ECE | {m['ece']:.3f} |",
        "",
        "_These are the Phase-1 step-4 holdout figures, attached as registry metadata. "
        "The holdout is spent and was not scored again in this step._",
        "",
        "## Reload-by-alias verification",
        "",
        f"Loaded `models:/{result['registered_name']}@{result['alias']}` and predicted on a "
        f"{N_INPUT_EXAMPLE_ROWS}-row sample: **{'OK' if result['reload_ok'] else 'FAILED'}** "
        "(finite, correctly shaped).",
        "",
        f"**Phase 4's API loads `models:/{result['registered_name']}@{result['alias']}`.**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = load_and_build()
    result = register_model(df)

    report = _render_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "model_registry.md"
    out_path.write_text(report, encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(
        f"registered_name={result['registered_name']} version={result['version']} "
        f"alias={result['alias']}"
    )
    print(f"artifact_uri={result['artifact_uri']}")
    print(f"reload_by_alias_ok={result['reload_ok']}")
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()

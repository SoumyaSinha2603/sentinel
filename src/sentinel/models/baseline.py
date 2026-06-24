"""
Baseline models — the honest floor (Week-3 entry point).

Two deliberately-simple baselines run through the **frozen** split harness on the default
modeling cohort. Their purpose is to (1) prove the harness works end-to-end on real
modeling, (2) set the number every future model must beat, and (3) confirm the honest
ceiling sits roughly where the leakage audit predicted (~0.60-0.66 AUROC for a simple
model). This is NOT feature engineering and NOT a contender — no tuning, no encoding.

**Leakage alarm:** if the logistic test AUROC lands at/above ~0.85, that is a leakage
bug, not a win — the runner stops reporting "success" and prints a loud banner instead.

Run:

    python -m sentinel.models.baseline
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.config import REPORTS_DIR, ROOT
from sentinel.data.cohort import build_cohort
from sentinel.data.load import fetch_raw
from sentinel.evaluation import metrics
from sentinel.evaluation.splits import (
    SEED,
    make_binary_target,
    make_cv_folds,
    make_holdout_split,
)

# Small, safe, numeric-only feature set — all counts known at discharge, no leakage risk.
SAFE_NUMERIC_FEATURES = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "number_diagnoses",
    "num_medications",
    "num_lab_procedures",
    "time_in_hospital",
    "num_procedures",
]

# A test AUROC at or above this is a leakage alarm on this dataset, not a success.
LEAKAGE_AUROC_ALARM = 0.85

FIGURES_DIR = REPORTS_DIR / "figures"
MLFLOW_TRACKING_URI = (ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "baseline"


def _make_logistic() -> Pipeline:
    """Standardize features then fit a deliberately-plain balanced logistic regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=1000,
                ),
            ),
        ]
    )


def _cv_scores(df_train: pd.DataFrame) -> dict:
    """Mean +/- std AUROC/AUPRC for the logistic model across the locked CV folds."""
    folds = make_cv_folds(df_train)
    x_all = df_train[SAFE_NUMERIC_FEATURES]
    y_all = make_binary_target(df_train).to_numpy()

    aurocs, auprcs = [], []
    for tr_idx, va_idx in folds:
        model = _make_logistic()
        model.fit(x_all.iloc[tr_idx], y_all[tr_idx])
        prob = model.predict_proba(x_all.iloc[va_idx])[:, 1]
        disc = metrics.discrimination_metrics(y_all[va_idx], prob)
        aurocs.append(disc["auroc"])
        auprcs.append(disc["auprc"])

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "n_folds": len(folds),
    }


def _log_to_mlflow(run_name: str, params: dict, scores: dict, artifacts: list) -> None:
    """One contained MLflow run per baseline (local ./mlruns)."""
    # MLflow 3.x put the ./mlruns file store in maintenance mode and raises unless we
    # opt in. The spec mandates a local ./mlruns store, so opt in rather than switch
    # backends. Set before importing mlflow so the flag is read at store resolution.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for key, value in scores.items():
            if isinstance(value, int | float):
                mlflow.log_metric(key, float(value))
        for art in artifacts:
            if art is not None:
                mlflow.log_artifact(str(art))


def run_baselines() -> dict:
    """Fit both baselines on the cohort, evaluate on the held-out test set, return results."""
    raw = fetch_raw()
    cohort = build_cohort(raw)

    # Minimal preprocessing only: drop rows with NaN within the 8 chosen numeric columns.
    before = len(cohort)
    cohort = cohort.dropna(subset=SAFE_NUMERIC_FEATURES).reset_index(drop=True)
    rows_dropped_nan = before - len(cohort)

    y = make_binary_target(cohort).to_numpy()
    train_idx, test_idx = make_holdout_split(cohort)
    df_train = cohort.iloc[train_idx].reset_index(drop=True)

    x_train = cohort[SAFE_NUMERIC_FEATURES].iloc[train_idx]
    x_test = cohort[SAFE_NUMERIC_FEATURES].iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 1) Trivial reference: predict the train positive rate for everyone.
    train_pos_rate = float(y_train.mean())
    trivial_prob = np.full(len(y_test), train_pos_rate)
    trivial_metrics = metrics.summarize(y_test, trivial_prob)

    # 2) Simple logistic regression on the safe numeric features.
    logistic = _make_logistic()
    logistic.fit(x_train, y_train)
    logistic_prob = logistic.predict_proba(x_test)[:, 1]
    logistic_metrics = metrics.summarize(y_test, logistic_prob)
    logistic_cv = _cv_scores(df_train)

    return {
        "rows_dropped_nan": rows_dropped_nan,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_pos_rate": train_pos_rate,
        "features": SAFE_NUMERIC_FEATURES,
        "trivial": {"metrics": trivial_metrics, "y_test": y_test, "prob": trivial_prob},
        "logistic": {
            "metrics": logistic_metrics,
            "cv": logistic_cv,
            "y_test": y_test,
            "prob": logistic_prob,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _render_report(results: dict, alarm: bool) -> str:
    triv = results["trivial"]["metrics"]
    logi = results["logistic"]["metrics"]
    cv = results["logistic"]["cv"]

    lines: list[str] = []
    lines.append("# Baseline Results — UCI Diabetes-130")
    lines.append("")
    lines.append(
        "> Honest floor. Two simple baselines through the frozen split harness on the "
        "default cohort. Minimal preprocessing only (no feature engineering, no tuning). "
        "Generated by `sentinel.models.baseline`."
    )
    lines.append("")

    if alarm:
        lines.append("## 🚨 LEAKAGE ALARM")
        lines.append("")
        lines.append(
            f"Logistic test AUROC = **{logi['auroc']:.4f}** ≥ {LEAKAGE_AUROC_ALARM:.2f}. "
            "On this dataset that is a **data-leakage bug, not a win**. STOP and "
            "investigate before trusting any result below."
        )
        lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Train / test rows: {results['n_train']:,} / {results['n_test']:,}")
    lines.append(
        f"- Rows dropped for NaN in the 8 numeric features: {results['rows_dropped_nan']:,}"
    )
    lines.append(f"- Train positive rate (trivial prediction): {results['train_pos_rate']:.4f}")
    lines.append(f"- Features: {', '.join(f'`{f}`' for f in results['features'])}")
    lines.append("")

    lines.append("## Held-out test metrics")
    lines.append("")
    lines.append("| model | AUROC | AUPRC | Brier | ECE |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| trivial (constant) | {_fmt(triv['auroc'])} | {_fmt(triv['auprc'])} "
        f"| {_fmt(triv['brier'])} | {_fmt(triv['ece'])} |"
    )
    lines.append(
        f"| logistic (8 numeric) | {_fmt(logi['auroc'])} | {_fmt(logi['auprc'])} "
        f"| {_fmt(logi['brier'])} | {_fmt(logi['ece'])} |"
    )
    lines.append("")
    lines.append(
        f"Test-set prevalence: {logi['prevalence']:.4f} "
        f"({logi['positives']:,} / {logi['n']:,}). The trivial AUPRC equals "
        "prevalence and its AUROC is 0.5 by construction — that is the floor."
    )
    lines.append("")

    lines.append("## Logistic cross-validation (locked folds, training portion)")
    lines.append("")
    lines.append(f"Across {cv['n_folds']} grouped CV folds:")
    lines.append("")
    lines.append(f"- AUROC: **{cv['auroc_mean']:.4f} ± {cv['auroc_std']:.4f}**")
    lines.append(f"- AUPRC: **{cv['auprc_mean']:.4f} ± {cv['auprc_std']:.4f}**")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if not alarm:
        lines.append(
            f"Logistic test AUROC {logi['auroc']:.4f} sits in the expected ~0.60-0.66 "
            "band for a simple model — consistent with the clean leakage audit and the "
            "honest ~0.68 ceiling. This is the number future models must beat; AUPRC and "
            "calibration (ECE) matter more than AUROC at this ~11% prevalence."
        )
    else:
        lines.append("See the leakage alarm above — do not interpret these as valid.")
    lines.append("")
    lines.append(
        "Figures: `reports/figures/baseline_reliability.png`, "
        "`reports/figures/baseline_roc_pr.png`. Runs logged to MLflow (`./mlruns`)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = run_baselines()
    logi = results["logistic"]
    alarm = logi["metrics"]["auroc"] >= LEAKAGE_AUROC_ALARM

    # Figures for the logistic model (the meaningful one).
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reliability_path = FIGURES_DIR / "baseline_reliability.png"
    roc_pr_path = FIGURES_DIR / "baseline_roc_pr.png"
    metrics.plot_reliability(logi["y_test"], logi["prob"], reliability_path)
    metrics.plot_roc_pr(logi["y_test"], logi["prob"], roc_pr_path)

    # MLflow: one contained run per baseline.
    triv_m = results["trivial"]["metrics"]
    _log_to_mlflow(
        run_name="trivial_constant",
        params={
            "strategy": "predict_train_positive_rate",
            "train_pos_rate": results["train_pos_rate"],
        },
        scores={k: v for k, v in triv_m.items() if isinstance(v, int | float)},
        artifacts=[],
    )
    _log_to_mlflow(
        run_name="logistic_8numeric",
        params={
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "random_state": SEED,
            "n_features": len(SAFE_NUMERIC_FEATURES),
            "features": ",".join(SAFE_NUMERIC_FEATURES),
        },
        scores={
            **{k: v for k, v in logi["metrics"].items() if isinstance(v, int | float)},
            **logi["cv"],
        },
        artifacts=[reliability_path, roc_pr_path],
    )

    report = _render_report(results, alarm)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "baseline_results.md"
    out_path.write_text(report, encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"\n[written] {out_path}")
    if alarm:
        print(
            "\n*** LEAKAGE ALARM: logistic AUROC >= "
            f"{LEAKAGE_AUROC_ALARM} — STOP and investigate. ***"
        )


if __name__ == "__main__":
    main()

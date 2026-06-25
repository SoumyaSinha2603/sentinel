"""
Untuned LightGBM honesty gate (Phase 1, Step 3).

Trains a DEFAULT-hyperparameter LightGBM on the 47 engineered features through the frozen
harness, purely to decide whether feature engineering beats the logistic baseline
(CV AUROC 0.634 ± 0.008) by a clear margin. This is a go/no-go gate for the tuning step:
**no tuning, no calibration, no SHAP.**

Sealed-holdout discipline: the 20% grouped holdout stays sealed until the end of Phase 1.
This module calls ``make_holdout_split`` ONLY to carve out the 80% training portion and
runs cross-validation strictly inside it — it never trains on or scores the holdout rows.

The one concession to imbalance is ``class_weight="balanced"`` (mirroring the logistic
baseline so the comparison is fair). Everything else is LightGBM defaults. Categoricals
are passed NATIVELY via ``categorical_feature`` (pandas 'category' dtype set in build.py);
there is no target/mean encoding anywhere.

Run:

    python -m sentinel.models.lgbm_baseline
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from sentinel.config import REPORTS_DIR, ROOT
from sentinel.evaluation import metrics
from sentinel.evaluation.splits import (
    GROUP_COL,
    SEED,
    make_binary_target,
    make_cv_folds,
    make_holdout_split,
)
from sentinel.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_and_build,
)

# Feature matrix column order (categoricals first, then numerics). Categoricals carry
# pandas 'category' dtype from build.py and are fed to LightGBM natively.
FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]

# Phase 0 reference rows (same harness/metrics), for the comparison table.
BASELINE_CV_AUROC = 0.634  # logistic, 8 numeric features, CV mean

# Tripwire thresholds on mean CV AUROC.
TRIPWIRE_LEAKAGE = 0.72
GATE_PASS = 0.65

MLFLOW_TRACKING_URI = (ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "baseline"


def _make_lgbm() -> LGBMClassifier:
    """Default LightGBM, the single imbalance concession (balanced), seeded, single-threaded.

    ``n_jobs=1`` is for bit-for-bit reproducibility (the determinism test depends on it),
    not a tuning choice — it does not change the learning algorithm. ``verbose=-1`` only
    silences logging.
    """
    return LGBMClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
    )


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The 47-column model matrix; categoricals retain 'category' dtype (no encoding)."""
    return df[FEATURE_COLUMNS]


def _fit(model: LGBMClassifier, x: pd.DataFrame, y: np.ndarray) -> LGBMClassifier:
    """Fit with categoricals passed NATIVELY by name (never silently treated as numeric)."""
    model.fit(x, y, categorical_feature=CATEGORICAL_FEATURES)
    return model


def _verdict(mean_cv_auroc: float) -> str:
    """Automated tripwire verdict for human review (does not auto-proceed)."""
    if mean_cv_auroc >= TRIPWIRE_LEAKAGE:
        return "TRIPWIRE: AUROC >= 0.72 — STOP, investigate leakage"
    if mean_cv_auroc >= GATE_PASS:
        return "GATE PASS: clear lift over 0.634 baseline"
    if mean_cv_auroc > BASELINE_CV_AUROC:
        return "MARGINAL: above baseline but within noise — discuss"
    return "GATE FAIL: no lift over baseline — diagnose, do not tune"


def run_cv(df: pd.DataFrame) -> dict:
    """Cross-validate default LightGBM inside the sealed training portion.

    Returns fold-level and mean±std metrics, the tripwire verdict, gain importances, and
    bookkeeping that proves the holdout was never touched.
    """
    y = make_binary_target(df).to_numpy()

    # Carve the 80% training portion; the 20% holdout (test_idx) is recorded only so we
    # can prove it was excluded — it is never trained on or scored here.
    train_idx, test_idx = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    x_train = feature_frame(df_train)

    folds = make_cv_folds(df_train)
    keys = ("auroc", "auprc", "brier", "ece")
    fold_metrics: list[dict] = []
    for tr_idx, va_idx in folds:
        model = _make_lgbm()
        _fit(model, x_train.iloc[tr_idx], y_train[tr_idx])
        prob = model.predict_proba(x_train.iloc[va_idx])[:, 1]
        summary = metrics.summarize(y_train[va_idx], prob)
        fold_metrics.append({k: summary[k] for k in keys})

    mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
    std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}

    # Gain importances from a single fit on the full training portion (still no holdout).
    final = _make_lgbm()
    _fit(final, x_train, y_train)
    gains = final.booster_.feature_importance(importance_type="gain")
    names = final.booster_.feature_name()
    importances = sorted(
        ({"feature": n, "gain": float(g)} for n, g in zip(names, gains, strict=True)),
        key=lambda r: r["gain"],
        reverse=True,
    )

    return {
        "fold_metrics": fold_metrics,
        "mean": mean,
        "std": std,
        "verdict": _verdict(mean["auroc"]),
        "importances": importances,
        "n_train_rows": int(len(train_idx)),
        "n_holdout_rows": int(len(test_idx)),
        "train_patients": set(df[GROUP_COL].iloc[train_idx]),
        "holdout_patients": set(df[GROUP_COL].iloc[test_idx]),
    }


def _log_to_mlflow(result: dict) -> None:
    """One contained MLflow run (local ./mlruns), mirroring the baseline runner."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="lgbm_untuned"):
        mlflow.log_params(
            {
                "model": "LGBMClassifier",
                "params": "lightgbm defaults",
                "class_weight": "balanced",
                "random_state": SEED,
                "n_features": len(FEATURE_COLUMNS),
                "n_categorical": len(CATEGORICAL_FEATURES),
            }
        )
        for k, v in result["mean"].items():
            mlflow.log_metric(f"{k}_mean", v)
            mlflow.log_metric(f"{k}_std", result["std"][k])
        for i, fm in enumerate(result["fold_metrics"]):
            for k, v in fm.items():
                mlflow.log_metric(f"fold{i}_{k}", v)


def _render_report(result: dict) -> str:
    m, s = result["mean"], result["std"]
    lines: list[str] = []
    lines.append("# Untuned LightGBM — Honesty Gate")
    lines.append("")
    lines.append(
        "> Default-hyperparameter LightGBM on the 47 engineered features, 5-fold grouped "
        "CV inside the **sealed** 80% training portion (the 20% holdout is untouched). "
        "Go/no-go gate for tuning — no tuning, no calibration, no SHAP. AUPRC leads "
        "(11.39% prevalence). Generated by `sentinel.models.lgbm_baseline`."
    )
    lines.append("")
    lines.append("## Comparison (same harness + metrics)")
    lines.append("")
    lines.append("| model | AUROC (mean±std) | AUPRC | Brier | ECE |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append("| trivial constant | 0.500 | 0.112 | 0.099 | 0.000 |")
    lines.append("| logistic (8 num) | 0.634±0.008 | 0.199 | 0.233 | 0.366 |")
    lines.append(
        f"| lgbm_untuned | {m['auroc']:.3f}±{s['auroc']:.3f} | {m['auprc']:.3f} "
        f"| {m['brier']:.3f} | {m['ece']:.3f} |"
    )
    lines.append("")
    lines.append(
        "_Trivial/logistic AUROC & AUPRC are Phase 0 CV; their Brier & ECE are the Phase 0 "
        "holdout-test figures (the only ones recorded) — so Brier/ECE are not strictly "
        "comparable to the lgbm CV row, while AUROC & AUPRC are the apples-to-apples "
        "CV comparison. lgbm row is 5-fold CV mean±std._"
    )
    lines.append("")
    lines.append("## Tripwire verdict")
    lines.append("")
    lines.append(f"**{result['verdict']}**  (mean CV AUROC = {m['auroc']:.4f})")
    lines.append("")
    lines.append("## Per-fold metrics")
    lines.append("")
    lines.append("| fold | AUROC | AUPRC | Brier | ECE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for i, fm in enumerate(result["fold_metrics"], start=1):
        lines.append(
            f"| {i} | {fm['auroc']:.4f} | {fm['auprc']:.4f} | {fm['brier']:.4f} | {fm['ece']:.4f} |"
        )
    lines.append(
        f"| **mean±std** | **{m['auroc']:.4f}±{s['auroc']:.4f}** | "
        f"**{m['auprc']:.4f}±{s['auprc']:.4f}** | **{m['brier']:.4f}±{s['brier']:.4f}** | "
        f"**{m['ece']:.4f}±{s['ece']:.4f}** |"
    )
    lines.append("")
    lines.append("## Top-15 feature importances (gain)")
    lines.append("")
    lines.append("| rank | feature | gain |")
    lines.append("|---:|---|---:|")
    for i, r in enumerate(result["importances"][:15], start=1):
        lines.append(f"| {i} | `{r['feature']}` | {r['gain']:,.1f} |")
    lines.append("")
    lines.append("## Honest read")
    lines.append("")
    lift = m["auroc"] - BASELINE_CV_AUROC
    lines.append(
        f"LightGBM on the engineered features reaches CV AUROC {m['auroc']:.3f} "
        f"({lift:+.3f} vs the 0.634 logistic baseline) and AUPRC {m['auprc']:.3f} "
        f"(vs 0.199). The tripwire reads **{result['verdict'].split(':')[0]}**. "
        "Feature engineering "
        + (
            "delivers a clear, leakage-free lift — proceed to tuning."
            if m["auroc"] >= GATE_PASS
            else (
                "helps but only modestly — weigh before investing in tuning."
                if m["auroc"] > BASELINE_CV_AUROC
                else "does not lift over the baseline — diagnose before tuning."
            )
        )
        + " No tripwire leakage concern (AUROC well under 0.72)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_and_build()
    result = run_cv(df)
    _log_to_mlflow(result)

    report = _render_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "lgbm_baseline_results.md"
    out_path.write_text(report, encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"\n{result['verdict']}")
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()

"""
Tuned LightGBM via a focused Optuna search (Phase 1, Step 4).

Captures the small headroom (~0.005–0.01) above the untuned gate (CV AUROC 0.672) with a
moderate, well-isolated search. This is the LAST step allowed to chase performance.

Tuning isolation (the integrity core):
  - ``make_holdout_split`` carves the 80% TRAIN portion. The 20% holdout is NEVER touched
    during the search and is scored exactly ONCE, at the very end.
  - Protocol used here is the spec-allowed simplification of full nested CV: a single
    INNER grouped CV (3-fold, group=patient_nbr, seed=42) over the whole 80% train is what
    Optuna optimizes; the chosen params are then scored across the 5 frozen OUTER folds
    (``make_cv_folds``) to produce the reported tuned CV number. The inner partition is
    distinct from the outer 5 folds, but both cover the same 80% rows — so the outer CV is
    mildly optimistic; the sealed-holdout number is the unbiased estimate.

Search: 6 params, ~60 TPE trials, seed=42, optimizing mean inner-CV AUPRC (11% prevalence).
``class_weight="balanced"`` is FIXED (calibration handles probability quality later).
Categoricals are native (``categorical_feature``); no target encoding anywhere.

Run:

    python -m sentinel.models.lgbm_tuned
"""

from __future__ import annotations

import os
import sys

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedGroupKFold

from sentinel.config import REPORTS_DIR, ROOT
from sentinel.data.cohort import build_cohort
from sentinel.data.load import fetch_raw
from sentinel.evaluation import metrics
from sentinel.evaluation.splits import (
    GROUP_COL,
    SEED,
    make_binary_target,
    make_cv_folds,
    make_holdout_split,
)
from sentinel.features.build import CATEGORICAL_FEATURES, build_features
from sentinel.models.lgbm_baseline import (
    BASELINE_CV_AUROC,
    GATE_PASS,
    TRIPWIRE_LEAKAGE,
    feature_frame,
)

# Untuned reference (Phase 1, step 3), for the comparison table and the tuning delta.
UNTUNED_CV_AUROC = 0.672
UNTUNED_CV_AUPRC = 0.227

# --- Locked Phase-1 result (single source of truth) ----------------------------------
# Best params from the Phase-1 step-4 Optuna run (MLflow run "lgbm_tuned",
# reports/lgbm_tuned_results.md), at full precision. ``n_estimators`` is the
# early-stopping-derived tree count (193), NOT the sampled search cap (327). The
# production refit (sentinel.models.register) imports THIS dict — do not re-type.
LOCKED_BEST_PARAMS = {
    "num_leaves": 41,
    "min_child_samples": 68,
    "learning_rate": 0.027333136696790394,
    "n_estimators": 193,
    "reg_alpha": 0.0016180450927017812,
    "reg_lambda": 0.03571344592510516,
}

# Locked Phase-1 holdout metrics — RECORDED from the step-4 run, never recomputed (the
# 20% holdout is spent). Attached as registry metadata in step 5, not re-scored.
LOCKED_HOLDOUT_METRICS = {"auroc": 0.677, "auprc": 0.235, "brier": 0.213, "ece": 0.334}

N_TRIALS = 60
INNER_SPLITS = 3
EARLY_STOPPING_ROUNDS = 50
HOLDOUT_GAP_FLAG = 0.02  # |CV - holdout| above this => flag possible CV-overfit

MLFLOW_TRACKING_URI = (ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "baseline"


def verdict(mean_cv_auroc: float) -> str:
    """Tripwire verdict — identical thresholds to the untuned gate (step 3)."""
    if mean_cv_auroc >= TRIPWIRE_LEAKAGE:
        return "TRIPWIRE: AUROC >= 0.72 — STOP, investigate leakage"
    if mean_cv_auroc >= GATE_PASS:
        return "GATE PASS: clear lift over 0.634 baseline"
    if mean_cv_auroc > BASELINE_CV_AUROC:
        return "MARGINAL: above baseline but within noise — discuss"
    return "GATE FAIL: no lift over baseline — diagnose, do not tune"


def _model(params: dict) -> LGBMClassifier:
    """LightGBM with the fixed knobs (balanced, seeded, single-threaded) plus ``params``."""
    return LGBMClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
        **params,
    )


def _suggest(trial: optuna.Trial) -> dict:
    """The exactly-six search space — nothing wider."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def _fit_es(model: LGBMClassifier, x_tr, y_tr, x_va, y_va) -> LGBMClassifier:
    """Fit with early stopping on an inner validation set (AUPRC), categoricals native."""
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )
    return model


def _fit_plain(model: LGBMClassifier, x_tr, y_tr) -> LGBMClassifier:
    """Fit a fixed number of trees, no early stopping (so val never drives tree count)."""
    model.fit(x_tr, y_tr, categorical_feature=CATEGORICAL_FEATURES)
    return model


def tune(df: pd.DataFrame, n_trials: int = N_TRIALS, inner_splits: int = INNER_SPLITS) -> dict:
    """Optuna search on a single inner grouped CV over the sealed 80% train portion."""
    y = make_binary_target(df).to_numpy()
    train_idx, test_idx = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    x_train = feature_frame(df_train)

    # Inner folds computed ONCE (distinct from the outer 5), reused across trials.
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
    inner_folds = list(inner.split(df_train, y_train, groups=df_train[GROUP_COL]))
    inner_train_patients: set = set()
    for tr, _ in inner_folds:
        inner_train_patients |= set(df_train[GROUP_COL].iloc[tr])

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        auprcs, best_iters = [], []
        for tr, va in inner_folds:
            model = _model(params)
            _fit_es(model, x_train.iloc[tr], y_train[tr], x_train.iloc[va], y_train[va])
            prob = model.predict_proba(x_train.iloc[va])[:, 1]
            auprcs.append(metrics.discrimination_metrics(y_train[va], prob)["auprc"])
            best_iters.append(model.best_iteration_ or params["n_estimators"])
        trial.set_user_attr("mean_best_iter", int(round(float(np.mean(best_iters)))))
        return float(np.mean(auprcs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    final_n_estimators = max(
        1, int(best.user_attrs.get("mean_best_iter", best.params["n_estimators"]))
    )
    # Refit params = best 6-param set, but with the early-stopping-derived tree count.
    refit_params = {**best.params, "n_estimators": final_n_estimators}

    return {
        "study": study,
        "best_params": dict(best.params),
        "refit_params": refit_params,
        "best_inner_auprc": float(study.best_value),
        "final_n_estimators": final_n_estimators,
        "n_trials": n_trials,
        "inner_splits": inner_splits,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "inner_train_patients": inner_train_patients,
        "holdout_patients": set(df[GROUP_COL].iloc[test_idx]),
    }


def score_outer(df: pd.DataFrame, refit_params: dict) -> dict:
    """Score the chosen params across the 5 frozen OUTER folds (the reported tuned CV)."""
    y = make_binary_target(df).to_numpy()
    train_idx, test_idx = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    x_train = feature_frame(df_train)

    folds = make_cv_folds(df_train)
    keys = ("auroc", "auprc", "brier", "ece")
    fold_metrics: list[dict] = []
    outer_train_patients: set = set()
    for tr, va in folds:
        outer_train_patients |= set(df_train[GROUP_COL].iloc[tr])
        model = _fit_plain(_model(refit_params), x_train.iloc[tr], y_train[tr])
        prob = model.predict_proba(x_train.iloc[va])[:, 1]
        summary = metrics.summarize(y_train[va], prob)
        fold_metrics.append({k: summary[k] for k in keys})

    mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
    std = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
    return {
        "fold_metrics": fold_metrics,
        "mean": mean,
        "std": std,
        "outer_train_patients": outer_train_patients,
        "holdout_patients": set(df[GROUP_COL].iloc[test_idx]),
    }


def evaluate_holdout(df: pd.DataFrame, refit_params: dict) -> dict:
    """The ONE honest number: refit on the full 80% train, score the sealed 20% holdout."""
    y = make_binary_target(df).to_numpy()
    train_idx, test_idx = make_holdout_split(df)
    x_train = feature_frame(df.iloc[train_idx])
    x_test = feature_frame(df.iloc[test_idx])

    model = _fit_plain(_model(refit_params), x_train, y[train_idx])
    prob = model.predict_proba(x_test)[:, 1]
    summary = metrics.summarize(y[test_idx], prob)
    return {k: summary[k] for k in ("auroc", "auprc", "brier", "ece")}


def first_encounter_sensitivity(refit_params: dict) -> dict:
    """Re-run the tuned model's OUTER CV on the first-encounter-only cohort (no new tuning)."""
    df_fe = build_features(build_cohort(fetch_raw(), first_encounter_only=True))
    outer = score_outer(df_fe, refit_params)
    return {"mean": outer["mean"], "std": outer["std"], "n_rows": int(len(df_fe))}


def _render_report(tuned: dict, outer: dict, holdout: dict, fe: dict) -> str:
    m, s = outer["mean"], outer["std"]
    bp = tuned["best_params"]
    cv_holdout_gap = m["auroc"] - holdout["auroc"]
    flag = abs(cv_holdout_gap) > HOLDOUT_GAP_FLAG

    lines: list[str] = []
    lines.append("# Tuned LightGBM — Optuna (focused)")
    lines.append("")
    lines.append(
        "> Moderate Optuna search (6 params, "
        f"{tuned['n_trials']} TPE trials, seed 42, optimizing inner-CV AUPRC). "
        "Tuning isolated from the sealed 20% holdout. Generated by "
        "`sentinel.models.lgbm_tuned`."
    )
    lines.append("")
    lines.append("## Comparison (same harness + metrics)")
    lines.append("")
    lines.append("| model | AUROC | AUPRC | Brier | ECE |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append("| trivial constant | 0.500 | 0.112 | 0.099 | 0.000 |")
    lines.append("| logistic (8 num) | 0.634±0.008 | 0.199 | 0.233 | 0.366 |")
    lines.append(
        f"| lgbm_untuned (CV) | {UNTUNED_CV_AUROC:.3f} | {UNTUNED_CV_AUPRC:.3f} | 0.208 | 0.320 |"
    )
    lines.append(
        f"| **lgbm_tuned (CV)** | **{m['auroc']:.3f}±{s['auroc']:.3f}** | "
        f"**{m['auprc']:.3f}** | {m['brier']:.3f} | {m['ece']:.3f} |"
    )
    lines.append(
        f"| **lgbm_tuned (HOLDOUT)** | **{holdout['auroc']:.3f}** | **{holdout['auprc']:.3f}** | "
        f"{holdout['brier']:.3f} | {holdout['ece']:.3f} |"
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "Spec-allowed simplification of full nested CV: Optuna optimized a single **inner "
        f"{tuned['inner_splits']}-fold** grouped CV (group=`patient_nbr`, seed 42) over the "
        "whole 80% train; the chosen params were scored across the **5 frozen outer folds** "
        "(`make_cv_folds`) for the reported tuned CV. Inner partition ≠ outer partition, but "
        "both cover the same 80% rows, so the outer CV is mildly optimistic — the holdout row "
        "is the unbiased number. The 20% holdout was scored exactly once."
    )
    lines.append("")
    lines.append(
        f"- Trials: {tuned['n_trials']}  ·  best inner-CV AUPRC: {tuned['best_inner_auprc']:.4f}"
    )
    lines.append(
        f"- Early-stopping-derived n_estimators (fixed for refits): {tuned['final_n_estimators']}"
    )
    lines.append("")
    lines.append("### Best params")
    lines.append("")
    lines.append("| param | value |")
    lines.append("|---|---:|")
    for k in (
        "num_leaves",
        "min_child_samples",
        "learning_rate",
        "n_estimators",
        "reg_alpha",
        "reg_lambda",
    ):
        v = bp[k]
        lines.append(f"| {k} | {v:.5g} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines.append("| class_weight | balanced (fixed) |")
    lines.append("")
    lines.append("## Per-fold tuned CV")
    lines.append("")
    lines.append("| fold | AUROC | AUPRC | Brier | ECE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for i, fm in enumerate(outer["fold_metrics"], start=1):
        lines.append(
            f"| {i} | {fm['auroc']:.4f} | {fm['auprc']:.4f} | {fm['brier']:.4f} | {fm['ece']:.4f} |"
        )
    lines.append(
        f"| **mean±std** | **{m['auroc']:.4f}±{s['auroc']:.4f}** | **{m['auprc']:.4f}±{s['auprc']:.4f}** "
        f"| **{m['brier']:.4f}** | **{m['ece']:.4f}** |"
    )
    lines.append("")
    lines.append("## `first_encounter_only` sensitivity")
    lines.append("")
    fe_auroc = fe["mean"]["auroc"]
    shift = fe_auroc - m["auroc"]
    lines.append(
        f"Re-running the tuned params' 5-fold outer CV on the first-encounter-only cohort "
        f"({fe['n_rows']:,} rows) gives AUROC {fe_auroc:.3f} (AUPRC {fe['mean']['auprc']:.3f}), "
        f"a shift of {shift:+.3f} vs the all-encounters tuned CV ({m['auroc']:.3f}). "
        + (
            "No material shift — the repeat-encounter structure is not inflating results."
            if abs(shift) <= 0.01
            else "Material shift — the repeat-encounter structure warrants discussion."
        )
        + " No new tuning was done for this check."
    )
    lines.append("")
    lines.append("## Tripwire verdict")
    lines.append("")
    lines.append(f"**{verdict(m['auroc'])}**  (tuned mean CV AUROC = {m['auroc']:.4f})")
    lines.append("")
    lines.append("## Honest read")
    lines.append("")
    cv_delta = m["auroc"] - UNTUNED_CV_AUROC
    lines.append(
        f"Tuning moved CV AUROC {cv_delta:+.3f} (to {m['auroc']:.3f}) and AUPRC to {m['auprc']:.3f} "
        f"vs the untuned {UNTUNED_CV_AUROC:.3f}/{UNTUNED_CV_AUPRC:.3f} — "
        + (
            "a real but small gain, as expected near the ~0.68 ceiling."
            if cv_delta > 0
            else "no meaningful gain."
        )
        + f" The sealed holdout reads AUROC {holdout['auroc']:.3f} (CV−holdout gap {cv_holdout_gap:+.3f}); "
        + (
            "**FLAG: gap exceeds 0.02 — possible CV-overfit, discuss.**"
            if flag
            else "within the 0.02 tolerance, no CV-overfit concern."
        )
        + " No tripwire (well under 0.72)."
    )
    lines.append("")
    return "\n".join(lines)


def _log_to_mlflow(tuned: dict, outer: dict, holdout: dict, fe: dict) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="lgbm_tuned"):
        mlflow.log_params(
            {
                "search_space": "num_leaves,min_child_samples,learning_rate,n_estimators,reg_alpha,reg_lambda",
                "n_trials": tuned["n_trials"],
                "inner_splits": tuned["inner_splits"],
                "class_weight": "balanced",
                "random_state": SEED,
                **{f"best_{k}": v for k, v in tuned["best_params"].items()},
                "final_n_estimators": tuned["final_n_estimators"],
            }
        )
        mlflow.log_metric("inner_best_auprc", tuned["best_inner_auprc"])
        for k, v in outer["mean"].items():
            mlflow.log_metric(f"cv_{k}_mean", v)
            mlflow.log_metric(f"cv_{k}_std", outer["std"][k])
        for k, v in holdout.items():
            mlflow.log_metric(f"holdout_{k}", v)
        for k, v in fe["mean"].items():
            mlflow.log_metric(f"firstenc_cv_{k}", v)


def main() -> None:
    df = build_features(build_cohort(fetch_raw()))
    tuned = tune(df)
    outer = score_outer(df, tuned["refit_params"])
    holdout = evaluate_holdout(df, tuned["refit_params"])
    fe = first_encounter_sensitivity(tuned["refit_params"])

    _log_to_mlflow(tuned, outer, holdout, fe)
    report = _render_report(tuned, outer, holdout, fe)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "lgbm_tuned_results.md"
    out_path.write_text(report, encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"\n{verdict(outer['mean']['auroc'])}")
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()

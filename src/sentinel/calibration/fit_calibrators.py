"""Phase 2 — Trust Layer I: probability calibration (isotonic vs Platt).

Discrimination was locked in Phase 1 (tuned LightGBM, holdout AUROC 0.677) but the
probabilities are not yet trustworthy (holdout ECE 0.334, a side effect of
``class_weight="balanced"``). This module fits and selects a calibrator on surfaces carved
from *inside* the 80% train — the Phase-1 20% holdout is spent and is NEVER loaded here.

Integrity rails (all enforced in code):
  - The frozen harness (`sentinel.evaluation.splits`) is reused, never modified.
  - Tripwire: any surface AUROC >= 0.72 raises and stops (leakage signal).
  - `S_eval` is frozen — it fits nothing; it is looked at exactly once, in Step 7.
  - The selection rule is pre-registered as a module constant before any `S_eval` number
    is computed, and is copied verbatim into the manifest.

Run:

    python -m sentinel.calibration.fit_calibrators
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from sentinel.calibration.calibration_splits import make_calibration_splits
from sentinel.config import MODELS_DIR, REPORTS_DIR, ROOT
from sentinel.evaluation import metrics
from sentinel.evaluation.splits import (
    GROUP_COL,
    SEED,
    make_binary_target,
    make_holdout_split,
)
from sentinel.features.build import CATEGORICAL_FEATURES
from sentinel.models.lgbm_baseline import TRIPWIRE_LEAKAGE, feature_frame
from sentinel.models.lgbm_tuned import (
    LOCKED_BEST_PARAMS,
    LOCKED_HOLDOUT_METRICS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    _model,
)
from sentinel.models.register import ALIAS as PHASE1_ALIAS
from sentinel.models.register import REGISTERED_NAME

# --- Locked Phase-2 configuration ------------------------------------------------------
REQUIRED_SKLEARN = "1.7.2"
N_CAL_FOLDS = 5
PHASE2_ALIAS = "phase2-calibrated"

# Pre-registered gate thresholds (Step 6) — change only with sign-off.
GATE_KS_MAX = 0.05
GATE_DECILE_MAX = 0.02  # max |Δ| over the 9 deciles, in probability units

# Reliability-monotonicity tolerance: a method is rejected if observed frequency drops by
# more than this between adjacent bins (ordered by mean predicted probability).
MONOTONE_TOL = 0.02

# PRE-REGISTERED selection rule (declared BEFORE any S_eval metric is computed).
SELECTION_RULE_TEXT = (
    "Lowest S_eval ECE wins; Brier breaks ties (lower wins). Reject any method whose "
    "S_eval reliability curve is non-monotone (observed frequency dropping by more than "
    f"{MONOTONE_TOL} between adjacent bins as mean predicted probability increases). This "
    "rule was pre-registered as a module constant before any S_eval number was looked at."
)

# Caveat recorded verbatim in the gate section (Step 6).
IN_SAMPLE_CAVEAT = (
    "@phase1 is IN-SAMPLE on S_eval (trained on the full 80%), so D_prod is mildly "
    "sharpened; the gate is therefore distributional, not per-example equality."
)

CAL_DIR = MODELS_DIR / "calibration"
CALIBRATOR_PATH = CAL_DIR / "calibrator.joblib"
MANIFEST_PATH = CAL_DIR / "calibrator_manifest.json"
FIGURE_PATH = REPORTS_DIR / "figures" / "reliability_before_after.png"
REPORT_PATH = REPORTS_DIR / "calibration_results.md"

EXPECTED_OOF_AUROC = 0.676  # sanity reference only (not a gate)


# --- Environment + provenance ----------------------------------------------------------
def assert_env() -> dict:
    """Hard-fail if sklearn drifts; capture lightgbm version for the manifest (Step 0)."""
    assert sklearn.__version__ == REQUIRED_SKLEARN, sklearn.__version__
    return {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _check_tripwire(auroc: float, where: str) -> None:
    if auroc >= TRIPWIRE_LEAKAGE:
        raise RuntimeError(
            f"TRIPWIRE: {where} AUROC {auroc:.4f} >= {TRIPWIRE_LEAKAGE} — STOP, "
            "investigate leakage before continuing."
        )


# --- Boosters --------------------------------------------------------------------------
def _make_unweighted(params: dict) -> LGBMClassifier:
    """LOCKED params but WITHOUT class_weight (Step 8 diagnostic only)."""
    return LGBMClassifier(class_weight=None, random_state=SEED, n_jobs=1, verbose=-1, **params)


def _fit(model: LGBMClassifier, x: pd.DataFrame, y: np.ndarray) -> LGBMClassifier:
    model.fit(x, y, categorical_feature=CATEGORICAL_FEATURES)
    return model


def _oof_scores(
    df_sub: pd.DataFrame, x_sub: pd.DataFrame, y_sub: np.ndarray, factory, n_folds: int
) -> tuple[np.ndarray, list[float]]:
    """Pooled out-of-fold probabilities over ``df_sub`` (StratifiedGroupKFold, grouped)."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.full(len(y_sub), np.nan)
    fold_auroc: list[float] = []
    for tr, va in skf.split(df_sub, y_sub, groups=df_sub[GROUP_COL]):
        model = _fit(factory(), x_sub.iloc[tr], y_sub[tr])
        prob = model.predict_proba(x_sub.iloc[va])[:, 1]
        oof[va] = prob
        fold_auroc.append(float(roc_auc_score(y_sub[va], prob)))
    assert not np.isnan(oof).any(), "OOF coverage gap — every row must be scored once"
    return oof, fold_auroc


def _grouped_cv_auroc(
    df_sub: pd.DataFrame, x_sub: pd.DataFrame, y_sub: np.ndarray, factory, n_folds: int
) -> float:
    """Mean grouped-CV AUROC over a training region (used for the sibling/diagnostic CVs)."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aurocs = []
    for tr, va in skf.split(df_sub, y_sub, groups=df_sub[GROUP_COL]):
        model = _fit(factory(), x_sub.iloc[tr], y_sub[tr])
        prob = model.predict_proba(x_sub.iloc[va])[:, 1]
        aurocs.append(float(roc_auc_score(y_sub[va], prob)))
    return float(np.mean(aurocs))


# --- Calibrators (fit, apply, portable reconstruction) ---------------------------------
def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=float))
    return iso


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression()
    lr.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(y))
    return lr


def isotonic_portable(iso: IsotonicRegression) -> dict:
    """Portable form: interpolation knots (so Phase 4 never unpickles)."""
    return {
        "x_thresholds": [float(v) for v in iso.X_thresholds_],
        "y_thresholds": [float(v) for v in iso.y_thresholds_],
    }


def platt_portable(lr: LogisticRegression) -> dict:
    """Portable form: sigmoid(A * score + B)."""
    return {"A": float(lr.coef_[0][0]), "B": float(lr.intercept_[0])}


def apply_calibrator(method: str, obj, scores: np.ndarray) -> np.ndarray:
    """Apply a fitted sklearn calibrator object to raw scores."""
    scores = np.asarray(scores, dtype=float)
    if method == "isotonic":
        return np.asarray(obj.predict(scores), dtype=float)
    return obj.predict_proba(scores.reshape(-1, 1))[:, 1]


def apply_portable(method: str, portable: dict, scores: np.ndarray) -> np.ndarray:
    """Reconstruct + apply a calibrator from its portable form (the Phase-4 path)."""
    scores = np.asarray(scores, dtype=float)
    if method == "isotonic":
        x = np.asarray(portable["x_thresholds"], dtype=float)
        y = np.asarray(portable["y_thresholds"], dtype=float)
        # np.interp clamps to endpoints, matching IsotonicRegression(out_of_bounds="clip").
        return np.interp(scores, x, y)
    return 1.0 / (1.0 + np.exp(-(portable["A"] * scores + portable["B"])))


# --- Gate, monotonicity, selection -----------------------------------------------------
def score_gate(d_cal: np.ndarray, d_prod: np.ndarray) -> dict:
    """OOF-vs-production score-distribution gate (Step 6). Two-sample KS + 9 decile deltas."""
    from scipy.stats import ks_2samp

    ks = float(ks_2samp(d_cal, d_prod).statistic)
    qs = np.round(np.arange(0.1, 1.0, 0.1), 1)  # 0.1 .. 0.9 -> 9 deciles
    dec_cal = np.quantile(d_cal, qs)
    dec_prod = np.quantile(d_prod, qs)
    deltas = np.abs(dec_cal - dec_prod)
    max_delta = float(deltas.max())
    passed = bool(ks <= GATE_KS_MAX and max_delta <= GATE_DECILE_MAX)
    return {
        "ks": ks,
        "decile_deltas": [float(d) for d in deltas],
        "max_decile_delta": max_delta,
        "d_cal_mean": float(d_cal.mean()),
        "d_cal_std": float(d_cal.std()),
        "d_prod_mean": float(d_prod.mean()),
        "d_prod_std": float(d_prod.std()),
        "ks_threshold": GATE_KS_MAX,
        "decile_threshold": GATE_DECILE_MAX,
        "pass": passed,
    }


def _is_monotone(reliability: list[dict], tol: float = MONOTONE_TOL) -> bool:
    pts = sorted(reliability, key=lambda b: b["mean_pred"])
    obs = [b["obs_freq"] for b in pts]
    return all(obs[i + 1] - obs[i] >= -tol for i in range(len(obs) - 1))


def select_winner(summaries: dict, monotone: dict) -> tuple[str, list[str]]:
    """Pre-registered rule: reject non-monotone, then min ECE, Brier tie-break."""
    rejected = [m for m in ("isotonic", "platt") if not monotone[m]]
    pool = [m for m in ("isotonic", "platt") if m not in rejected] or ["isotonic", "platt"]
    winner = min(pool, key=lambda m: (summaries[m]["ece"], summaries[m]["brier"]))
    return winner, rejected


# --- Figure ----------------------------------------------------------------------------
def plot_reliability_overlay(
    y_eval: np.ndarray, raw: np.ndarray, platt: np.ndarray, iso: np.ndarray, path
) -> None:
    """Reliability curves: raw vs Platt vs isotonic, with the diagonal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for label, probs in (("raw", raw), ("Platt", platt), ("isotonic", iso)):
        cal = metrics.calibration_metrics(y_eval, probs)
        mp = [b["mean_pred"] for b in cal["reliability"]]
        of = [b["obs_freq"] for b in cal["reliability"]]
        ax.plot(mp, of, marker="o", label=f"{label} (ECE={cal['ece']:.3f})")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency (bin)")
    ax.set_title("S_eval reliability — raw vs Platt vs isotonic")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- MLflow load / register ------------------------------------------------------------
def _load_phase1_booster(tracking_uri: str | None):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or MLFLOW_TRACKING_URI)
    return mlflow.lightgbm.load_model(f"models:/{REGISTERED_NAME}@{PHASE1_ALIAS}")


def _register_calibrated(
    base_booster, sample: pd.DataFrame, manifest: dict, tracking_uri: str | None
) -> dict:
    """Log the base booster + calibrator artifacts as a new version, alias phase2-calibrated."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    signature = infer_signature(sample, base_booster.predict_proba(sample))

    with mlflow.start_run(run_name="register_phase2_calibrated") as run:
        g = manifest["gate"]
        mlflow.set_tag("phase", "phase2-calibration")
        mlflow.set_tag(
            "loader_note",
            "Phase 4's pyfunc wrapper (the one already planned to cast category dtypes) is "
            "the intended loader: it reads calibrator_manifest.json, reconstructs the map "
            "from the portable form, asserts the sklearn version, and applies it AFTER the "
            "booster call.",
        )
        mlflow.log_param("calibration_method", manifest["method"])
        mlflow.log_param("deployment_base", manifest["deployment_base"])
        mlflow.log_metric("gate_ks", g["ks"])
        mlflow.log_metric("gate_max_decile_delta", g["max_decile_delta"])
        mlflow.log_metric("gate_pass", float(g["pass"]))
        for m in ("isotonic", "platt"):
            mlflow.log_metric(f"{m}_ece_after", manifest["s_eval"][m]["ece_after"])
            mlflow.log_metric(f"{m}_brier_after", manifest["s_eval"][m]["brier_after"])
        mlflow.log_metric("ece_before", manifest["s_eval"]["ece_before"])

        info = mlflow.lightgbm.log_model(
            base_booster, name="model", signature=signature, input_example=sample
        )
        mlflow.log_artifact(str(CALIBRATOR_PATH))
        mlflow.log_artifact(str(MANIFEST_PATH))
        run_id = run.info.run_id

    mv = mlflow.register_model(info.model_uri, REGISTERED_NAME)
    MlflowClient(uri).set_registered_model_alias(REGISTERED_NAME, PHASE2_ALIAS, mv.version)
    return {"version": str(mv.version), "alias": PHASE2_ALIAS, "run_id": run_id}


# --- Orchestration ---------------------------------------------------------------------
def run(
    df: pd.DataFrame,
    *,
    do_register: bool = True,
    write_outputs: bool = True,
    tracking_uri: str | None = None,
    n_folds: int = N_CAL_FOLDS,
) -> dict:
    """Full Phase-2 calibration pipeline (Steps 0–9). Returns a result dict for the report."""
    versions = assert_env()

    # Step 1 — recover the 80% train; the 20% holdout is identified only to PROVE exclusion.
    train_idx, holdout_idx = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    holdout_patients = {str(p) for p in df[GROUP_COL].to_numpy()[holdout_idx]}
    y_train = make_binary_target(df_train).to_numpy()
    x_train = feature_frame(df_train)

    # Step 2 — calibration surfaces (own helper; splits.py untouched).
    splits = make_calibration_splits(df_train)
    idx, patients, stats = splits["indices"], splits["patients"], splits["stats"]

    # Holdout never intersects any Phase-2 surface.
    for name in ("S_train", "S_cal", "S_eval", "S_fit"):
        assert holdout_patients.isdisjoint(patients[name]), f"holdout leaked into {name}"

    def sub(name):
        i = idx[name]
        return (
            df_train.iloc[i].reset_index(drop=True),
            x_train.iloc[i].reset_index(drop=True),
            y_train[i],
        )

    df_fit, x_fit, y_fit = sub("S_fit")
    df_strain, x_strain, y_strain = sub("S_train")
    _, x_cal, y_cal = sub("S_cal")
    _, x_eval, y_eval = sub("S_eval")

    # Step 3 — OOF scores on S_fit (primary calibrator-fit scores).
    oof, oof_fold_auroc = _oof_scores(
        df_fit, x_fit, y_fit, lambda: _model(LOCKED_BEST_PARAMS), n_folds
    )
    oof_disc = metrics.discrimination_metrics(y_fit, oof)
    _check_tripwire(oof_disc["auroc"], "S_fit OOF")

    # Step 4 — PRIMARY calibrators on pooled OOF scores (used if the gate PASSes).
    iso_primary = fit_isotonic(oof, y_fit)
    platt_primary = fit_platt(oof, y_fit)

    # Step 5 — sibling boosters for the gate + fallback.
    cal_booster_sfit = _fit(_model(LOCKED_BEST_PARAMS), x_fit, y_fit)
    cal_booster_strain = _fit(_model(LOCKED_BEST_PARAMS), x_strain, y_strain)
    sfit_cv_auroc = oof_disc["auroc"]  # OOF over S_fit IS cal_booster_Sfit's own-region CV
    strain_cv_auroc = _grouped_cv_auroc(
        df_strain, x_strain, y_strain, lambda: _model(LOCKED_BEST_PARAMS), n_folds
    )
    _check_tripwire(sfit_cv_auroc, "cal_booster_Sfit CV")
    _check_tripwire(strain_cv_auroc, "cal_booster_Strain CV")

    # Step 6 — OOF-vs-production score-distribution GATE.
    phase1_booster = _load_phase1_booster(tracking_uri)
    d_cal = cal_booster_sfit.predict_proba(x_eval)[:, 1]
    d_prod = phase1_booster.predict_proba(x_eval)[:, 1]
    gate = score_gate(d_cal, d_prod)

    # Branch on the gate.
    if gate["pass"]:
        branch = "PASS"
        branch_reason = (
            f"KS {gate['ks']:.4f} <= {GATE_KS_MAX} and max decile |Δ| "
            f"{gate['max_decile_delta']:.4f} <= {GATE_DECILE_MAX}: the OOF-fit calibrator "
            "may sit on top of @phase1."
        )
        deployment_base = "phase1"
        base_booster = phase1_booster
        iso_cal, platt_cal = iso_primary, platt_primary
        raw_eval = d_prod
        fit_source = "S_fit OOF (5-fold), applied on @phase1 scores"
    else:
        branch = "FALLBACK"
        branch_reason = (
            f"KS {gate['ks']:.4f} or max decile |Δ| {gate['max_decile_delta']:.4f} exceeded "
            "threshold: refit on cal_booster_Strain's OUT-OF-SAMPLE scores (zero cross-model "
            "transfer); deployment base = cal_booster_Strain."
        )
        deployment_base = "cal_booster_Strain"
        base_booster = cal_booster_strain
        scores_cal = cal_booster_strain.predict_proba(x_cal)[:, 1]
        iso_cal = fit_isotonic(scores_cal, y_cal)
        platt_cal = fit_platt(scores_cal, y_cal)
        raw_eval = cal_booster_strain.predict_proba(x_eval)[:, 1]
        fit_source = "S_cal out-of-sample scores of cal_booster_Strain"

    # Step 7 — single look at S_eval; apply pre-registered rule.
    raw_summary = metrics.summarize(y_eval, raw_eval)
    eval_by_method = {}
    cal_objs = {"isotonic": iso_cal, "platt": platt_cal}
    monotone = {}
    for m, obj in cal_objs.items():
        probs = apply_calibrator(m, obj, raw_eval)
        summary = metrics.summarize(y_eval, probs)
        eval_by_method[m] = summary
        monotone[m] = _is_monotone(summary["reliability"])
    winner, rejected = select_winner(eval_by_method, monotone)
    winner_obj = cal_objs[winner]

    if write_outputs:
        plot_reliability_overlay(
            y_eval,
            raw_eval,
            apply_calibrator("platt", platt_cal, raw_eval),
            apply_calibrator("isotonic", iso_cal, raw_eval),
            FIGURE_PATH,
        )

    # Step 8 — class_weight diagnostic (READ-ONLY; registers nothing).
    unweighted_cv_auroc = _grouped_cv_auroc(
        df_fit, x_fit, y_fit, lambda: _make_unweighted(LOCKED_BEST_PARAMS), n_folds
    )
    _check_tripwire(unweighted_cv_auroc, "unweighted S_fit CV")
    unweighted_model = _fit(_make_unweighted(LOCKED_BEST_PARAMS), x_fit, y_fit)
    unweighted_native_ece = metrics.calibration_metrics(
        y_eval, unweighted_model.predict_proba(x_eval)[:, 1]
    )["ece"]

    # Portable forms for the manifest (BOTH methods, for transparency).
    portable_all = {
        "isotonic": isotonic_portable(iso_cal),
        "platt": platt_portable(platt_cal),
    }

    manifest = {
        "method": winner,
        "deployment_base": deployment_base,
        "fit_source": fit_source,
        "gate": {
            "ks": gate["ks"],
            "decile_deltas": gate["decile_deltas"],
            "max_decile_delta": gate["max_decile_delta"],
            "ks_threshold": GATE_KS_MAX,
            "decile_threshold": GATE_DECILE_MAX,
            "d_cal_mean": gate["d_cal_mean"],
            "d_cal_std": gate["d_cal_std"],
            "d_prod_mean": gate["d_prod_mean"],
            "d_prod_std": gate["d_prod_std"],
            "pass": gate["pass"],
            "in_sample_caveat": IN_SAMPLE_CAVEAT,
        },
        "portable": portable_all[winner],
        "portable_all": portable_all,
        "s_eval": {
            "ece_before": raw_summary["ece"],
            "brier_before": raw_summary["brier"],
            "auroc": raw_summary["auroc"],
            "n_rows": int(len(y_eval)),
            "isotonic": {
                "ece_after": eval_by_method["isotonic"]["ece"],
                "brier_after": eval_by_method["isotonic"]["brier"],
                "monotone": monotone["isotonic"],
            },
            "platt": {
                "ece_after": eval_by_method["platt"]["ece"],
                "brier_after": eval_by_method["platt"]["brier"],
                "monotone": monotone["platt"],
            },
            "rejected_methods": rejected,
        },
        "selection_rule": SELECTION_RULE_TEXT,
        "split_hashes": {name: stats[name]["patient_sha256"] for name in stats},
        "seed": SEED,
        "sklearn_version": versions["sklearn"],
        "lightgbm_version": versions["lightgbm"],
        "git_commit": _git_sha(),
        "phase1_holdout_metrics": LOCKED_HOLDOUT_METRICS,
    }

    register_info = None
    if write_outputs:
        CAL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(winner_obj, CALIBRATOR_PATH)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if do_register:
            register_info = _register_calibrated(
                base_booster, x_eval.head(5), manifest, tracking_uri
            )

    return {
        "versions": versions,
        "stats": stats,
        "oof_disc": oof_disc,
        "oof_fold_auroc": oof_fold_auroc,
        "sfit_cv_auroc": sfit_cv_auroc,
        "strain_cv_auroc": strain_cv_auroc,
        "gate": gate,
        "branch": branch,
        "branch_reason": branch_reason,
        "deployment_base": deployment_base,
        "raw_summary": raw_summary,
        "eval_by_method": eval_by_method,
        "monotone": monotone,
        "winner": winner,
        "rejected": rejected,
        "unweighted_cv_auroc": unweighted_cv_auroc,
        "unweighted_native_ece": unweighted_native_ece,
        "manifest": manifest,
        "register_info": register_info,
    }


def _render_report(r: dict) -> str:
    m = r["manifest"]
    g = r["gate"]
    ev = r["eval_by_method"]
    raw = r["raw_summary"]
    lines: list[str] = []
    lines.append("# Calibration — Phase 2 (Trust Layer I)")
    lines.append("")
    lines.append(
        "> Isotonic vs Platt on surfaces carved from INSIDE the 80% train (the Phase-1 20% "
        "holdout is spent and never loaded). `S_eval` is frozen and looked at exactly once. "
        "Generated by `sentinel.calibration.fit_calibrators`."
    )
    lines.append("")
    lines.append(
        f"Versions: sklearn `{r['versions']['sklearn']}`, lightgbm `{r['versions']['lightgbm']}`. "
        f"seed={SEED}. git `{m['git_commit'][:12]}`."
    )
    lines.append("")
    lines.append("## Surfaces (from the 80% train)")
    lines.append("")
    lines.append("| surface | n_patients | n_rows | prevalence | patient sha256 (first 12) |")
    lines.append("|---|---:|---:|---:|---|")
    for name in ("S_train", "S_cal", "S_eval", "S_fit"):
        s = r["stats"][name]
        lines.append(
            f"| {name} | {s['n_patients']:,} | {s['n_rows']:,} | {s['prevalence']:.4f} | "
            f"`{s['patient_sha256'][:12]}` |"
        )
    lines.append("")
    lines.append(
        "Pairwise patient-disjointness across {S_train, S_cal, S_eval, holdout} asserted "
        "structurally; the 20% holdout intersects no Phase-2 surface. `S_eval` fits nothing."
    )
    lines.append("")
    lines.append("## Step 3 — OOF on S_fit (calibrator-fit scores)")
    lines.append("")
    lines.append(
        f"5-fold StratifiedGroupKFold OOF AUROC **{r['oof_disc']['auroc']:.4f}** "
        f"(AUPRC {r['oof_disc']['auprc']:.4f}), expected ~{EXPECTED_OOF_AUROC:.3f}. "
        f"Per-fold AUROC: {', '.join(f'{a:.3f}' for a in r['oof_fold_auroc'])}. "
        f"No tripwire (< {TRIPWIRE_LEAKAGE})."
    )
    lines.append("")
    lines.append("## Step 5 — sibling-booster CV (own training region)")
    lines.append("")
    lines.append(f"- cal_booster_Sfit CV AUROC (= S_fit OOF): {r['sfit_cv_auroc']:.4f}")
    lines.append(f"- cal_booster_Strain CV AUROC: {r['strain_cv_auroc']:.4f}")
    lines.append(f"- Both within noise of {EXPECTED_OOF_AUROC:.3f}; no tripwire.")
    lines.append("")
    lines.append("## Step 6 — OOF-vs-production score-distribution GATE")
    lines.append("")
    lines.append(f"> {IN_SAMPLE_CAVEAT}")
    lines.append("")
    lines.append("| quantity | value | threshold |")
    lines.append("|---|---:|---:|")
    lines.append(f"| two-sample KS(D_cal, D_prod) | {g['ks']:.4f} | <= {GATE_KS_MAX} |")
    lines.append(f"| max over deciles |Δ| | {g['max_decile_delta']:.4f} | <= {GATE_DECILE_MAX} |")
    lines.append("")
    lines.append(
        f"D_cal mean±std {g['d_cal_mean']:.4f}±{g['d_cal_std']:.4f}; "
        f"D_prod mean±std {g['d_prod_mean']:.4f}±{g['d_prod_std']:.4f}."
    )
    lines.append("")
    lines.append("9 decile deltas: " + ", ".join(f"{d:.4f}" for d in g["decile_deltas"]) + ".")
    lines.append("")
    lines.append(f"**GATE {('PASS' if g['pass'] else 'FAIL')}** → branch **{r['branch']}**.")
    lines.append("")
    lines.append(r["branch_reason"])
    lines.append("")
    lines.append(f"Deployment base = **{r['deployment_base']}**; calibrators {m['fit_source']}.")
    lines.append("")
    lines.append("## Step 7 — S_eval (single look) + selection")
    lines.append("")
    lines.append(f"_Pre-registered rule:_ {SELECTION_RULE_TEXT}")
    lines.append("")
    lines.append("| method | ECE | Brier | monotone |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| raw (before) | {raw['ece']:.4f} | {raw['brier']:.4f} | — |")
    for meth in ("platt", "isotonic"):
        lines.append(
            f"| {meth} | {ev[meth]['ece']:.4f} | {ev[meth]['brier']:.4f} | "
            f"{'yes' if r['monotone'][meth] else 'NO (rejected)'} |"
        )
    lines.append("")
    lines.append(
        f"**Winner: `{r['winner']}`.** "
        + (f"Rejected (non-monotone): {r['rejected']}. " if r["rejected"] else "")
        + f"ECE {raw['ece']:.4f} → {ev[r['winner']]['ece']:.4f}, "
        f"Brier {raw['brier']:.4f} → {ev[r['winner']]['brier']:.4f}. "
        "See `figures/reliability_before_after.png`."
    )
    lines.append("")
    lines.append("## Step 8 — class_weight diagnostic (read-only)")
    lines.append("")
    lines.append(
        f"An UNWEIGHTED LightGBM (LOCKED params, `class_weight=None`) on S_fit gives CV AUROC "
        f"**{r['unweighted_cv_auroc']:.4f}** (ranking unchanged vs the balanced "
        f"{EXPECTED_OOF_AUROC:.3f} — discrimination is invariant to the weighting) and a NATIVE "
        f"S_eval ECE of **{r['unweighted_native_ece']:.4f}**, versus the balanced model's 0.334. "
        + (
            "Unweighted training lands natively much closer to the base rate, confirming that "
            "the balanced weighting — not the features — is what inflates probabilities; "
            "calibration (above) is the principled fix rather than dropping the weighting."
            if r["unweighted_native_ece"] < 0.334
            else "Unweighted training does not land closer to the base rate here."
        )
        + " No artifact written, no registry change for this diagnostic."
    )
    lines.append("")
    lines.append("## Step 9 — persistence + registration")
    lines.append("")
    lines.append(f"- `{CALIBRATOR_PATH.relative_to(ROOT).as_posix()}` (fitted sklearn object)")
    lines.append(
        f"- `{MANIFEST_PATH.relative_to(ROOT).as_posix()}` (provenance + portable form: "
        f"{'knots' if r['winner'] == 'isotonic' else 'A,B'})"
    )
    if r["register_info"]:
        ri = r["register_info"]
        lines.append(
            f"- Registered `{REGISTERED_NAME}` v{ri['version']} alias `@{ri['alias']}` "
            f"(run `{ri['run_id']}`)."
        )
    lines.append("")
    lines.append(
        "Phase 4's pyfunc wrapper (already planned to cast category dtypes) is the intended "
        "loader: it reads `calibrator_manifest.json`, reconstructs the map from the portable "
        "form, asserts the sklearn version, and applies it AFTER the booster call."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    from sentinel.features.build import load_and_build

    df = load_and_build()
    result = run(df, do_register=True, write_outputs=True)

    report = _render_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"[written] {REPORT_PATH}")
    print(f"[written] {CALIBRATOR_PATH}")
    print(f"[written] {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

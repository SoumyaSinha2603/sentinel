"""Phase 2 / W6 — clinical-utility evaluation entrypoint (read-only).

Evaluates the shipped calibrated-probability path (@phase1 booster + committed isotonic
manifest) on the FROZEN `S_eval` surface: decision-curve analysis, precision@k /
recall@k / lift / alert-burden, and fixed-budget operating points. Trains nothing,
registers nothing, and never loads the Phase-1 holdout.

Run:

    python -m sentinel.clinical_utility.evaluate
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from sentinel.calibration.calibration_splits import make_calibration_splits
from sentinel.clinical_utility import calibrated, dca, ranking
from sentinel.config import MODELS_DIR, REPORTS_DIR, ROOT
from sentinel.evaluation.splits import GROUP_COL, make_binary_target, make_holdout_split
from sentinel.features.build import load_and_build

REQUIRED_SKLEARN = "1.7.2"
DOGFOOD_TOL = 1e-12
INVARIANCE_TOL = 1e-9

OPERATING_POINTS_PATH = MODELS_DIR / "clinical" / "operating_points.json"
FIG_DIR = REPORTS_DIR / "figures"
REPORT_PATH = REPORTS_DIR / "clinical_utility_results.md"

# Verbatim surface-reuse note (W6 spec) — copied into the report and the committed JSON.
SURFACE_REUSE_NOTE = (
    "S_eval was used to SELECT the calibrator in W5. Reusing it here means DCA inherits the "
    "calibrator-selection optimism — negligible, since selection was a two-way near-tie "
    "(ECE 0.0264 vs 0.0272). precision@k / recall@k are RANKING-based and calibration-"
    "invariant, so they carry ZERO selection optimism. The model never trained on S_eval. "
    "No cleaner internal surface remains; the honest move is reuse-with-documentation, not "
    "re-fragmenting the data."
)

INVARIANCE_SENTENCE = (
    "calibration changes displayed confidence and DCA, NOT who is on the worklist."
)


def assert_env() -> dict:
    assert sklearn.__version__ == REQUIRED_SKLEARN, sklearn.__version__
    return {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _recover_s_eval(df: pd.DataFrame, manifest: dict) -> dict:
    """Recover the frozen S_eval surface and prove it is the identical W5 partition."""
    train_idx, holdout_idx = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    holdout_patients = {str(p) for p in df[GROUP_COL].to_numpy()[holdout_idx]}

    splits = make_calibration_splits(df_train)
    s_eval_hash = splits["stats"]["S_eval"]["patient_sha256"]
    expected = manifest["split_hashes"]["S_eval"]
    if s_eval_hash != expected:
        raise AssertionError(
            f"S_eval patient hash drift: recomputed {s_eval_hash} != manifest {expected}"
        )

    idx = splits["indices"]["S_eval"]
    df_eval = df_train.iloc[idx].reset_index(drop=True)
    s_eval_patients = splits["patients"]["S_eval"]
    overlap = s_eval_patients & holdout_patients
    if overlap:
        raise AssertionError(f"S_eval ∩ holdout non-empty: {len(overlap)} patients")

    return {
        "df_eval": df_eval,
        "s_eval_hash": s_eval_hash,
        "patient_ids": df_eval[GROUP_COL].to_numpy(),
    }


def compute(df: pd.DataFrame, *, tracking_uri: str | None = None, n_boot: int = 1000) -> dict:
    """Run the full W6 computation and return everything needed for JSON/figures/report."""
    versions = assert_env()
    manifest = calibrated.load_manifest()

    rec = _recover_s_eval(df, manifest)
    df_eval = rec["df_eval"]
    patient_ids = rec["patient_ids"]
    y = make_binary_target(df_eval).to_numpy()

    booster = calibrated.load_phase1_booster(tracking_uri)
    p_raw = calibrated.raw_proba(df_eval, booster)
    p_cal = calibrated.get_calibrated_proba(df_eval, booster=booster, manifest=manifest)

    # Step 0 dogfood: committed joblib must match the manifest-reconstructed map at use.
    iso_joblib = joblib.load(calibrated.CALIBRATOR_JOBLIB)
    joblib_pred = np.asarray(iso_joblib.predict(p_raw), dtype=float)
    dogfood_max_diff = float(np.max(np.abs(joblib_pred - p_cal)))
    if dogfood_max_diff > DOGFOOD_TOL:
        raise AssertionError(
            f"calibrated path != committed joblib: max abs diff {dogfood_max_diff:.2e}"
        )

    n = int(len(y))
    n_pos = int(np.sum(y == 1))
    prevalence = n_pos / n

    # Step 2 — DCA on calibrated probs.
    grid = dca.dca_grid(y, p_cal, dca.THRESHOLDS)
    ci_lower, ci_upper = dca.dca_bootstrap_ci(y, p_cal, patient_ids, dca.THRESHOLDS, n_boot=n_boot)
    band = dca.useful_band(grid)

    # Step 3 — ranking metrics + calibration-invariance.
    order = ranking.worklist_order(p_cal, p_raw)
    k_curve = ranking.precision_recall_at_k(y, order, ranking.K_GRID)
    invariance_max_diff = ranking.precision_recall_invariant(y, p_cal, p_raw, ranking.K_GRID)
    if invariance_max_diff > INVARIANCE_TOL:
        raise AssertionError(
            f"precision@k not calibration-invariant: max abs diff {invariance_max_diff:.2e}"
        )

    # Step 4 — fixed-budget operating points.
    op_rows = ranking.precision_recall_at_k(y, order, ranking.BUDGETS)
    operating_points = []
    for k, row in zip(ranking.BUDGETS, op_rows, strict=True):
        ci = ranking.bootstrap_pr_ci(y, p_cal, p_raw, patient_ids, k, n_boot=n_boot)
        operating_points.append(
            {
                "budget": float(k),
                "alerts_per_100": round(k * 100, 1),
                "n_flag": row["n_flag"],
                "precision": row["precision"],
                "recall": row["recall"],
                "lift": row["lift"],
                "nnf": row["nnf"],
                "implied_threshold": ranking.implied_threshold(p_cal, order, k),
                "precision_ci": ci["precision_ci"],
                "recall_ci": ci["recall_ci"],
            }
        )

    return {
        "versions": versions,
        "manifest": manifest,
        "s_eval_hash": rec["s_eval_hash"],
        "dogfood_max_diff": dogfood_max_diff,
        "invariance_max_diff": invariance_max_diff,
        "n": n,
        "n_pos": n_pos,
        "prevalence": prevalence,
        "grid": grid,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "band": band,
        "k_curve": k_curve,
        "operating_points": operating_points,
        "git_commit": _git_sha(),
    }


# --- Persistence ----------------------------------------------------------------------
def build_operating_points_json(r: dict) -> dict:
    g = r["grid"]
    return {
        "operating_points": r["operating_points"],
        "k_grid_curve": [
            {
                "k": row["k"],
                "precision": row["precision"],
                "recall": row["recall"],
                "lift": row["lift"],
                "alert_rate": row["alert_rate"],
                "nnf": row["nnf"],
                "n_flag": row["n_flag"],
            }
            for row in r["k_curve"]
        ],
        "dca_grid": [
            {
                "p_t": float(t),
                "nb_model": float(m),
                "nb_all": float(a),
                "nb_none": float(none),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
            }
            for t, m, a, none, lo, hi in zip(
                g["thresholds"],
                g["nb_model"],
                g["nb_all"],
                g["nb_none"],
                r["ci_lower"],
                r["ci_upper"],
                strict=True,
            )
        ],
        "prevalence": r["prevalence"],
        "n": r["n"],
        "n_pos": r["n_pos"],
        "useful_band": r["band"],
        "calibration_invariance_max_abs_diff": r["invariance_max_diff"],
        "s_eval_patient_sha256": r["s_eval_hash"],
        "seed": 42,
        "sklearn_version": r["versions"]["sklearn"],
        "lightgbm_version": r["versions"]["lightgbm"],
        "git_commit": r["git_commit"],
        "surface_reuse_note": SURFACE_REUSE_NOTE,
    }


# --- Figures --------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_decision_curve(r: dict, path: Path) -> None:
    plt = _plt()
    g = r["grid"]
    thr = g["thresholds"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(thr, r["ci_lower"], r["ci_upper"], color="C0", alpha=0.2, label="model 95% CI")
    ax.plot(thr, g["nb_model"], color="C0", label="treat per model (calibrated)")
    ax.plot(thr, g["nb_all"], color="C1", linestyle="--", label="treat all")
    ax.plot(thr, g["nb_none"], color="grey", linestyle=":", label="treat none")
    if r["band"]["any"]:
        ax.axvspan(r["band"]["min"], r["band"]["max"], color="C2", alpha=0.10, label="useful band")
    ax.set_xlabel("threshold probability $p_t$ (risk tolerance)")
    ax.set_ylabel("net benefit")
    ax.set_ylim(min(-0.02, float(np.min(g["nb_model"]))), float(np.max(g["nb_model"])) * 1.2 + 1e-3)
    ax.set_title(f"Decision curve — S_eval (prevalence={r['prevalence']:.4f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.98,
        0.02,
        "DCA on calibrated probs; p_t is an odds axis",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="grey",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mark_points(ax, xs, ys):
    for x, y in zip(xs, ys, strict=True):
        ax.scatter([x], [y], color="C3", zorder=5)
        ax.annotate(f"{x:.0%}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)


def plot_alert_burden(r: dict, path: Path) -> None:
    plt = _plt()
    ks = [row["alert_rate"] for row in r["k_curve"]]
    recalls = [row["recall"] for row in r["k_curve"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, recalls, color="C0", label="recall @ alert rate")
    op = {p["budget"]: p["recall"] for p in r["operating_points"]}
    _mark_points(ax, list(op.keys()), list(op.values()))
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="alerting at random")
    ax.set_xlabel("alert rate (fraction of encounters flagged)")
    ax.set_ylabel("recall (% of 30-day readmissions captured)")
    ax.set_title("Alert-burden curve — S_eval")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.02,
        0.98,
        INVARIANCE_SENTENCE,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="grey",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_precision_recall_at_k(r: dict, path: Path) -> None:
    plt = _plt()
    ks = [row["k"] for row in r["k_curve"]]
    precisions = [row["precision"] for row in r["k_curve"]]
    recalls = [row["recall"] for row in r["k_curve"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, precisions, color="C0", label="precision@k (PPV)")
    ax.plot(ks, recalls, color="C1", label="recall@k (sensitivity)")
    ax.axhline(
        r["prevalence"], color="grey", linestyle="--", label=f"prevalence={r['prevalence']:.4f}"
    )
    op_p = {p["budget"]: p["precision"] for p in r["operating_points"]}
    op_r = {p["budget"]: p["recall"] for p in r["operating_points"]}
    _mark_points(ax, list(op_p.keys()), list(op_p.values()))
    _mark_points(ax, list(op_r.keys()), list(op_r.values()))
    ax.set_xlabel("k (top fraction of worklist)")
    ax.set_ylabel("precision / recall")
    ax.set_title("precision@k and recall@k — S_eval")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Report ---------------------------------------------------------------------------
def _render_report(r: dict) -> str:
    band = r["band"]
    lines: list[str] = []
    lines.append("# Clinical utility — Phase 2 / W6 (DCA, precision@k, operating points)")
    lines.append("")
    lines.append(
        "> Read-only evaluation of the shipped calibrated path (@phase1 booster + committed "
        "isotonic manifest) on the FROZEN `S_eval` surface. No model trained or registered; "
        "the Phase-1 holdout was never loaded. Generated by `sentinel.clinical_utility.evaluate`."
    )
    lines.append("")
    lines.append(
        f"Versions: sklearn `{r['versions']['sklearn']}`, lightgbm `{r['versions']['lightgbm']}`. "
        f"seed=42. git `{r['git_commit'][:12]}`. "
        f"S_eval patient sha256 `{r['s_eval_hash'][:12]}` (matches W5 manifest)."
    )
    lines.append("")
    lines.append(
        f"S_eval: N={r['n']:,}, positives={r['n_pos']:,}, prevalence={r['prevalence']:.4f}."
    )
    lines.append("")
    lines.append("## Surface-reuse note")
    lines.append("")
    lines.append(SURFACE_REUSE_NOTE)
    lines.append("")
    lines.append("## Dogfood — Phase-4 calibrated path")
    lines.append("")
    lines.append(
        f"`get_calibrated_proba` reconstructs the isotonic map from the committed manifest's "
        f"portable knots (not the joblib) and matches the committed joblib on the S_eval scores "
        f"to within **{r['dogfood_max_diff']:.1e}** (tol {DOGFOOD_TOL:.0e}) — re-confirms W5 "
        "CHECK 2 at point of use."
    )
    lines.append("")
    lines.append("## Decision-curve analysis (net benefit)")
    lines.append("")
    lines.append(
        "DCA is computed on **calibrated** probabilities — the threshold axis `p_t` is an "
        "expected-utility (odds) axis and is only meaningful on calibrated probs."
    )
    lines.append("")
    if band["any"]:
        contig = "" if band["contiguous"] else " (non-contiguous; range shown is min–max)"
        thr = r["grid"]["thresholds"]
        spans_full = band["min"] <= float(thr[0]) + 1e-9 and band["max"] >= float(thr[-1]) - 1e-9
        lines.append(
            f"NB_model beats BOTH treat-all and treat-none over **p_t ∈ "
            f"[{band['min']:.3f}, {band['max']:.3f}]**{contig} — the clinically useful "
            "risk-tolerance band."
        )
        if spans_full:
            lines.append("")
            lines.append(
                "This covers the **entire evaluated grid**: across all clinically plausible "
                "thresholds, acting on the model dominates both alternatives. The absolute net "
                "benefit shrinks toward treat-none as `p_t` rises (fewer patients clear the bar, "
                "so the odds penalty on false positives bites), but stays above treat-all "
                "throughout. Only at `p_t → 0` (below the grid, where flagging everyone is "
                "optimal) does treat-all converge to the model."
            )
        else:
            lines.append("")
            lines.append(
                "Below that band treat-all is competitive (cheap to act); above it the model's "
                "net benefit collapses toward treat-none as the odds penalty on false positives "
                "grows."
            )
    else:
        lines.append("NB_model never strictly beats both alternatives across the grid.")
    lines.append("")
    lines.append("## Operating points (fixed budgets, no data-snooping)")
    lines.append("")
    lines.append(
        "Budgets are pre-declared care-team capacities (5/10/20% of the worklist), not the "
        "best-looking points. CIs are patient-grouped bootstrap (B=1000, seed=42)."
    )
    lines.append("")
    lines.append(
        "| budget | alerts/100 | precision (95% CI) | recall (95% CI) | lift | NNF | implied p_t |"
    )
    lines.append("|---:|---:|---|---|---:|---:|---:|")
    for p in r["operating_points"]:
        pc = p["precision_ci"]
        rc = p["recall_ci"]
        lines.append(
            f"| {p['budget']:.0%} | {p['alerts_per_100']:.0f} | "
            f"{p['precision']:.3f} [{pc[0]:.3f}, {pc[1]:.3f}] | "
            f"{p['recall']:.3f} [{rc[0]:.3f}, {rc[1]:.3f}] | "
            f"{p['lift']:.2f} | {p['nnf']:.1f} | {p['implied_threshold']:.4f} |"
        )
    lines.append("")
    lines.append(
        "The implied probability threshold connects the top-k worklist view to the DCA "
        "threshold axis: flagging the top-k% is equivalent to acting above that calibrated risk."
    )
    lines.append("")
    lines.append("## Calibration invariance")
    lines.append("")
    lines.append(
        f"precision@k computed on the calibrated ordering equals that on the raw @phase1 "
        f"ordering to within **{r['invariance_max_diff']:.1e}** across the whole k-grid "
        f"(isotonic is monotone → identical ranking). In one line: **{INVARIANCE_SENTENCE}** "
        "Ties in calibrated probability (isotonic's flat regions) are broken by the underlying "
        "raw score, which is why the worklist is invariant."
    )
    lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    op = {p["budget"]: p for p in r["operating_points"]}
    r10 = op[0.10]["recall"]
    r20 = op[0.20]["recall"]
    lines.append(
        f"Discrimination is modest (~0.67 AUROC), which caps achievable recall at low alert "
        f"rates: flagging the top 10% of encounters captures **{r10:.1%}** of 30-day "
        f"readmissions, the top 20% captures **{r20:.1%}**. These are real, un-dressed numbers "
        "— useful for triage relative to treat-all/treat-none over the band above, but no "
        "substitute for the discrimination ceiling. The cleanest performance number will come "
        "from an external validation set in Phase 8, not from this internal surface."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `figures/decision_curve.png` — NB_model (+95% CI), treat-all, treat-none.")
    lines.append("- `figures/alert_burden_curve.png` — recall vs alert rate, 5/10/20% marked.")
    lines.append("- `figures/precision_recall_at_k.png` — precision@k & recall@k vs k.")
    lines.append("")
    lines.append("## Persistence")
    lines.append("")
    lines.append(
        f"- `{OPERATING_POINTS_PATH.relative_to(ROOT).as_posix()}` — operating points, full "
        "k-grid curve, DCA grid (with CI), provenance + surface-reuse note. This is the "
        "committed source of truth Phase-5's worklist UI + alert-burden control read "
        "(`mlruns/` is not load-bearing)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_and_build()
    r = compute(df)

    OPERATING_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPERATING_POINTS_PATH.write_text(
        json.dumps(build_operating_points_json(r), indent=2) + "\n", encoding="utf-8"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_decision_curve(r, FIG_DIR / "decision_curve.png")
    plot_alert_burden(r, FIG_DIR / "alert_burden_curve.png")
    plot_precision_recall_at_k(r, FIG_DIR / "precision_recall_at_k.png")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(r), encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(_render_report(r))
    print(f"[written] {OPERATING_POINTS_PATH}")
    print(f"[written] {REPORT_PATH}")


if __name__ == "__main__":
    main()

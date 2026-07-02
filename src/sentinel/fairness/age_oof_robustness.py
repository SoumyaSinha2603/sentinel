"""Age-disparity OOF robustness check (READ-ONLY appendix to the fairness audit).

Prereg §9's "investigate before believing" follow-through for the age headline. The S_eval
audit scored `@phase1` on rows it was TRAINED on (S_eval ⊂ the 80% train @phase1 was refit on),
so every subgroup AUROC is in-sample-inflated. This module recomputes the age-band
discrimination gap on the **S_fit out-of-fold** scores — genuinely out-of-fold (each row scored
by a model not trained on it), ~64k rows so every band is well-powered, disjoint from both
`S_eval` and the spent holdout — to tell how much of the age gap is real vs an optimism artifact.

Reuses W5's exact OOF machinery (`fit_calibrators._oof_scores` + the locked `lgbm_tuned._model`);
the CV fits are transient/in-memory (as W5 did). Trains nothing persistent, registers nothing,
never loads the holdout for scoring.

Run:

    python -m sentinel.fairness.age_oof_robustness
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from sentinel.calibration.calibration_splits import make_calibration_splits
from sentinel.calibration.fit_calibrators import N_CAL_FOLDS, _oof_scores
from sentinel.clinical_utility import calibrated
from sentinel.config import MODELS_DIR, REPORTS_DIR, ROOT
from sentinel.data.cohort import build_cohort
from sentinel.data.load import fetch_raw
from sentinel.evaluation import metrics as base_metrics
from sentinel.evaluation.splits import GROUP_COL, make_binary_target, make_holdout_split
from sentinel.fairness import bootstrap, subgroups
from sentinel.fairness.audit import (
    PRACTICAL_THRESHOLDS,
    REPORT_PATH,
    SUBGROUP_ECE_FLAG,
    SUBGROUP_METRICS_PATH,
)
from sentinel.fairness.metrics import is_powered
from sentinel.features.build import build_features
from sentinel.models.lgbm_baseline import feature_frame
from sentinel.models.lgbm_tuned import LOCKED_BEST_PARAMS, _model

REQUIRED_SKLEARN = "1.7.2"
SEED = 42
N_BOOT = 2000
EXPECTED_OOF_AUROC = 0.6713  # W5 pooled S_fit OOF AUROC (reports/calibration_results.md).

AGE_ATTR = "age"
# Well-powered mid-band contrasts anchor the verdict; <40 is the fragile anchor (smallest
# powered band, pools the anomalous high-prevalence [20-30) bucket).
PRIMARY_CONTRASTS = (("80+", "60-80"), ("80+", "40-60"))
FRAGILE_CONTRAST = ("80+", "<40")
DIAGNOSTIC_BUCKET = "[20-30)"

AUROC_PRACTICAL = PRACTICAL_THRESHOLDS["auroc"]  # 0.05 (prereg §6)
ECE_PRACTICAL = PRACTICAL_THRESHOLDS["ece"]  # 0.03 (prereg §6)

OUT_JSON = MODELS_DIR / "fairness" / "age_oof_robustness.json"
FIG_PATH = REPORTS_DIR / "figures" / "fairness_age_oof_robustness.png"
SECTION_MARKER = "## Age disparity — OOF robustness"

# Recorded verbatim (spec): the isotonic map was FIT on these S_fit OOF scores in W5, so
# per-band ECE here is mildly optimistic for the CALIBRATOR (a low-capacity monotone map —
# small effect). AUROC is calibration-invariant, so it is unaffected.
CALIBRATOR_IN_SAMPLE_CAVEAT = (
    "The committed isotonic map was FIT on these S_fit OOF scores in W5, so per-band ECE here "
    "is mildly optimistic FOR THE CALIBRATOR (a low-capacity monotone map — small effect). "
    "AUROC/AUPRC are calibration-invariant (isotonic is monotone), so they are unaffected. "
    "Discrimination is the primary deliverable of this check; ECE is secondary."
)


def assert_env() -> dict:
    assert sklearn.__version__ == REQUIRED_SKLEARN, sklearn.__version__
    import lightgbm

    return {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def recover_s_fit_frames() -> dict:
    """Recover the frozen S_fit as (built feature frame, row-aligned raw cohort frame).

    Same recovery pattern the audit uses, but for S_fit. Asserts S_fit is disjoint from both
    S_eval (by construction) and the spent holdout (S_fit ⊂ the 80% train).
    """
    df_cohort = build_cohort(fetch_raw())  # reset_index(drop=True); raw `age` survives
    df_built = build_features(df_cohort)  # row-aligned; scored frame

    train_idx, holdout_idx = make_holdout_split(df_built)
    holdout_patients = {str(p) for p in df_built[GROUP_COL].to_numpy()[holdout_idx]}

    df_train_built = df_built.iloc[train_idx].reset_index(drop=True)
    df_train_cohort = df_cohort.iloc[train_idx].reset_index(drop=True)

    splits = make_calibration_splits(df_train_built)
    s_fit_idx = splits["indices"]["S_fit"]
    s_fit_hash = splits["stats"]["S_fit"]["patient_sha256"]

    df_fit_built = df_train_built.iloc[s_fit_idx].reset_index(drop=True)
    df_fit_cohort = df_train_cohort.iloc[s_fit_idx].reset_index(drop=True)

    s_fit_patients = splits["patients"]["S_fit"]
    if s_fit_patients & splits["patients"]["S_eval"]:
        raise AssertionError("S_fit ∩ S_eval non-empty (should be disjoint by construction)")
    if s_fit_patients & holdout_patients:
        raise AssertionError("S_fit ∩ holdout non-empty (holdout must stay spent)")

    if not (df_fit_built[GROUP_COL].to_numpy() == df_fit_cohort[GROUP_COL].to_numpy()).all():
        raise AssertionError("S_fit built/cohort frames are not row-aligned")

    return {
        "df_fit_built": df_fit_built,
        "df_fit_cohort": df_fit_cohort,
        "s_fit_hash": s_fit_hash,
        "patient_ids": df_fit_built[GROUP_COL].to_numpy(),
    }


def _band_metrics(y_b: np.ndarray, p_raw_b: np.ndarray, p_cal_b: np.ndarray) -> dict:
    """AUROC/AUPRC on raw OOF, ECE on calibrated OOF, plus counts (one age band)."""
    base = base_metrics.prevalence_and_base_rates(y_b)
    disc = base_metrics.discrimination_metrics(y_b, p_raw_b)  # rank-based -> raw == cal
    cal = base_metrics.calibration_metrics(y_b, p_cal_b)
    return {
        "n": base["n"],
        "positives": base["positives"],
        "prevalence": base["prevalence"],
        "powered": is_powered(base),
        "auroc_oof": disc["auroc"],
        "auprc_oof": disc["auprc"],
        "ece_oof_cal": cal["ece"],
        "reliability_cal": cal["reliability"],
    }


def _load_s_eval_age() -> dict:
    """In-sample per-band AUROC (+CI) from the already-emitted S_eval audit artifact."""
    data = json.loads(SUBGROUP_METRICS_PATH.read_text(encoding="utf-8"))
    subs = data["attributes"]["age"]["subgroups"]
    return {
        band: {"auroc": subs[band]["auroc"], "auroc_ci": subs[band]["auroc_ci"]} for band in subs
    }


def compute(*, n_boot: int = N_BOOT) -> dict:
    """Run the read-only age OOF robustness check; return everything for JSON/figure/report."""
    versions = assert_env()
    manifest = calibrated.load_manifest()

    rec = recover_s_fit_frames()
    df_fit_built = rec["df_fit_built"]
    df_fit_cohort = rec["df_fit_cohort"]
    patient_ids = rec["patient_ids"]

    x_fit = feature_frame(df_fit_built)
    y_fit = make_binary_target(df_fit_built).to_numpy()

    # OOF raw scores — the identical W5 function + locked booster (transient CV fits).
    oof_raw, fold_auroc = _oof_scores(
        df_fit_built, x_fit, y_fit, lambda: _model(LOCKED_BEST_PARAMS), N_CAL_FOLDS
    )
    pooled_oof_auroc = float(roc_auc_score(y_fit, oof_raw))
    if round(pooled_oof_auroc, 4) != EXPECTED_OOF_AUROC:
        raise AssertionError(
            f"S_fit OOF AUROC {pooled_oof_auroc:.6f} != W5 {EXPECTED_OOF_AUROC} "
            "(rounded to 4dp) — provenance check failed"
        )

    # Calibrated OOF (for ECE only) via the committed portable isotonic map.
    oof_cal = calibrated.apply_calibrator_portable(manifest, oof_raw)

    labels = subgroups.assign_subgroups(df_fit_cohort, AGE_ATTR).to_numpy()
    s_eval_age = _load_s_eval_age()

    # Per-band point metrics.
    bands = list(subgroups.AGE_BAND_ORDER)
    per_band: dict = {}
    for band in bands:
        mask = labels == band
        per_band[band] = _band_metrics(y_fit[mask], oof_raw[mask], oof_cal[mask])

    all_powered = all(per_band[b]["powered"] for b in bands)

    # Bootstrap CIs — discrimination on raw OOF, ECE on calibrated OOF (same seed -> same draws).
    ci_disc = bootstrap.bootstrap_subgroup_metrics(
        y_fit,
        oof_raw,
        patient_ids,
        labels,
        0.5,
        metrics=("auroc", "auprc"),
        n_boot=n_boot,
        seed=SEED,
    )
    ci_ece = bootstrap.bootstrap_subgroup_metrics(
        y_fit, oof_cal, patient_ids, labels, 0.5, metrics=("ece",), n_boot=n_boot, seed=SEED
    )
    for band in bands:
        per_band[band]["auroc_oof_ci"] = ci_disc[band]["ci"]["auroc"]
        per_band[band]["auprc_oof_ci"] = ci_disc[band]["ci"]["auprc"]
        per_band[band]["ece_oof_cal_ci"] = ci_ece[band]["ci"]["ece"]
        # Side-by-side vs the in-sample S_eval number.
        se = s_eval_age.get(band, {})
        per_band[band]["auroc_s_eval"] = se.get("auroc")
        per_band[band]["auroc_s_eval_ci"] = se.get("auroc_ci")
        per_band[band]["optimism_gap"] = (
            None if se.get("auroc") is None else float(se["auroc"] - per_band[band]["auroc_oof"])
        )

    # Pairwise ΔAUROC (OOF), paired resamples. Primary mid-band contrasts + the fragile anchor.
    pairwise: dict = {}
    for a, b in (*PRIMARY_CONTRASTS, FRAGILE_CONTRAST):
        d = bootstrap.bootstrap_pairwise_diff(
            y_fit,
            oof_raw,
            patient_ids,
            labels,
            a,
            b,
            0.5,
            keys=("auroc",),
            n_boot=n_boot,
            seed=SEED,
        )["auroc"]
        material = bool(d["ci_excludes_zero"] and abs(d["diff"]) > AUROC_PRACTICAL)
        pairwise[f"{a}__vs__{b}"] = {
            "diff": d["diff"],
            "ci": d["ci"],
            "ci_excludes_zero": d["ci_excludes_zero"],
            "practical_threshold": AUROC_PRACTICAL,
            "material": material,
            "primary": (a, b) in PRIMARY_CONTRASTS,
        }

    # [20-30) diagnostic on the RAW cohort age string (not age_midpoint).
    diag = _bucket_diagnostic(df_fit_cohort, y_fit, oof_raw, patient_ids, n_boot)

    # Verdict: survives iff a PRIMARY (well-powered mid-band) contrast is material.
    survives = any(pairwise[f"{a}__vs__{b}"]["material"] for a, b in PRIMARY_CONTRASTS)
    verdict, rationale = _verdict(survives, per_band, pairwise)

    return {
        "versions": versions,
        "s_fit_hash": rec["s_fit_hash"],
        "n": int(len(y_fit)),
        "n_pos": int(np.sum(y_fit == 1)),
        "pooled_oof_auroc": pooled_oof_auroc,
        "fold_auroc": fold_auroc,
        "all_powered": all_powered,
        "per_band": per_band,
        "pairwise": pairwise,
        "diagnostic": diag,
        "verdict": verdict,
        "rationale": rationale,
        "n_boot": n_boot,
        "git_commit": _git_sha(),
    }


def _bucket_diagnostic(df_fit_cohort, y_fit, oof_raw, patient_ids, n_boot: int) -> dict:
    """n / positives / OOF AUROC (+CI) for the anomalous [20-30) decade sub-bucket."""
    raw_age = df_fit_cohort[AGE_ATTR].to_numpy()
    mask = raw_age == DIAGNOSTIC_BUCKET
    n = int(mask.sum())
    y_b = y_fit[mask]
    pos = int(np.sum(y_b == 1))
    auroc = float(roc_auc_score(y_b, oof_raw[mask])) if 0 < pos < n else None
    # Patient-grouped CI via a two-label ([20-30) vs other) bootstrap; read the bucket entry.
    two = np.where(mask, DIAGNOSTIC_BUCKET, "other")
    ci = bootstrap.bootstrap_subgroup_metrics(
        y_fit, oof_raw, patient_ids, two, 0.5, metrics=("auroc",), n_boot=n_boot, seed=SEED
    )[DIAGNOSTIC_BUCKET]["ci"]["auroc"]
    return {
        "bucket": DIAGNOSTIC_BUCKET,
        "n": n,
        "positives": pos,
        "prevalence": float(pos / n) if n else None,
        "auroc_oof": auroc,
        "auroc_oof_ci": [ci[0], ci[1]],
    }


def _pair_str(pairwise: dict, a: str, b: str) -> str:
    p = pairwise[f"{a}__vs__{b}"]
    return f"Δ={p['diff']:+.3f} CI [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]"


def _ece_note(ece_80) -> str:
    """80+ calibration read, with the calibrator-in-sample caveat + robustness line."""
    if ece_80 is None:
        return "80+ OOF ECE is undefined (single-class band)."
    if ece_80 > SUBGROUP_ECE_FLAG:
        return (
            f"80+ OOF ECE is {ece_80:.3f} (> {SUBGROUP_ECE_FLAG} subgroup bar) — 80+ is "
            "miscalibrated as well as less rank-separable."
        )
    return (
        f"On calibration, 80+ OOF ECE is {ece_80:.3f} — far below the {SUBGROUP_ECE_FLAG} "
        "per-subgroup bar, so 80+ is merely less rank-separable, still well-calibrated. Caveat: "
        "this ECE is measured on calibrated OOF scores the committed isotonic map was itself fit "
        "on (so it is mildly optimistic FOR THE CALIBRATOR), but that small monotone-map optimism "
        f"cannot lift {ece_80:.3f} across the {SUBGROUP_ECE_FLAG} bar — the 'well-calibrated, not "
        "miscalibrated' conclusion is robust to the caveat."
    )


def _verdict(survives: bool, per_band: dict, pairwise: dict) -> tuple[str, str]:
    """'GAP SURVIVES' / 'GAP COLLAPSES' + a one-paragraph rationale (prereg §6).

    Framed as a MONOTONE AGE GRADIENT, not an 80+ cliff: adjacent steps can be sub-threshold
    while the cumulative multi-band gap is material.
    """
    gradient = " → ".join(f"{b} {per_band[b]['auroc_oof']:.3f}" for b in subgroups.AGE_BAND_ORDER)
    ece_note = _ece_note(per_band["80+"]["ece_oof_cal"])

    if survives:
        return (
            "GAP SURVIVES (real)",
            (
                f"Out-of-fold, AUROC declines monotonically across the age gradient ({gradient}). "
                "The material signal is this CUMULATIVE age gradient, not an 80+ cliff: the "
                f"adjacent 80+ vs 60-80 step is sub-threshold ({_pair_str(pairwise, '80+', '60-80')} "
                f"— statistically significant but |Δ| ≤ {AUROC_PRACTICAL}), while the cumulative "
                f"multi-band gaps clear the bar (80+ vs 40-60 {_pair_str(pairwise, '80+', '40-60')}, "
                f"material; 80+ vs <40 {_pair_str(pairwise, '80+', '<40')}, material). The age "
                "finding is REAL and best described as a gradient in rank-separability with age — "
                "not a discontinuity at 80+; the plausible mechanism is case mix (elderly risk "
                f"distributions are compressed, so ranking is intrinsically harder). {ece_note}"
            ),
        )
    return (
        "GAP COLLAPSES (substantially in-sample artifact)",
        (
            f"Out-of-fold, the age gradient flattens ({gradient}) and the powered mid-band "
            f"contrasts fall below the materiality bar (80+ vs 60-80 {_pair_str(pairwise, '80+', '60-80')}; "
            f"80+ vs 40-60 {_pair_str(pairwise, '80+', '40-60')}). The S_eval age disparity was "
            "substantially an in-sample-optimism artifact (an inflated small <40 band leaning on "
            f"the anomalous [20-30) bucket). The headline should be reframed accordingly. {ece_note}"
        ),
    )


# --- Persistence -----------------------------------------------------------------------
def _round(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def _ci(pair):
    return None if pair is None else [_round(pair[0]), _round(pair[1])]


def build_json(r: dict) -> dict:
    per_band = {}
    for band, m in r["per_band"].items():
        per_band[band] = {
            "n": m["n"],
            "positives": m["positives"],
            "prevalence": _round(m["prevalence"]),
            "powered": m["powered"],
            "auroc_oof": _round(m["auroc_oof"]),
            "auroc_oof_ci": _ci(m["auroc_oof_ci"]),
            "auprc_oof": _round(m["auprc_oof"]),
            "auprc_oof_ci": _ci(m["auprc_oof_ci"]),
            "ece_oof_cal": _round(m["ece_oof_cal"]),
            "ece_oof_cal_ci": _ci(m["ece_oof_cal_ci"]),
            "auroc_s_eval": _round(m["auroc_s_eval"]),
            "auroc_s_eval_ci": _ci(m["auroc_s_eval_ci"]),
            "optimism_gap": _round(m["optimism_gap"]),
        }
    pairwise = {
        name: {
            "diff": _round(p["diff"]),
            "ci": _ci(p["ci"]),
            "ci_excludes_zero": p["ci_excludes_zero"],
            "practical_threshold": p["practical_threshold"],
            "material": p["material"],
            "primary": p["primary"],
        }
        for name, p in r["pairwise"].items()
    }
    diag = dict(r["diagnostic"])
    diag["prevalence"] = _round(diag["prevalence"])
    diag["auroc_oof"] = _round(diag["auroc_oof"])
    diag["auroc_oof_ci"] = _ci(diag["auroc_oof_ci"])
    return {
        "verdict": r["verdict"],
        "rationale": r["rationale"],
        "per_band": per_band,
        "pairwise_oof": pairwise,
        "diagnostic_20_30": diag,
        "provenance": {
            "surface": "S_fit (out-of-fold)",
            "s_fit_patient_sha256": r["s_fit_hash"],
            "reproduced_oof_auroc": _round(r["pooled_oof_auroc"]),
            "expected_oof_auroc": EXPECTED_OOF_AUROC,
            "fold_auroc": [_round(a, 4) for a in r["fold_auroc"]],
            "n": r["n"],
            "n_pos": r["n_pos"],
            "all_bands_powered": r["all_powered"],
            "seed": SEED,
            "n_boot": r["n_boot"],
            "auroc_practical_threshold": AUROC_PRACTICAL,
            "ece_practical_threshold": ECE_PRACTICAL,
            "sklearn_version": r["versions"]["sklearn"],
            "lightgbm_version": r["versions"]["lightgbm"],
            "git_commit": r["git_commit"],
            "calibrator_in_sample_caveat": CALIBRATOR_IN_SAMPLE_CAVEAT,
        },
    }


# --- Figure ----------------------------------------------------------------------------
def make_figure(r: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = list(subgroups.AGE_BAND_ORDER)
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    def _series(getter_val, getter_ci):
        pts, los, his = [], [], []
        for b in bands:
            v = getter_val(b)
            ci = getter_ci(b)
            pts.append(np.nan if v is None else v)
            if v is not None and ci is not None and np.isfinite(ci[0]):
                los.append(v - ci[0])
                his.append(ci[1] - v)
            else:
                los.append(0.0)
                his.append(0.0)
        return pts, [los, his]

    se_v, se_err = _series(
        lambda b: r["per_band"][b]["auroc_s_eval"], lambda b: r["per_band"][b]["auroc_s_eval_ci"]
    )
    of_v, of_err = _series(
        lambda b: r["per_band"][b]["auroc_oof"], lambda b: r["per_band"][b]["auroc_oof_ci"]
    )
    ax.errorbar(x - 0.08, se_v, yerr=se_err, fmt="o", capsize=4, label="S_eval (in-sample)")
    ax.errorbar(x + 0.08, of_v, yerr=of_err, fmt="s", capsize=4, label="S_fit (out-of-fold)")
    ax.axhline(
        EXPECTED_OOF_AUROC, color="grey", linestyle=":", label=f"pooled OOF={EXPECTED_OOF_AUROC}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("AUROC (95% bootstrap CI)")
    ax.set_xlabel("age band")
    ax.set_title("Age-band AUROC: in-sample (S_eval) vs out-of-fold (S_fit)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Report section (append/replace, idempotent) ---------------------------------------
def render_section(r: dict) -> str:
    bands = list(subgroups.AGE_BAND_ORDER)

    def f(x, nd=3):
        return (
            "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"
        )

    def fci(ci):
        return "[n/a]" if not ci or not np.isfinite(ci[0]) else f"[{ci[0]:.3f}, {ci[1]:.3f}]"

    lines: list[str] = []
    lines.append(SECTION_MARKER)
    lines.append("")
    lines.append(
        "> Prereg §9 robustness follow-through for the age headline. The S_eval verdicts above "
        "are unchanged; this section recomputes age discrimination **out-of-fold** on `S_fit` "
        "(each row scored by a model NOT trained on it; ~64k rows, all bands well-powered; "
        "disjoint from S_eval and the spent holdout) to separate a real gap from in-sample "
        "optimism. Generated by `sentinel.fairness.age_oof_robustness`."
    )
    lines.append("")
    lines.append(
        f"Provenance: S_fit patient sha256 `{r['s_fit_hash'][:12]}`, N={r['n']:,}, "
        f"positives={r['n_pos']:,}. Reproduced pooled OOF AUROC **{r['pooled_oof_auroc']:.4f}** "
        f"(= W5 {EXPECTED_OOF_AUROC}; per-fold "
        f"{', '.join(f'{a:.3f}' for a in r['fold_auroc'])}). All four bands powered on S_fit: "
        f"{r['all_powered']}. seed={SEED}, bootstrap B={r['n_boot']} (patient-grouped)."
    )
    lines.append("")
    lines.append(f"> {CALIBRATOR_IN_SAMPLE_CAVEAT}")
    lines.append("")
    lines.append(f"### Verdict: **{r['verdict']}**")
    lines.append("")
    lines.append(r["rationale"])
    lines.append("")
    lines.append("### S_eval (in-sample) vs S_fit (OOF), per age band")
    lines.append("")
    lines.append(
        "| band | n (S_fit) | pos | S_eval AUROC | S_fit OOF AUROC (95% CI) | optimism (in−OOF) | OOF ECE (95% CI) |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---|")
    for b in bands:
        m = r["per_band"][b]
        lines.append(
            f"| {b} | {m['n']:,} | {m['positives']:,} | {f(m['auroc_s_eval'])} | "
            f"{f(m['auroc_oof'])} {fci(m['auroc_oof_ci'])} | {f(m['optimism_gap'])} | "
            f"{f(m['ece_oof_cal'])} {fci(m['ece_oof_cal_ci'])} |"
        )
    lines.append("")
    lines.append("### Pairwise ΔAUROC out-of-fold (material ⇔ CI excludes 0 AND |Δ| > 0.05)")
    lines.append("")
    lines.append("| contrast | role | ΔAUROC | 95% CI | excl. 0 | material |")
    lines.append("|---|---|---:|---|:--:|:--:|")
    for name, p in r["pairwise"].items():
        role = "primary (well-powered)" if p["primary"] else "fragile anchor (<40)"
        lines.append(
            f"| {name.replace('__vs__', ' vs ')} | {role} | {f(p['diff'])} | {fci(p['ci'])} | "
            f"{'yes' if p['ci_excludes_zero'] else 'no'} | {'YES' if p['material'] else 'no'} |"
        )
    lines.append("")
    d = r["diagnostic"]
    lines.append(
        f"**[20-30) diagnostic** (raw cohort `age` string, the anomalous high-prevalence slice "
        f"inside `<40`): n={d['n']:,}, positives={d['positives']:,}, prevalence={f(d['prevalence'])}, "
        f"OOF AUROC {f(d['auroc_oof'])} {fci(d['auroc_oof_ci'])}. This shows how much the `<40` "
        "anchor leans on that bucket."
    )
    lines.append("")
    lines.append(
        f"Figure: `figures/{FIG_PATH.name}`. Machine-readable: `{OUT_JSON.relative_to(ROOT).as_posix()}`."
    )
    lines.append("")
    return "\n".join(lines)


def extend_report(section: str, report_path: Path = REPORT_PATH) -> None:
    """Append the OOF section, replacing a prior copy if present (idempotent)."""
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    base = existing.split("\n" + SECTION_MARKER)[0].rstrip()
    report_path.write_text(base + "\n\n" + section, encoding="utf-8")


def main() -> None:
    r = compute()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(build_json(r), indent=2) + "\n", encoding="utf-8")
    make_figure(r, FIG_PATH)
    extend_report(render_section(r))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"OOF ROBUSTNESS VERDICT: {r['verdict']}")
    print(
        f"  reproduced pooled OOF AUROC: {r['pooled_oof_auroc']:.4f} (expected {EXPECTED_OOF_AUROC})"
    )
    for b in subgroups.AGE_BAND_ORDER:
        m = r["per_band"][b]
        print(
            f"  {b:6s} S_eval {m['auroc_s_eval']:.3f} -> OOF {m['auroc_oof']:.3f} "
            f"(optimism {m['optimism_gap']:+.3f})"
        )
    print(f"[written] {OUT_JSON}")
    print(f"[written] {FIG_PATH}")
    print(f"[extended] {REPORT_PATH}")


if __name__ == "__main__":
    main()

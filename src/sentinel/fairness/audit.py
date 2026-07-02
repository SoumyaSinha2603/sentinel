"""Phase-3 fairness audit — entrypoint (READ-ONLY).

Slices the shipped calibrated-probability path (@phase1 booster + committed isotonic
manifest) across pre-registered sensitive subgroups on the FROZEN `S_eval` surface, and
emits an honest audit. Trains nothing, fits nothing, registers nothing; the Phase-1 holdout
is never loaded for scoring — only its patients are identified to ASSERT exclusion.

Sensitive attributes (`race`/`gender`/`age`/`payer_code`) are read from the raw cohort
columns ONLY as slicing labels; they are never added to any feature frame. Scoring uses the
built feature frame; slicing uses the row-aligned cohort frame (build_features preserves row
order, build_cohort resets the index), recovered under the same S_eval hash guard Phase 2 uses.

Run:

    python -m sentinel.fairness.audit
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn

from sentinel.calibration.calibration_splits import make_calibration_splits
from sentinel.clinical_utility import calibrated
from sentinel.config import MODELS_DIR, REPORTS_DIR, ROOT
from sentinel.data.cohort import build_cohort
from sentinel.data.load import fetch_raw
from sentinel.evaluation.splits import GROUP_COL, make_binary_target, make_holdout_split
from sentinel.fairness import bootstrap, metrics, subgroups
from sentinel.features.build import build_features

# --- Pre-registered constants (echoed from reports/fairness_prereg.md — no magic numbers) --
REQUIRED_SKLEARN = "1.7.2"
SEED = 42
N_BOOT = 2000
S_EVAL_EXPECTED_HASH = "71a7a8b69e3bbfd3ddd41b0dd833490948037b34cff78b562a19e0ce5a68e148"

PRIMARY_BUDGET = 0.10  # prereg §4: 10% capacity is the headline parity point.

# Practical-significance thresholds (prereg §6). A pairwise difference is MATERIAL only if the
# bootstrap CI excludes 0 AND |diff| exceeds this threshold for that metric.
PRACTICAL_THRESHOLDS = {"auroc": 0.05, "tpr": 0.10, "fpr": 0.10, "ece": 0.03}
SUBGROUP_ECE_FLAG = 0.05  # per-subgroup calibration flag (prereg §6).
PAIR_KEYS = ("auroc", "tpr", "fpr", "ece")

ATTRIBUTES = ("race", "gender", "age", "payer_code")

FAIRNESS_DIR = MODELS_DIR / "fairness"
FIG_DIR = REPORTS_DIR / "figures"
SUBGROUP_METRICS_PATH = FAIRNESS_DIR / "subgroup_metrics.json"
PAIRWISE_PATH = FAIRNESS_DIR / "pairwise_disparities.json"
REPORT_PATH = REPORTS_DIR / "fairness_audit.md"

SENSITIVE_USE_NOTE = (
    "Sensitive attributes (race/gender/age/payer_code) were used ONLY as slicing labels for "
    "subgroup metrics, read from the raw cohort columns. They are never added to any feature "
    "frame and the deployed model does not consume them as decision inputs."
)

# Honesty caveat (prereg §9): absolute subgroup AUROCs land above the ~0.62-0.68 band because
# @phase1 is IN-SAMPLE on S_eval. This is a property of the frozen surface, not audit leakage,
# and the deliverable is the relative parity BETWEEN subgroups, not the absolute levels.
IN_SAMPLE_NOTE = (
    "In-sample optimism — read absolute levels with care. `@phase1` was refit on the full 80% "
    "train, which CONTAINS `S_eval` (S_eval is a 20% patient-grouped slice carved from inside "
    "that 80% for Phase 2). Scoring S_eval with @phase1 is therefore IN-SAMPLE, so the absolute "
    "subgroup AUROCs here (~0.70-0.84) are optimistic versus the honest locked holdout AUROC "
    "0.677 and the ~0.62-0.68 ceiling — they are NOT a discrimination win. This audit's "
    "deliverable is the RELATIVE parity between subgroups, where the optimism is largely "
    "common-mode; absolute discrimination is not comparable to the holdout figure. This is a "
    "known, documented property of the frozen S_eval surface (Phase-2 calibration used "
    "out-of-fold S_fit scores precisely to avoid this optimism), not audit leakage."
)


def assert_env() -> dict:
    """Pin sklearn (calibration knots were fit under it) and report versions."""
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


def recover_s_eval_frames(manifest: dict) -> dict:
    """Recover the frozen S_eval as (built feature frame, row-aligned raw cohort frame).

    Replicates the Phase-2 recovery (`evaluate._recover_s_eval`) and MUST assert the identical
    ``manifest["split_hashes"]["S_eval"]`` and the ``S_eval ∩ holdout == ∅`` check. The cohort
    frame is carried through the exact same positional index chain so the raw slicing labels
    (raw decade age buckets, raw NaN race, raw "Unknown/Invalid" gender) align row-for-row with
    the scored built frame — the built frame has already collapsed those into "Missing".
    """
    df_cohort = build_cohort(fetch_raw())  # reset_index(drop=True); raw sensitive columns
    df_built = build_features(df_cohort)  # row-aligned; scored frame

    train_idx, holdout_idx = make_holdout_split(df_built)
    holdout_patients = {str(p) for p in df_built[GROUP_COL].to_numpy()[holdout_idx]}

    df_train_built = df_built.iloc[train_idx].reset_index(drop=True)
    df_train_cohort = df_cohort.iloc[train_idx].reset_index(drop=True)

    splits = make_calibration_splits(df_train_built)
    s_eval_hash = splits["stats"]["S_eval"]["patient_sha256"]
    expected = manifest["split_hashes"]["S_eval"]
    if s_eval_hash != expected:
        raise AssertionError(
            f"S_eval patient hash drift: recomputed {s_eval_hash} != manifest {expected}"
        )
    if s_eval_hash != S_EVAL_EXPECTED_HASH:
        raise AssertionError(
            f"S_eval hash != pre-registered {S_EVAL_EXPECTED_HASH}: got {s_eval_hash}"
        )

    idx = splits["indices"]["S_eval"]
    df_eval_built = df_train_built.iloc[idx].reset_index(drop=True)
    df_eval_cohort = df_train_cohort.iloc[idx].reset_index(drop=True)

    overlap = splits["patients"]["S_eval"] & holdout_patients
    if overlap:
        raise AssertionError(f"S_eval ∩ holdout non-empty: {len(overlap)} patients")

    return {
        "df_eval_built": df_eval_built,
        "df_eval_cohort": df_eval_cohort,
        "s_eval_hash": s_eval_hash,
        "patient_ids": df_eval_built[GROUP_COL].to_numpy(),
    }


def _load_thresholds() -> dict:
    """Operating-point implied thresholds from the committed clinical-utility artifact."""
    ops = json.loads(
        (MODELS_DIR / "clinical" / "operating_points.json").read_text(encoding="utf-8")
    )
    return {
        round(float(p["budget"]), 2): float(p["implied_threshold"]) for p in ops["operating_points"]
    }


# --- Verdict logic (prereg §5-7) -------------------------------------------------------
def _classify_pair_metric(diff: dict, key: str) -> str:
    """'material' / 'flag' / 'pass' for one pairwise metric (prereg §5)."""
    thr = PRACTICAL_THRESHOLDS[key]
    excludes_zero = diff["ci_excludes_zero"]
    d = diff["diff"]
    magnitude = abs(d) if np.isfinite(d) else 0.0
    if excludes_zero and magnitude > thr:
        return "material"
    if excludes_zero:
        return "flag"  # statistically significant but sub-threshold
    return "pass"


def _attribute_verdict(pair_results: dict, powered_ece_flag: bool) -> dict:
    """Roll powered-pair classifications up to a per-metric-family + attribute verdict."""
    per_metric: dict = {}
    for key in PAIR_KEYS:
        classes = [_classify_pair_metric(pairs[key], key) for pairs in pair_results.values()]
        if "material" in classes:
            per_metric[key] = "MATERIAL DISPARITY"
        elif "flag" in classes:
            per_metric[key] = "FLAG (monitor)"
        else:
            per_metric[key] = "PASS"
    if powered_ece_flag and per_metric.get("ece") == "PASS":
        per_metric["ece"] = "FLAG (monitor)"

    overall = max(per_metric.values(), key=lambda v: _VERDICT_ORDER[v]) if per_metric else "PASS"
    return {"per_metric": per_metric, "overall": overall}


_VERDICT_ORDER = {"PASS": 0, "FLAG (monitor)": 1, "MATERIAL DISPARITY": 2}


def project_verdict(per_attribute: dict) -> str:
    """Worst overall verdict among the PRIMARY attributes only.

    `payer_code` (exploratory) is excluded, so it cannot alone produce a project-level
    MATERIAL verdict (prereg §2.4).
    """
    primary = [
        ares["verdict"]["overall"]
        for attr, ares in per_attribute.items()
        if ares["tier"] == "primary"
    ]
    return max(primary, key=lambda v: _VERDICT_ORDER[v]) if primary else "PASS"


# --- Core computation ------------------------------------------------------------------
def _attribute_result(attr: str, rec: dict, y, p_cal, thresholds: dict, n_boot: int) -> dict:
    """Full per-attribute audit: subgroup metrics + CIs + powered pairwise diffs + verdict."""
    primary_t = thresholds[PRIMARY_BUDGET]
    labels_all = subgroups.assign_subgroups(rec["df_eval_cohort"], attr).to_numpy()

    # Drop the pre-registered gender "Unknown/Invalid" bucket (n=1), logged.
    keep = labels_all != subgroups.GENDER_DROP_LABEL
    n_dropped = int((~keep).sum())
    labels = labels_all[keep]
    y_a = np.asarray(y)[keep]
    p_a = np.asarray(p_cal)[keep]
    pid_a = rec["patient_ids"][keep]

    present = list(dict.fromkeys(labels.tolist()))
    subgroup_out: dict = {}
    powered_labels: list[str] = []
    for lab in present:
        mask = labels == lab
        m = metrics.subgroup_metrics(y_a[mask], p_a[mask], primary_t)
        m["powered"] = metrics.is_powered(m)
        # TPR/FPR sensitivity at the 5% / 20% operating points.
        m["sensitivity_points"] = {
            f"{int(b * 100)}pct": metrics.confusion_at_threshold(y_a[mask], p_a[mask], t)
            for b, t in thresholds.items()
            if b != PRIMARY_BUDGET
        }
        subgroup_out[lab] = m
        if m["powered"]:
            powered_labels.append(lab)

    # Bootstrap CIs for EVERY subgroup (underpowered ones stay descriptive, with wide CIs).
    ci = bootstrap.bootstrap_subgroup_metrics(
        y_a, p_a, pid_a, labels, primary_t, n_boot=n_boot, seed=SEED
    )
    for lab in present:
        subgroup_out[lab]["ci"] = ci[lab]["ci"]
        subgroup_out[lab]["ci_n_eff"] = ci[lab]["n_eff"]

    # Powered pairwise differences only (prereg §3: descriptive-only groups excluded from verdicts).
    pair_results: dict = {}
    for a, b in itertools.combinations(powered_labels, 2):
        pair = bootstrap.bootstrap_pairwise_diff(
            y_a, p_a, pid_a, labels, a, b, primary_t, keys=PAIR_KEYS, n_boot=n_boot, seed=SEED
        )
        for key in PAIR_KEYS:
            pair[key]["practical_threshold"] = PRACTICAL_THRESHOLDS[key]
            pair[key]["verdict"] = _classify_pair_metric(pair[key], key)
        for key in ("tpr", "fpr", "selection_rate"):
            pair[f"four_fifths_{key}"] = metrics.four_fifths_ratio(
                subgroup_out[a], subgroup_out[b], key
            )
        pair_results[f"{a}__vs__{b}"] = pair

    powered_ece_flag = any(
        (subgroup_out[lab]["ece"] is not None and subgroup_out[lab]["ece"] > SUBGROUP_ECE_FLAG)
        for lab in powered_labels
    )
    verdict = _attribute_verdict(pair_results, powered_ece_flag)

    return {
        "tier": subgroups.ATTRIBUTE_TIER[attr],
        "n_dropped_unknown": n_dropped,
        "primary_threshold": primary_t,
        "subgroups": subgroup_out,
        "powered_labels": powered_labels,
        "pairwise": pair_results,
        "powered_ece_flag": powered_ece_flag,
        "verdict": verdict,
    }


def compute(*, n_boot: int = N_BOOT) -> dict:
    """Run the full read-only fairness audit and return everything for JSON/figures/report."""
    versions = assert_env()
    manifest = calibrated.load_manifest()
    thresholds = _load_thresholds()

    rec = recover_s_eval_frames(manifest)
    df_eval_built = rec["df_eval_built"]
    y = make_binary_target(df_eval_built).to_numpy()

    booster = calibrated.load_phase1_booster()
    p_cal = calibrated.get_calibrated_proba(df_eval_built, booster=booster, manifest=manifest)

    n = int(len(y))
    n_pos = int(np.sum(y == 1))

    per_attribute = {
        attr: _attribute_result(attr, rec, y, p_cal, thresholds, n_boot) for attr in ATTRIBUTES
    }

    # Project-level verdict: worst of the PRIMARY attributes only. payer_code is exploratory
    # and cannot, alone, produce a project-level MATERIAL verdict (prereg §2.4).
    proj_verdict = project_verdict(per_attribute)

    return {
        "versions": versions,
        "thresholds": thresholds,
        "s_eval_hash": rec["s_eval_hash"],
        "n": n,
        "n_pos": n_pos,
        "prevalence": n_pos / n,
        "p_cal": p_cal,
        "y": y,
        "rec": rec,
        "per_attribute": per_attribute,
        "project_verdict": proj_verdict,
        "n_boot": n_boot,
        "git_commit": _git_sha(),
    }


# --- Persistence -----------------------------------------------------------------------
def _round(x, nd=6):
    if x is None:
        return None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return round(float(x), nd)


def _subgroup_json(r: dict) -> dict:
    provenance = {
        "s_eval_patient_sha256": r["s_eval_hash"],
        "seed": SEED,
        "n_boot": r["n_boot"],
        "sklearn_version": r["versions"]["sklearn"],
        "lightgbm_version": r["versions"]["lightgbm"],
        "git_commit": r["git_commit"],
        "primary_budget": PRIMARY_BUDGET,
        "primary_threshold": r["thresholds"][PRIMARY_BUDGET],
        "operating_point_thresholds": {str(k): v for k, v in r["thresholds"].items()},
        "practical_thresholds": PRACTICAL_THRESHOLDS,
        "subgroup_ece_flag": SUBGROUP_ECE_FLAG,
        "sensitive_attribute_use_note": SENSITIVE_USE_NOTE,
        "in_sample_optimism_note": IN_SAMPLE_NOTE,
        "n": r["n"],
        "n_pos": r["n_pos"],
        "prevalence": r["prevalence"],
    }
    metric_keys = ("auroc", "auprc", "ece", "brier", "tpr", "fpr", "selection_rate")
    attrs: dict = {}
    for attr, ares in r["per_attribute"].items():
        subs: dict = {}
        for lab, m in ares["subgroups"].items():
            row = {
                "n": m["n"],
                "positives": m["positives"],
                "prevalence": _round(m["prevalence"]),
                "powered": m["powered"],
                "threshold": m["threshold"],
            }
            for k in metric_keys:
                row[k] = _round(m.get(k))
                ci = m["ci"].get(k)
                row[f"{k}_ci"] = [_round(ci[0]), _round(ci[1])] if ci else None
            row["sensitivity_points"] = {
                sp: {
                    "tpr": _round(v["tpr"]),
                    "fpr": _round(v["fpr"]),
                    "selection_rate": _round(v["selection_rate"]),
                }
                for sp, v in m["sensitivity_points"].items()
            }
            subs[lab] = row
        attrs[attr] = {
            "tier": ares["tier"],
            "n_dropped_unknown": ares["n_dropped_unknown"],
            "powered_labels": ares["powered_labels"],
            "subgroups": subs,
        }
    return {"provenance": provenance, "attributes": attrs}


def _pairwise_json(r: dict) -> dict:
    attrs: dict = {}
    for attr, ares in r["per_attribute"].items():
        pairs: dict = {}
        for pair_name, pair in ares["pairwise"].items():
            entry: dict = {}
            for key in PAIR_KEYS:
                d = pair[key]
                entry[key] = {
                    "diff": _round(d["diff"]),
                    "ci": [_round(d["ci"][0]), _round(d["ci"][1])],
                    "ci_excludes_zero": d["ci_excludes_zero"],
                    "practical_threshold": d["practical_threshold"],
                    "n_eff": d["n_eff"],
                    "verdict": d["verdict"],
                }
            for key in ("tpr", "fpr", "selection_rate"):
                entry[f"four_fifths_{key}"] = _round(pair[f"four_fifths_{key}"], 4)
            pairs[pair_name] = entry
        attrs[attr] = {
            "tier": ares["tier"],
            "pairs": pairs,
            "verdict": ares["verdict"],
        }
    return {
        "attributes": attrs,
        "project_verdict": r["project_verdict"],
        "project_verdict_note": (
            "Project verdict is the worst of the PRIMARY attributes (race/gender/age). "
            "payer_code is exploratory and cannot alone produce a project MATERIAL verdict."
        ),
        "s_eval_patient_sha256": r["s_eval_hash"],
        "seed": SEED,
        "n_boot": r["n_boot"],
        "git_commit": r["git_commit"],
    }


# --- Figures ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_metric_comparison(attr: str, ares: dict, path: Path) -> None:
    plt = _plt()
    labels = ares["powered_labels"]
    if not labels:
        return
    keys = ("auroc", "tpr", "fpr")
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels) + 3), 5))
    width = 0.25
    x = np.arange(len(labels))
    for j, key in enumerate(keys):
        pts, los, his = [], [], []
        for lab in labels:
            m = ares["subgroups"][lab]
            v = m.get(key)
            ci = m["ci"].get(key)
            pts.append(np.nan if v is None else v)
            if ci and np.isfinite(ci[0]) and v is not None:
                los.append(v - ci[0])
                his.append(ci[1] - v)
            else:
                los.append(0.0)
                his.append(0.0)
        ax.bar(x + (j - 1) * width, pts, width, yerr=[los, his], capsize=3, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("metric value (95% bootstrap CI)")
    ax.set_title(
        f"Subgroup metrics — {attr} (powered groups, threshold={ares['primary_threshold']:.4f})"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_reliability_overlay(attr: str, ares: dict, path: Path) -> None:
    plt = _plt()
    labels = ares["powered_labels"]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    for lab in labels:
        rel = ares["subgroups"][lab]["reliability"]
        ax.plot([b["mean_pred"] for b in rel], [b["obs_freq"] for b in rel], marker="o", label=lab)
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency (bin)")
    ax.set_title(f"Reliability by subgroup — {attr}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _make_figures(r: dict) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for attr in ("race", "gender", "age"):  # primary attributes
        ares = r["per_attribute"][attr]
        mc = FIG_DIR / f"fairness_{attr}_metrics.png"
        rel = FIG_DIR / f"fairness_{attr}_reliability.png"
        _plot_metric_comparison(attr, ares, mc)
        _plot_reliability_overlay(attr, ares, rel)
        written += [mc.name, rel.name]
    return written


# --- Report ----------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _fmt_ci(ci):
    if not ci or not np.isfinite(ci[0]):
        return "[n/a]"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _render_report(r: dict, figures: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Fairness audit — Phase 3 (subgroup metrics, equalized-odds, calibration)")
    lines.append("")
    lines.append(
        "> Read-only audit of the calibrated production path (@phase1 booster + committed "
        "isotonic manifest) across the pre-registered sensitive subgroups on the FROZEN "
        "`S_eval` surface. Method and every threshold are fixed in `reports/fairness_prereg.md` "
        "(the source of truth); this report presents results. No model trained/registered; the "
        "Phase-1 holdout was never loaded. Generated by `sentinel.fairness.audit`."
    )
    lines.append("")
    lines.append(
        f"Versions: sklearn `{r['versions']['sklearn']}`, lightgbm `{r['versions']['lightgbm']}`. "
        f"seed=42, bootstrap B={r['n_boot']} (patient-grouped). git `{r['git_commit'][:12]}`. "
        f"S_eval patient sha256 `{r['s_eval_hash'][:12]}` (matches manifest)."
    )
    lines.append("")
    lines.append(
        f"S_eval: N={r['n']:,}, positives={r['n_pos']:,}, prevalence={r['prevalence']:.4f}. "
        f"Primary operating point = {int(PRIMARY_BUDGET * 100)}% capacity "
        f"(implied threshold {r['thresholds'][PRIMARY_BUDGET]:.4f}); 5%/20% reported for sensitivity."
    )
    lines.append("")
    lines.append(f"> {SENSITIVE_USE_NOTE}")
    lines.append("")
    lines.append(f"> **{IN_SAMPLE_NOTE}**")
    lines.append("")
    lines.append(f"## Project-level verdict: **{r['project_verdict']}**")
    lines.append("")
    lines.append(
        "Worst of the primary attributes (race/gender/age). A MATERIAL verdict does not fail the "
        "project — it triggers honest documentation and a recorded decision on whether a separately "
        "pre-registered mitigation phase is warranted (prereg §7). `payer_code` is exploratory and "
        "cannot alone drive this verdict."
    )
    lines.append("")

    for attr in ATTRIBUTES:
        ares = r["per_attribute"][attr]
        lines.append(f"## `{attr}` — {ares['tier']}  →  {ares['verdict']['overall']}")
        lines.append("")
        if ares["n_dropped_unknown"]:
            lines.append(
                f"_Dropped {ares['n_dropped_unknown']} row(s) labeled "
                f"`{subgroups.GENDER_DROP_LABEL}` (prereg §2.2)._"
            )
            lines.append("")
        pm = ares["verdict"]["per_metric"]
        lines.append(
            "Per-metric verdicts (powered pairs only): "
            + ", ".join(f"{k}={v}" for k, v in pm.items())
            + "."
        )
        lines.append("")
        lines.append(
            "| subgroup | n | pos | prev | powered | AUROC (95% CI) | ECE (95% CI) | TPR | FPR | sel. rate |"
        )
        lines.append("|---|---:|---:|---:|:--:|---|---|---:|---:|---:|")
        for lab, m in ares["subgroups"].items():
            ci = m["ci"]
            tag = "✓" if m["powered"] else "underpowered"
            lines.append(
                f"| {lab} | {m['n']:,} | {m['positives']:,} | {_fmt(m['prevalence'], 3)} | {tag} | "
                f"{_fmt(m['auroc'])} {_fmt_ci(ci.get('auroc'))} | "
                f"{_fmt(m['ece'])} {_fmt_ci(ci.get('ece'))} | "
                f"{_fmt(m['tpr'])} | {_fmt(m['fpr'])} | {_fmt(m['selection_rate'])} |"
            )
        lines.append("")
        if ares["pairwise"]:
            lines.append(
                "Pairwise differences at the primary operating point (material ⇔ CI excludes 0 AND |Δ| > threshold):"
            )
            lines.append("")
            lines.append("| pair | metric | Δ | 95% CI | excl. 0 | thr | 4/5 ratio | verdict |")
            lines.append("|---|---|---:|---|:--:|---:|---:|---|")
            for pair_name, pair in ares["pairwise"].items():
                for key in PAIR_KEYS:
                    d = pair[key]
                    ff = pair.get(f"four_fifths_{key}")
                    ff_s = _fmt(ff, 3) if ff is not None else "—"
                    lines.append(
                        f"| {pair_name.replace('__vs__', ' vs ')} | {key} | {_fmt(d['diff'])} | "
                        f"{_fmt_ci(d['ci'])} | {'yes' if d['ci_excludes_zero'] else 'no'} | "
                        f"{d['practical_threshold']:.2f} | {ff_s} | {d['verdict']} |"
                    )
            lines.append("")
        else:
            lines.append(
                "_No powered pair available for pairwise parity testing (see power column)._"
            )
            lines.append("")

    lines.append("## Findings (plain language)")
    lines.append("")
    lines.append(_findings_text(r))
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for f in figures:
        lines.append(f"- `figures/{f}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        f"- `{SUBGROUP_METRICS_PATH.relative_to(ROOT).as_posix()}` — per-subgroup metrics + CIs + provenance."
    )
    lines.append(
        f"- `{PAIRWISE_PATH.relative_to(ROOT).as_posix()}` — powered pairwise diffs, four-fifths ratios, verdicts."
    )
    lines.append("")
    return "\n".join(lines)


def _findings_text(r: dict) -> str:
    parts: list[str] = []
    parts.append(
        "Race parity is effectively a **two-group comparison (Caucasian vs AfricanAmerican)** — "
        "the other four race groups (Hispanic, Other, Missing, Asian) fall below the ≥100-row / "
        "≥30-positive power bar and are reported descriptively with wide CIs, never used for a "
        "pass/fail verdict. `payer_code` is a noisy, 39%-missing socioeconomic proxy and is "
        "exploratory only."
    )
    for attr in ATTRIBUTES:
        ares = r["per_attribute"][attr]
        materials = [
            (pair_name, key)
            for pair_name, pair in ares["pairwise"].items()
            for key in PAIR_KEYS
            if pair[key]["verdict"] == "material"
        ]
        if materials:
            desc = "; ".join(f"{p.replace('__vs__', ' vs ')} on {k}" for p, k in materials)
            parts.append(
                f"**`{attr}`**: material disparity found ({desc}). Plausible mechanisms to weigh "
                "before any mitigation: differential missingness, subgroup base-rate differences, "
                "and feature availability — not evidence of an intentionally biased input, since "
                "sensitive attributes are not model inputs."
            )
        else:
            parts.append(
                f"**`{attr}`**: no material disparity among powered groups (verdict "
                f"{ares['verdict']['overall']}). Any sub-threshold-but-significant gaps are flagged "
                "for monitoring, not called failures."
            )
    parts.append(
        "Cross-subgroup AUPRC differences are interpreted as base-rate artifacts (AUPRC scales with "
        "prevalence), not bias. Small-subgroup AUROC is high-variance; a suspicious gap is more "
        "likely a power/variance artifact than a real effect — read every point estimate with its CI."
    )
    return "\n\n".join(parts)


def main() -> None:
    r = compute()

    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    SUBGROUP_METRICS_PATH.write_text(
        json.dumps(_subgroup_json(r), indent=2) + "\n", encoding="utf-8"
    )
    PAIRWISE_PATH.write_text(json.dumps(_pairwise_json(r), indent=2) + "\n", encoding="utf-8")

    figures = _make_figures(r)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _render_report(r, figures)
    REPORT_PATH.write_text(report, encoding="utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"PROJECT VERDICT: {r['project_verdict']}")
    for attr in ATTRIBUTES:
        ares = r["per_attribute"][attr]
        print(f"  {attr:12s} [{ares['tier']:11s}] -> {ares['verdict']['overall']}")
    print(f"[written] {SUBGROUP_METRICS_PATH}")
    print(f"[written] {PAIRWISE_PATH}")
    print(f"[written] {REPORT_PATH}")
    for f in figures:
        print(f"[written] {FIG_DIR / f}")


if __name__ == "__main__":
    main()

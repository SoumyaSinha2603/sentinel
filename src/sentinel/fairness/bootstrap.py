"""Patient-grouped bootstrap CIs for subgroup and pairwise-difference metrics (prereg §5).

Mirrors the resampling machinery of `clinical_utility.ranking.bootstrap_pr_ci` — resample
**patients** (not rows), gather their rows — but recomputes *subgroup* and *pairwise-
difference* metrics inside each resample. Patient-grouped so repeat encounters do not inflate
the effective sample size, consistent with the project's grouped-split discipline.

Determinism: a single ``np.random.default_rng(seed)`` builds the full patient-index matrix
once; both entrypoints construct it identically from the same seed, so subgroup CIs and
pairwise differences are computed on the *same* resamples (required for the paired diff to be
valid). Degenerate draws (a resampled subgroup with <2 classes / 0 positives) record NaN for
the affected metric and are dropped before percentiles; the effective draw count is reported.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinel.evaluation import metrics as base_metrics
from sentinel.fairness.metrics import confusion_at_threshold

DEFAULT_METRICS = ("auroc", "auprc", "ece", "tpr", "fpr", "selection_rate")
DEFAULT_DIFF_KEYS = ("auroc", "tpr", "fpr", "ece")


def _metric_on_subset(y_sub: np.ndarray, p_sub: np.ndarray, threshold: float, key: str) -> float:
    """Single metric on a subgroup subset; ``nan`` where the metric is undefined for the draw."""
    n = len(y_sub)
    if n == 0:
        return float("nan")
    pos = int(np.sum(y_sub == 1))
    neg = n - pos

    if key in ("auroc", "auprc"):
        if pos == 0 or neg == 0:
            return float("nan")
        fn = roc_auc_score if key == "auroc" else average_precision_score
        return float(fn(y_sub, p_sub))
    if key == "ece":
        if pos == 0 or neg == 0:
            return float("nan")
        return float(base_metrics.calibration_metrics(y_sub, p_sub)["ece"])
    if key == "brier":
        return float(np.mean((p_sub - y_sub.astype(float)) ** 2))
    if key in ("tpr", "fpr", "selection_rate"):
        v = confusion_at_threshold(y_sub, p_sub, threshold)[key]
        return float("nan") if v is None else float(v)
    raise ValueError(f"unknown metric key: {key!r}")


def _patient_draws(patient_ids, n_boot: int, seed: int):
    """Return (per-patient row-index lists, choice matrix of shape (n_boot, n_patients)).

    Constructed deterministically from ``seed`` so independent callers get identical draws.
    """
    pid = np.asarray(patient_ids)
    uniq, inverse = np.unique(pid, return_inverse=True)
    rows = [np.where(inverse == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return rows, choice


def _percentile_ci(values: list[float]) -> tuple:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n_eff = int(arr.size)
    if n_eff == 0:
        return (float("nan"), float("nan")), 0
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))), n_eff


def bootstrap_subgroup_metrics(
    y,
    p_prob,
    patient_ids,
    subgroup_labels,
    threshold: float,
    *,
    metrics=DEFAULT_METRICS,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Per subgroup label, 95% percentile CIs for each metric.

    Returns ``{label: {"ci": {metric: (lo, hi)}, "n_eff": {metric: int}}}``. Resamples unique
    patients with replacement, concatenates their rows, then subsets to each subgroup and
    recomputes metrics (NaN-guarded, see module docstring).
    """
    y = np.asarray(y)
    p_prob = np.asarray(p_prob, dtype=float)
    labels = np.asarray(subgroup_labels)
    rows, choice = _patient_draws(patient_ids, n_boot, seed)

    unique_labels = list(dict.fromkeys(labels.tolist()))
    draws = {lab: {m: [] for m in metrics} for lab in unique_labels}

    for b in range(n_boot):
        boot_rows = np.concatenate([rows[i] for i in choice[b]])
        yb = y[boot_rows]
        pb = p_prob[boot_rows]
        lb = labels[boot_rows]
        for lab in unique_labels:
            mask = lb == lab
            ys = yb[mask]
            ps = pb[mask]
            for m in metrics:
                draws[lab][m].append(_metric_on_subset(ys, ps, threshold, m))

    out: dict = {}
    for lab in unique_labels:
        ci: dict = {}
        n_eff: dict = {}
        for m in metrics:
            ci[m], n_eff[m] = _percentile_ci(draws[lab][m])
        out[lab] = {"ci": ci, "n_eff": n_eff}
    return out


def bootstrap_pairwise_diff(
    y,
    p_prob,
    patient_ids,
    subgroup_labels,
    group_a: str,
    group_b: str,
    threshold: float,
    *,
    keys=DEFAULT_DIFF_KEYS,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """95% CI of ``metric_a - metric_b`` per key, PAIRED on the same resampled patients.

    Returns ``{key: {"diff": point, "ci": (lo, hi), "ci_excludes_zero": bool, "n_eff": int}}``.
    The point difference is computed on the full (un-resampled) surface; the CI comes from the
    paired bootstrap so it captures the correlation between the two subgroups.
    """
    y = np.asarray(y)
    p_prob = np.asarray(p_prob, dtype=float)
    labels = np.asarray(subgroup_labels)
    rows, choice = _patient_draws(patient_ids, n_boot, seed)

    diffs = {k: [] for k in keys}
    for b in range(n_boot):
        boot_rows = np.concatenate([rows[i] for i in choice[b]])
        yb = y[boot_rows]
        pb = p_prob[boot_rows]
        lb = labels[boot_rows]
        mask_a = lb == group_a
        mask_b = lb == group_b
        for k in keys:
            va = _metric_on_subset(yb[mask_a], pb[mask_a], threshold, k)
            vb = _metric_on_subset(yb[mask_b], pb[mask_b], threshold, k)
            diffs[k].append(va - vb)

    full_a = labels == group_a
    full_b = labels == group_b
    out: dict = {}
    for k in keys:
        va = _metric_on_subset(y[full_a], p_prob[full_a], threshold, k)
        vb = _metric_on_subset(y[full_b], p_prob[full_b], threshold, k)
        point = va - vb
        (lo, hi), n_eff = _percentile_ci(diffs[k])
        excludes_zero = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
        out[k] = {
            "diff": float(point),
            "ci": (lo, hi),
            "ci_excludes_zero": excludes_zero,
            "n_eff": n_eff,
        }
    return out

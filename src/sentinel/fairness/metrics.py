"""Per-subgroup fairness metrics: threshold confusion, subgroup summary, pairwise diffs.

Reuses `sentinel.evaluation.metrics` for discrimination + calibration and adds the
threshold-based error-rate components (TPR/FPR/selection_rate) needed for equalized-odds
and demographic-parity slicing. Pure and free of model/training imports so it stays
unit-testable without loading the booster.

The decision threshold is the GLOBAL operating-point threshold — the same value for every
subgroup (prereg §8 forbids group-conditional thresholds).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinel.evaluation import metrics as base_metrics

# Minimum-power inclusion rule (prereg §3).
MIN_ROWS = 100
MIN_POSITIVES = 30


def confusion_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """TP/FP/TN/FN and TPR/FPR/selection_rate at a FIXED threshold (``y_prob >= threshold``).

    Pure. TPR (=recall/sensitivity) and FPR are the equalized-odds components; selection_rate
    is the demographic-parity quantity. Rates whose denominator is empty return ``None``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))

    pos = tp + fn
    neg = fp + tn
    n = len(y_true)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": (tp / pos) if pos else None,
        "fpr": (fp / neg) if neg else None,
        "selection_rate": ((tp + fp) / n) if n else None,
    }


def subgroup_metrics(y_true, y_prob, threshold: float, n_bins: int = 10) -> dict:
    """Discrimination + calibration + threshold rates for one subgroup.

    Merges ``base_metrics.calibration_metrics`` (ece, reliability), prevalence/counts, Brier,
    and ``confusion_at_threshold``. Guards: if the subgroup has 0 positives or is single-class,
    AUROC/AUPRC are ``None`` (undefined) rather than raising — the power rule marks such a
    subgroup descriptive-only anyway.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    out = dict(base_metrics.prevalence_and_base_rates(y_true))
    n = out["n"]
    pos = out["positives"]

    out["brier"] = float(np.mean((y_prob - y_true.astype(float)) ** 2)) if n else None
    if 0 < pos < n:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = None
        out["auprc"] = None

    cal = base_metrics.calibration_metrics(y_true, y_prob, n_bins=n_bins)
    out["ece"] = cal["ece"]
    out["n_bins_used"] = cal["n_bins_used"]
    out["reliability"] = cal["reliability"]

    out.update(confusion_at_threshold(y_true, y_prob, threshold))
    out["threshold"] = float(threshold)
    return out


def is_powered(m: dict) -> bool:
    """Prereg §3 inclusion rule: at least ``MIN_ROWS`` rows AND ``MIN_POSITIVES`` positives."""
    return m["n"] >= MIN_ROWS and m["positives"] >= MIN_POSITIVES


def pairwise_difference(m_a: dict, m_b: dict, key: str) -> float | None:
    """Point difference ``m_a[key] - m_b[key]`` (``None`` if either value is undefined)."""
    a = m_a.get(key)
    b = m_b.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def four_fifths_ratio(m_a: dict, m_b: dict, key: str) -> float:
    """Ratio ``m_a[key]/m_b[key]`` for the EEOC 80%-rule secondary screen.

    Returns ``nan`` when either value is undefined or the denominator is zero (ratio tests
    are unstable at low rates — prereg §6 uses this only as a reported secondary screen).
    """
    a = m_a.get(key)
    b = m_b.get(key)
    if a is None or b is None or not b:
        return float("nan")
    return float(a) / float(b)

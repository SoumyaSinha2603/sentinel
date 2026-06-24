"""
Model-agnostic evaluation metrics for Sentinel.

Pure functions over ``y_true`` (0/1) and ``y_prob`` (predicted probability of the
positive class). This layer is reused by every model in the project (baseline, LightGBM,
calibrated, fairness-sliced), so it stays general and free of model/global state.

At the ~11% positive prevalence of this product, **AUPRC and calibration matter more
than AUROC** — they are first-class here, not afterthoughts. Plot helpers import
matplotlib locally (with a non-interactive backend) so importing this module stays light.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUROC, average precision (AUPRC), and Brier score."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Expected Calibration Error (ECE) and reliability-curve points.

    Uses **quantile (equal-frequency) bins** so the low positive prevalence does not
    leave bins empty or unstable. ECE is the count-weighted mean absolute gap between
    each bin's mean predicted probability and its observed positive frequency. The
    reliability points are returned so a caller can plot bin mean-pred vs observed-freq.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    # Equal-frequency bin edges from prediction quantiles; collapse duplicate edges
    # (e.g. a constant predictor) so we never create empty/degenerate bins.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    # np.digitize needs interior edges; clamp bin ids into [0, n_real_bins - 1].
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    bins = []
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        obs_freq = float(y_true[mask].mean())
        bins.append(
            {
                "mean_pred": mean_pred,
                "obs_freq": obs_freq,
                "count": count,
            }
        )
        ece += (count / n) * abs(mean_pred - obs_freq)

    return {"ece": float(ece), "n_bins_used": len(bins), "reliability": bins}


def prevalence_and_base_rates(y_true: np.ndarray) -> dict:
    """Positive rate, total n, and positive count."""
    y_true = np.asarray(y_true)
    n = int(len(y_true))
    positives = int(np.sum(y_true == 1))
    return {
        "n": n,
        "positives": positives,
        "prevalence": float(positives / n) if n else 0.0,
    }


def summarize(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Merge discrimination, calibration, and prevalence metrics into one dict."""
    merged: dict = {}
    merged.update(discrimination_metrics(y_true, y_prob))
    merged.update(calibration_metrics(y_true, y_prob, n_bins=n_bins))
    merged.update(prevalence_and_base_rates(y_true))
    return merged


def _new_axes():
    """Local matplotlib import with a non-interactive backend (keeps module import light)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, path, n_bins: int = 10):
    """Save a reliability diagram (bin mean-pred vs observed-freq) with a diagonal."""
    plt = _new_axes()
    cal = calibration_metrics(y_true, y_prob, n_bins=n_bins)
    mean_pred = [b["mean_pred"] for b in cal["reliability"]]
    obs_freq = [b["obs_freq"] for b in cal["reliability"]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(mean_pred, obs_freq, marker="o", label="model")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency (bin)")
    ax.set_title(f"Reliability curve (ECE={cal['ece']:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, path):
    """Save side-by-side ROC and precision-recall curves."""
    plt = _new_axes()
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true == 1))

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 5))
    ax_roc.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")

    ax_pr.plot(recall, precision, label=f"AUPRC={auprc:.3f}")
    ax_pr.axhline(prevalence, linestyle="--", color="grey", label=f"prevalence={prevalence:.3f}")
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

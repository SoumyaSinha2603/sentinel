"""Decision-curve analysis (net benefit) on calibrated probabilities.

Net benefit puts true positives and false positives on a common scale via the
risk-tolerance odds ``p_t / (1 - p_t)`` (Vickers & Elkin 2006). The ``p_t`` axis is an
expected-utility axis and is only meaningful on **calibrated** probabilities — DCA here is
always computed on ``p_cal``. The treat-all curve has a closed form; treat-none is 0.
"""

from __future__ import annotations

import numpy as np

# Threshold grid: 0.01 .. 0.50 step 0.005 (rounded to kill float drift).
THRESHOLDS = np.round(np.arange(0.01, 0.5001, 0.005), 5)


def net_benefit_all(prevalence: float, p_t) -> np.ndarray:
    """Treat-all net benefit (closed form): prev - (1-prev) * odds(p_t)."""
    p_t = np.asarray(p_t, dtype=float)
    return prevalence - (1.0 - prevalence) * (p_t / (1.0 - p_t))


def net_benefit_model_grid(y: np.ndarray, p: np.ndarray, thresholds=THRESHOLDS) -> np.ndarray:
    """NB_model = TP/N - (FP/N)*odds(p_t) across a threshold grid (sort-once, O(N log N))."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros(len(thresholds))

    order = np.argsort(-p, kind="stable")  # descending by probability
    p_sorted = p[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)

    # n_flag(t) = #{p >= t}. p_sorted is descending, so -p_sorted is ascending.
    n_flag = np.searchsorted(-p_sorted, -thresholds, side="right")
    take = np.clip(n_flag - 1, 0, n - 1)
    tp = np.where(n_flag > 0, cum_tp[take], 0)
    fp = np.where(n_flag > 0, cum_fp[take], 0)
    odds = thresholds / (1.0 - thresholds)
    return tp / n - (fp / n) * odds


def net_benefit_model(y: np.ndarray, p: np.ndarray, p_t: float) -> float:
    """Scalar NB_model at one threshold."""
    return float(net_benefit_model_grid(y, p, np.array([p_t]))[0])


def dca_grid(y: np.ndarray, p: np.ndarray, thresholds=THRESHOLDS) -> dict:
    """Point-estimate DCA: NB_model, NB_all (closed form), NB_none across the grid."""
    thresholds = np.asarray(thresholds, dtype=float)
    prevalence = float(np.mean(np.asarray(y) == 1))
    return {
        "thresholds": thresholds,
        "nb_model": net_benefit_model_grid(y, p, thresholds),
        "nb_all": net_benefit_all(prevalence, thresholds),
        "nb_none": np.zeros(len(thresholds)),
    }


def _patient_row_index(patient_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique patients + the row positions belonging to each (for grouped resampling)."""
    uniq, inverse = np.unique(patient_ids, return_inverse=True)
    rows = [np.where(inverse == i)[0] for i in range(len(uniq))]
    return uniq, rows


def dca_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    thresholds=THRESHOLDS,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-grouped bootstrap 95% band for NB_model across the grid.

    Resamples unique ``patient_nbr`` WITH replacement to the same unique-patient count,
    gathers all their rows, recomputes NB_model. Returns (2.5th, 97.5th) percentile bands.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    uniq, rows = _patient_row_index(np.asarray(patient_ids))
    n_patients = len(uniq)
    rng = np.random.default_rng(seed)

    samples = np.empty((n_boot, len(thresholds)), dtype=float)
    idx_space = np.arange(n_patients)
    for b in range(n_boot):
        chosen = rng.choice(idx_space, size=n_patients, replace=True)
        boot_rows = np.concatenate([rows[i] for i in chosen])
        samples[b] = net_benefit_model_grid(y[boot_rows], p[boot_rows], thresholds)

    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)
    return lower, upper


def useful_band(grid: dict) -> dict:
    """Where NB_model strictly beats BOTH treat-all and treat-none, as a threshold range."""
    mask = (grid["nb_model"] > grid["nb_all"]) & (grid["nb_model"] > grid["nb_none"])
    thr = grid["thresholds"]
    if not mask.any():
        return {"any": False, "min": None, "max": None, "contiguous": True}
    sel = thr[mask]
    # contiguous iff every threshold between min and max is also selected
    contiguous = bool(mask[(thr >= sel.min()) & (thr <= sel.max())].all())
    return {"any": True, "min": float(sel.min()), "max": float(sel.max()), "contiguous": contiguous}

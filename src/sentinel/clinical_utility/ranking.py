"""Worklist ranking metrics: precision@k, recall@k, lift, alert burden, operating points.

These are RANKING metrics, so they are calibration-invariant: isotonic is monotone, so
ranking by calibrated probability gives the same worklist as ranking by the raw booster
score. Calibration's only role here is the *implied probability threshold* of an operating
point. Ties in the calibrated probability (isotonic's flat regions) are broken by the
underlying raw score — which is exactly why the worklist is invariant.
"""

from __future__ import annotations

import numpy as np

# Highlighted operating budgets (standard care-team capacities) — pre-declared, not tuned.
BUDGETS = (0.05, 0.10, 0.20)

# Worklist k-grid: 1% .. 50% step 1% (5/10/20% fall on the grid).
K_GRID = np.round(np.arange(0.01, 0.5001, 0.01), 4)


def worklist_order(p_primary: np.ndarray, p_secondary: np.ndarray | None = None) -> np.ndarray:
    """Descending worklist order. ``p_secondary`` (e.g. raw score) breaks ties in primary."""
    p_primary = np.asarray(p_primary, dtype=float)
    if p_secondary is None:
        return np.argsort(-p_primary, kind="stable")
    p_secondary = np.asarray(p_secondary, dtype=float)
    # np.lexsort: last key is primary. Negate for descending.
    return np.lexsort((-p_secondary, -p_primary))


def _n_flag(k: float, n: int) -> int:
    return int(min(max(round(k * n), 1), n))


def precision_recall_at_k(y: np.ndarray, order: np.ndarray, k_grid=K_GRID) -> list[dict]:
    """precision@k (PPV), recall@k (sens), lift, alert_rate, NNF over a k-grid."""
    y = np.asarray(y)
    n = len(y)
    n_pos = int(np.sum(y == 1))
    prevalence = n_pos / n if n else 0.0
    cum_tp = np.cumsum(y[order] == 1)

    out: list[dict] = []
    for k in k_grid:
        nf = _n_flag(float(k), n)
        tp = int(cum_tp[nf - 1])
        precision = tp / nf
        recall = tp / n_pos if n_pos else 0.0
        lift = precision / prevalence if prevalence else 0.0
        nnf = (1.0 / precision) if precision > 0 else float("inf")
        out.append(
            {
                "k": float(k),
                "n_flag": nf,
                "precision": precision,
                "recall": recall,
                "lift": lift,
                "alert_rate": float(k),
                "nnf": nnf,
            }
        )
    return out


def implied_threshold(p_scores: np.ndarray, order: np.ndarray, k: float) -> float:
    """Calibrated probability of the lowest-ranked flagged patient at budget ``k``."""
    p_scores = np.asarray(p_scores, dtype=float)
    nf = _n_flag(float(k), len(p_scores))
    return float(p_scores[order][nf - 1])


def precision_recall_invariant(
    y: np.ndarray, p_cal: np.ndarray, p_raw: np.ndarray, k_grid=K_GRID
) -> float:
    """Max |precision@k(cal-order) - precision@k(raw-order)| over the grid (expect ~0)."""
    order_cal = worklist_order(p_cal, p_raw)  # cal primary, raw tiebreak
    order_raw = worklist_order(p_raw)
    pc = precision_recall_at_k(y, order_cal, k_grid)
    pr = precision_recall_at_k(y, order_raw, k_grid)
    return max(abs(a["precision"] - b["precision"]) for a, b in zip(pc, pr, strict=True))


def bootstrap_pr_ci(
    y: np.ndarray,
    p_primary: np.ndarray,
    p_secondary: np.ndarray,
    patient_ids: np.ndarray,
    k: float,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Patient-grouped bootstrap 95% CI for precision@k and recall@k at one budget."""
    y = np.asarray(y)
    p_primary = np.asarray(p_primary, dtype=float)
    p_secondary = np.asarray(p_secondary, dtype=float)
    pid = np.asarray(patient_ids)
    uniq, inverse = np.unique(pid, return_inverse=True)
    rows = [np.where(inverse == i)[0] for i in range(len(uniq))]
    n_patients = len(uniq)
    rng = np.random.default_rng(seed)

    precisions = np.empty(n_boot)
    recalls = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(np.arange(n_patients), size=n_patients, replace=True)
        boot_rows = np.concatenate([rows[i] for i in chosen])
        yb = y[boot_rows]
        order_b = worklist_order(p_primary[boot_rows], p_secondary[boot_rows])
        res = precision_recall_at_k(yb, order_b, [k])[0]
        precisions[b] = res["precision"]
        recalls[b] = res["recall"]

    return {
        "precision_ci": (
            float(np.percentile(precisions, 2.5)),
            float(np.percentile(precisions, 97.5)),
        ),
        "recall_ci": (float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))),
    }

"""
Cohort builder — raw frame → modeling cohort (Week-2 closeout).

Pure, in-memory transform that runs *before* the locked split harness
(`sentinel.evaluation.splits`, which is frozen and unchanged by this module). It applies
three cohort decisions, reports exactly what it removed and why, and never mutates the
raw CSV or the input dataframe.

Decisions implemented (see `build_cohort`):
  1. Remove death/hospice discharge encounters — structurally ineligible for 30-day
     readmission prediction (a deceased/hospice-bound patient is never scored at
     discharge in production). Matches Strack et al. 2014.
  2. Drop only *zero-variance* columns (exactly one unique value); keep near-constant
     ones (they may carry rare-subgroup signal).
  3. Default = all eligible encounters; optional `first_encounter_only` keeps each
     patient's earliest encounter for a robustness/sensitivity check.

Run:

    python -m sentinel.data.cohort
"""

from __future__ import annotations

import sys

import pandas as pd

from sentinel.config import REPORTS_DIR
from sentinel.data.load import fetch_raw
from sentinel.evaluation.splits import (
    GROUP_COL,
    IDENTIFIER_COLS,
    TARGET_COL,
    make_binary_target,
)

DISPOSITION_COL = "discharge_disposition_id"
ENCOUNTER_COL = "encounter_id"

# Death / hospice discharge dispositions. Source: the dataset's own ``IDS_mapping.csv``
# (Diabetes 130-US Hospitals, UCI id=296) and Strack et al. 2014 — ids 11/13/14/19/20/21
# are "Expired" and "Hospice" variants. A deceased or hospice-bound patient cannot be
# readmitted within 30 days and would never be scored at discharge, so these encounters
# are ineligible for the modeling cohort. (Id 21 has 0 rows in this dataset.)
DEATH_HOSPICE_DISPOSITIONS = frozenset({11, 13, 14, 19, 20, 21})

# Columns that are kept regardless of variance: identifiers and the raw target.
_VARIANCE_EXEMPT = set(IDENTIFIER_COLS) | {TARGET_COL}


def _zero_variance_columns(df: pd.DataFrame) -> list[str]:
    """Columns with exactly one unique value (NaN counted), excluding ids and target."""
    return [
        col
        for col in df.columns
        if col not in _VARIANCE_EXEMPT and df[col].nunique(dropna=False) == 1
    ]


def build_cohort(df: pd.DataFrame, *, first_encounter_only: bool = False) -> pd.DataFrame:
    """Turn the raw loaded frame into the modeling cohort. Pure — copies its input.

    Applies Decision 1 (remove death/hospice encounters), then Decision 2 (drop
    zero-variance columns), and Decision 3 (first-encounter-only) when flagged.

    Zero-variance columns are detected on the post-Decision-1 eligible cohort (per spec)
    and the same column set is dropped in both modes, so ``first_encounter_only`` never
    changes which columns survive — only which rows. Row filters (Decisions 1 & 3) and
    the column drop are order-independent given this.
    """
    cohort = df.copy()

    # Decision 1 — cohort eligibility: remove death/hospice encounters.
    cohort = cohort[~cohort[DISPOSITION_COL].isin(DEATH_HOSPICE_DISPOSITIONS)]

    # Decision 2 — detect zero-variance columns on the post-Decision-1 cohort.
    drop_cols = _zero_variance_columns(cohort)

    # Decision 3 (optional) — keep each patient's earliest encounter by encounter_id.
    if first_encounter_only:
        keep_idx = cohort.groupby(GROUP_COL)[ENCOUNTER_COL].idxmin()
        cohort = cohort.loc[keep_idx]

    cohort = cohort.drop(columns=drop_cols)
    return cohort.reset_index(drop=True)


def cohort_summary(raw_df: pd.DataFrame, cohort_df: pd.DataFrame) -> dict:
    """Report what the cohort transform removed and why (no mutation)."""
    raw_y = make_binary_target(raw_df)

    death_hospice_mask = raw_df[DISPOSITION_COL].isin(DEATH_HOSPICE_DISPOSITIONS)
    death_hospice_removed = int(death_hospice_mask.sum())
    # Positive (<30) cases removed by the death/hospice filter — should be near zero;
    # if not, the eligibility cut is discarding real readmissions and warrants a look.
    positives_removed_death_hospice = int(raw_y[death_hospice_mask].sum())

    cohort_y = make_binary_target(cohort_df)
    columns_dropped = [c for c in raw_df.columns if c not in cohort_df.columns]

    return {
        "rows_before": int(len(raw_df)),
        "rows_after": int(len(cohort_df)),
        "total_rows_removed": int(len(raw_df) - len(cohort_df)),
        "death_hospice_removed": death_hospice_removed,
        "positives_removed_death_hospice": positives_removed_death_hospice,
        "columns_dropped": columns_dropped,
        "patients_before": int(raw_df[GROUP_COL].nunique()),
        "patients_after": int(cohort_df[GROUP_COL].nunique()),
        "prevalence_before": float(raw_y.mean()),
        "prevalence_after": float(cohort_y.mean()),
    }


def _render_summary(summary: dict) -> str:
    dropped = summary["columns_dropped"]
    dropped_str = ", ".join(f"`{c}`" for c in dropped) if dropped else "none"

    lines: list[str] = []
    lines.append("# Cohort Summary — UCI Diabetes-130")
    lines.append("")
    lines.append(
        "> Default modeling cohort (all eligible encounters). Built by "
        "`sentinel.data.cohort` from the raw frame *before* the locked split harness. "
        "Pure transform — the raw CSV is never modified."
    )
    lines.append("")
    lines.append("## Eligibility & filtering")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| rows before | {summary['rows_before']:,} |")
    lines.append(f"| rows after | {summary['rows_after']:,} |")
    lines.append(f"| total rows removed | {summary['total_rows_removed']:,} |")
    lines.append(f"| death/hospice encounters removed | {summary['death_hospice_removed']:,} |")
    lines.append(
        "| of those, positive (`<30`) cases removed | "
        f"{summary['positives_removed_death_hospice']:,} |"
    )
    lines.append(f"| unique patients before | {summary['patients_before']:,} |")
    lines.append(f"| unique patients after | {summary['patients_after']:,} |")
    lines.append("")
    lines.append("## Decision 1 — death/hospice removal")
    lines.append("")
    lines.append(
        f"Removed encounters with `discharge_disposition_id` ∈ "
        f"{sorted(DEATH_HOSPICE_DISPOSITIONS)} (Expired / Hospice variants per the "
        "dataset IDS mapping and Strack et al. 2014). These patients are structurally "
        "ineligible for 30-day readmission prediction. `discharge_disposition_id` is "
        "**kept** as a feature — the remaining values are legitimate and known at discharge."
    )
    lines.append("")
    lines.append(
        f"Positive (`<30`) cases caught by this filter: "
        f"**{summary['positives_removed_death_hospice']:,}** "
        "(expected near zero — a deceased/hospice patient is not a 30-day readmission)."
    )
    lines.append("")
    lines.append("## Decision 2 — zero-variance columns dropped")
    lines.append("")
    lines.append(
        f"Dropped columns with exactly one unique value (detected on the post-Decision-1 "
        f"cohort): {dropped_str}. Near-constant (≥99% one value) columns are **kept** — "
        "they may carry rare-subgroup signal. Identifiers and target are always kept."
    )
    lines.append("")
    lines.append("## Target prevalence")
    lines.append("")
    lines.append("| | `<30` prevalence |")
    lines.append("|---|---:|")
    lines.append(f"| before (raw) | {summary['prevalence_before']:.2%} |")
    lines.append(f"| after (cohort) | {summary['prevalence_after']:.2%} |")
    lines.append("")
    lines.append(
        "_Removing structurally-ineligible death/hospice encounters slightly raises "
        "prevalence; this is the honest eligible-population base rate._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    raw_df = fetch_raw()
    cohort_df = build_cohort(raw_df)
    summary = cohort_summary(raw_df, cohort_df)
    report = _render_summary(summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "cohort_summary.md"
    out_path.write_text(report, encoding="utf-8")

    # Force UTF-8 on stdout so the report's Unicode (∈, ≥, em-dashes) prints on a
    # Windows cp1252 console instead of raising UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()

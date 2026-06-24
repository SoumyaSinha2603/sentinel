# Cohort Summary — UCI Diabetes-130

> Default modeling cohort (all eligible encounters). Built by `sentinel.data.cohort` from the raw frame *before* the locked split harness. Pure transform — the raw CSV is never modified.

## Eligibility & filtering

| metric | value |
|---|---:|
| rows before | 101,766 |
| rows after | 99,343 |
| total rows removed | 2,423 |
| death/hospice encounters removed | 2,423 |
| of those, positive (`<30`) cases removed | 43 |
| unique patients before | 71,518 |
| unique patients after | 69,990 |

## Decision 1 — death/hospice removal

Removed encounters with `discharge_disposition_id` ∈ [11, 13, 14, 19, 20, 21] (Expired / Hospice variants per the dataset IDS mapping and Strack et al. 2014). These patients are structurally ineligible for 30-day readmission prediction. `discharge_disposition_id` is **kept** as a feature — the remaining values are legitimate and known at discharge.

Positive (`<30`) cases caught by this filter: **43** (expected near zero — a deceased/hospice patient is not a 30-day readmission).

## Decision 2 — zero-variance columns dropped

Dropped columns with exactly one unique value (detected on the post-Decision-1 cohort): `examide`, `citoglipton`. Near-constant (≥99% one value) columns are **kept** — they may carry rare-subgroup signal. Identifiers and target are always kept.

## Target prevalence

| | `<30` prevalence |
|---|---:|
| before (raw) | 11.16% |
| after (cohort) | 11.39% |

_Removing structurally-ineligible death/hospice encounters slightly raises prevalence; this is the honest eligible-population base rate._

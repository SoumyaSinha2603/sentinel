# Spec: Cohort Builder (Week-2 closeout)

> Hand to Claude Code. Read `CLAUDE.md` first; locked decisions apply. **Present a plan
> and get my approval before writing code or committing.**

## Context

The leakage audit is done and the evaluation harness (`src/sentinel/evaluation/splits.py`)
is **frozen — do not modify it**. Three cohort decisions have now been made and must be
implemented as a documented, reproducible raw → modeling-cohort transform that runs
*before* splitting. The harness simply receives the eligible cohort; its logic is unchanged.

## Objective

Add `src/sentinel/data/cohort.py`: a pure, well-documented module that turns the raw
loaded dataframe into the modeling cohort by applying the decisions below. Report what it
removed and why. Do not mutate the raw CSV; operate in-memory on a copy.

## Decisions to implement

**Decision 1 — Remove death/hospice encounters (cohort eligibility).**
Remove encounters where `discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21} (death and
hospice variants — the ~2,423 flagged in the audit; id 21 simply has 0 rows). Rationale to
put in the docstring: a deceased or hospice-bound patient is structurally ineligible for a
30-day readmission prediction and would never be scored at discharge in production. This
matches Strack et al. 2014. Keep `discharge_disposition_id` as a feature afterward — the
remaining values are legitimate and known at discharge.

**Decision 2 — Drop only zero-variance columns; keep near-constant ones.**
Drop columns with exactly **one** unique value (computed on the cohort *after* Decision 1):
`examide`, `citoglipton`, and any other single-valued column — detect dynamically, do not
hardcode the full list. **Keep** the merely near-constant (≥99% one value) columns; they
may carry rare-subgroup signal and the model handles them fine. Always keep identifiers and
target regardless of variance.

**Decision 3 — All eligible encounters is the primary cohort; first-encounter-only is an
optional mode.**
Default behavior keeps all eligible encounters (relies on the locked patient-grouped split
for independence, matches product semantics: every discharge is scored). Provide a flag
`first_encounter_only: bool = False` that, when True, keeps only each patient's earliest
encounter by `encounter_id` (ascending) — for a documented robustness/sensitivity check, not
the primary path.

## Module shape

```python
def build_cohort(df, *, first_encounter_only=False) -> pd.DataFrame: ...
# applies Decision 1, then Decision 2 (and Decision 3 if flagged).
# Pure: copies input, never touches the raw file.

def cohort_summary(raw_df, cohort_df) -> dict: ...
# rows before/after, encounters removed by reason (death/hospice count),
# columns dropped (names), unique patients, binary-target prevalence before/after.
```

A `main()` that builds the default cohort, prints `cohort_summary`, and writes
`reports/cohort_summary.md` (versioned artifact). Use `make_binary_target` from
`evaluation/splits.py` for prevalence — do not redefine the target.

## Constants
Define `DEATH_HOSPICE_DISPOSITIONS = frozenset({11, 13, 14, 19, 20, 21})` with a source
comment referencing the dataset's IDS mapping / Strack et al. 2014.

## Tests (`tests/test_cohort.py`)
- `test_death_hospice_removed` — no rows with those disposition ids remain in the default cohort.
- `test_zero_variance_dropped` — a crafted single-valued column is dropped; a 99%-but-not-
  constant column survives; identifiers and target always survive.
- `test_first_encounter_mode` — with the flag on, each `patient_nbr` appears exactly once and
  it is the minimum `encounter_id` for that patient.
- `test_cohort_is_pure` — input dataframe is unchanged after `build_cohort` (no mutation).
Keep all existing tests green.

## Do NOT
- Do not modify `splits.py`, the raw CSV, or the target definition.
- Do not impute missing values, encode features, or drop near-constant (non-zero-variance)
  columns — that is later feature-engineering work, out of scope here.
- Do not drop `discharge_disposition_id` itself.
- Do not commit without showing me the plan and the printed `reports/cohort_summary.md`.

## Process
1. Restate understanding + short plan. Wait for approval.
2. Implement, keep ruff-clean. Run `python -m sentinel.data.cohort` and `pytest`.
3. Show me `reports/cohort_summary.md` and test results.
4. On approval: update CLAUDE.md "Current state" (cohort defined; give final row/patient/
   prevalence numbers), then commit `feat(data): cohort builder (eligibility + variance filters)`.
```

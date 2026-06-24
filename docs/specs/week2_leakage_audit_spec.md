# Spec: Week-2 Leakage Audit + Locked Evaluation Harness

> Hand this to Claude Code as a single task. Read `CLAUDE.md` in the repo root first;
> all locked decisions there apply. **Present a plan and get my approval before writing
> any code or committing.**

## Objective

Build (1) the **locked evaluation harness** (patient-grouped splits) and (2) a
**read-only leakage audit** that inspects the UCI Diabetes-130 dataset and reports every
leakage vector with evidence. This is the Week-2 hard gate: no modeling is valid until
this passes and the split is locked.

Two guiding rules, non-negotiable:
- The audit **reports**; it does **not** mutate or drop anything. Cohort decisions
  (e.g. removing death/hospice encounters) are made by a human AFTER reviewing the report.
- Honesty over metrics. A surprisingly strong signal is a **leakage suspect to surface**,
  never something to keep quiet about.

## Dataset facts (already verified — don't re-derive)

- Loaded by `src/sentinel/data/load.py` → `data/raw/diabetes_130.csv` (DVC-tracked, also
  reloadable in-memory via `load.fetch_raw()`). **Do not modify the raw file.**
- 101,766 encounters × 50 columns. `encounter_id` is unique; `patient_nbr` repeats
  (71,518 unique patients → ~30,248 repeat encounters).
- Target column `readmitted` ∈ {`<30`, `>30`, `NO`}. Binary target: `<30` → 1 else 0
  (prevalence ≈ 11.2%).
- `encounter_id` and `patient_nbr` are identifiers and must NEVER be model features.
- Missing values are encoded as the string `?` (not NaN) in several columns
  (e.g. `weight`, `payer_code`, `medical_specialty`, `race`).

## Deliverable A — Evaluation harness (`src/sentinel/evaluation/splits.py`)

Locked config as module constants: `SEED = 42`, `TEST_SIZE = 0.2`, `N_FOLDS = 5`,
`GROUP_COL = "patient_nbr"`.

Functions:
- `make_binary_target(df) -> pd.Series` — 1 where `readmitted == "<30"`, else 0. Pure,
  no mutation of input.
- `make_holdout_split(df) -> tuple[np.ndarray, np.ndarray]` — outer train/test split
  using `GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)` with
  `groups=df[GROUP_COL]`. Returns integer positional indices.
- `make_cv_folds(df_train) -> list[tuple[np.ndarray, np.ndarray]]` — within the training
  portion, `StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)` on
  the binary target with `groups=patient_nbr`. Returns a materialized list (deterministic).
- `assert_no_group_overlap(df, idx_a, idx_b) -> None` — raises `AssertionError` if any
  `patient_nbr` appears in both index sets. Call it inside `make_holdout_split` before
  returning, so the guarantee is structural, not optional.

This module is **frozen once approved**. Every model in the project is evaluated through
exactly these splits — no per-model bespoke splitting.

## Deliverable B — Leakage audit (`src/sentinel/data/leakage_checks.py`, replace the stub)

A `run_leakage_audit(df) -> dict` that performs the checks below, plus a `main()` that
runs it on the loaded data, prints a readable report, and writes the same report to
`reports/leakage_audit.md` (create `reports/` if needed). The markdown report is a
versioned portfolio artifact — make it clean and human-readable.

Checks (each appears in the report with concrete numbers):

1. **Identifier exclusion** — confirm `encounter_id` (cardinality = n rows) and
   `patient_nbr` are present, report their cardinality, and state explicitly that both
   are excluded from any feature set.
2. **Patient repetition** — encounters, unique patients, repeat-encounter count, and the
   max encounters for a single patient.
3. **Grouped-split integrity** — build the locked holdout split, call
   `assert_no_group_overlap`, and report: train/test row counts, unique-patient counts in
   each, and target prevalence in each (should be close to 11.2% in both).
4. **Post-outcome disposition leakage** — cross-tabulate `discharge_disposition_id`
   against the binary target; for each disposition id report encounter count and `<30`
   readmission rate. Flag the candidate death/hospice ids **{11, 13, 14, 19, 20, 21}**
   (verify their meaning against the dataset's variable documentation / IDS mapping if
   available — do not assume blindly). These should show a near-0% 30-day readmission rate
   because the patient died or entered hospice. **Report and flag only; do not drop.**
5. **Constant / near-constant columns** — flag any column with a single unique value or
   ≥99% concentration in one value.
6. **Duplicate rows** — count fully duplicated rows after excluding `encounter_id`.
7. **Missingness** — per-column rate of `?` and of NaN; list the worst offenders.
8. **Univariate predictive-power smell test** — for each candidate feature (identifiers
   excluded), estimate single-feature AUROC against the binary target. To avoid optimism,
   compute it under the grouped CV folds with any target/category encoding fit **inside**
   each training fold and applied to the validation fold (never fit on the whole data).
   Report the top 10 features by mean AUROC and **flag any feature with mean AUROC > 0.70**
   as a leakage suspect for manual review. (On this dataset, no legitimate single feature
   should be that predictive; if one is, we investigate before trusting any model.)
9. **Summary verdict** — a short section listing each flagged item and whether it needs a
   human decision before modeling.

## Deliverable C — Tests (`tests/`)

- `test_target_binarization` — `<30`→1, `>30`→0, `NO`→0 on a small crafted frame.
- `test_split_no_patient_overlap` — patient sets of train and test are disjoint.
- `test_split_determinism` — calling `make_holdout_split` twice yields identical indices.
- `test_identifiers_not_in_features` — whatever helper defines the feature list excludes
  `encounter_id` and `patient_nbr`.

Keep the existing smoke tests green. CI runs `ruff` + `pytest` on push.

## Do NOT

- Do not modify `data/raw/diabetes_130.csv` or re-run `dvc add`.
- Do not drop, filter, or impute any rows/columns in this task — audit only.
- Do not train any predictive model beyond the single-feature smell test in check 8.
- Do not "improve" the target definition or split strategy — they are locked in CLAUDE.md.
- Do not commit without showing me the plan and the resulting `reports/leakage_audit.md`.

## Process

1. Restate your understanding and present a short plan. Wait for approval.
2. Implement A, B, C. Keep everything `ruff`-clean (pre-commit will enforce it).
3. Run `python -m sentinel.data.leakage_checks` and `pytest`. Show me the printed audit
   report and the test results.
4. On approval, commit with: `feat(eval): locked patient-grouped splits + leakage audit`.

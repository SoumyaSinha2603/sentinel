# Spec: Baseline Model + Evaluation Metrics Layer

> Hand to Claude Code. Read `CLAUDE.md` first; locked decisions apply. **Present a plan
> and get my approval before writing code or committing.**

## Why this step exists (read before planning)

Before any feature engineering or tuned model, we establish an **honest floor**: the
simplest defensible model run through the frozen split harness. Its job is to (1) prove the
harness works end-to-end on real modeling, (2) set the number every future model must beat,
and (3) confirm the honest ceiling is roughly where we expect (~0.68 AUROC). The
**metrics layer** built here is reused by every later model (LightGBM, calibrated,
fairness-sliced), so it must be clean and general.

**Hard interpretation rule:** if the baseline AUROC lands anywhere near or above ~0.85,
that is a leakage alarm, not a success — STOP and report it loudly. The audit was clean, so
we expect ~0.60–0.66 for a simple model.

## Scope

Two deliverables: a reusable metrics module, and a baseline training/evaluation script.
Minimal preprocessing only — this is NOT feature engineering (that's the next step).

## Deliverable A — Metrics layer (`src/sentinel/evaluation/metrics.py`)

A model-agnostic module taking `y_true` and `y_prob` (predicted probabilities of the
positive class). Functions, all pure:

- `discrimination_metrics(y_true, y_prob) -> dict` — AUROC, average precision (AUPRC),
  and Brier score.
- `calibration_metrics(y_true, y_prob, n_bins=10) -> dict` — Expected Calibration Error
  (ECE) and the reliability-curve points (bin mean predicted prob vs bin observed
  frequency). Use a quantile/strategy that handles the 11% prevalence sensibly.
- `prevalence_and_base_rates(y_true) -> dict` — positive rate, n, positives.
- `summarize(y_true, y_prob) -> dict` — calls the above and returns one merged dict.
- `plot_reliability(y_true, y_prob, path)` and `plot_roc_pr(y_true, y_prob, path)` —
  save matplotlib figures to `reports/figures/`. No styling fuss; clear labels and a
  diagonal reference line on the reliability plot.

Design notes: pure functions, no global state; matplotlib only inside the plot helpers
(import locally). AUPRC and calibration matter more than AUROC at this prevalence, so make
them first-class, not afterthoughts.

## Deliverable B — Baseline runner (`src/sentinel/models/baseline.py`)

Run two simple baselines through the **frozen** harness (`evaluation/splits.py`) on the
**cohort** (`data/cohort.build_cohort`, default mode):

1. **Trivial reference** — predicts the train positive rate for everyone (a constant).
   Establishes the floor AUPRC (= prevalence) and shows AUROC = 0.5.
2. **Simple logistic regression** — on a small, *safe*, numeric-only feature set, no
   leakage risk: `number_inpatient`, `number_emergency`, `number_outpatient`,
   `number_diagnoses`, `num_medications`, `num_lab_procedures`, `time_in_hospital`,
   `num_procedures`. Standardize features; `class_weight="balanced"`;
   `random_state=SEED`. (Keep it deliberately minimal — this is a floor, not a contender.)

Protocol:
- Use `make_holdout_split` for the train/test split and `make_binary_target` for labels.
- Fit on train, evaluate on the held-out test set via `metrics.summarize`.
- Also report mean ± std AUROC/AUPRC across `make_cv_folds` on the training portion (so we
  see variance, not just a point estimate).
- Log params, metrics, and the figure artifacts to **MLflow** (local `./mlruns`), one run
  per baseline, clearly named. Keep MLflow usage minimal and contained.
- `main()` runs both baselines, prints a comparison table, writes
  `reports/baseline_results.md`, and saves the reliability + ROC/PR figures.

## Tests (`tests/test_metrics.py`)
- `test_perfect_and_random_auroc` — AUROC ≈ 1.0 for perfectly separated toy data, ≈ 0.5
  for random.
- `test_brier_and_ece_bounds` — Brier ∈ [0,1]; ECE ∈ [0,1]; a well-calibrated toy case
  gives low ECE.
- `test_summarize_keys` — `summarize` returns the expected keys.
- `test_constant_baseline_auroc_half` — constant predictor yields AUROC = 0.5 and
  AUPRC ≈ prevalence.
Keep all existing tests green. Do not put slow model fits in the test suite — metrics tests
use small synthetic arrays.

## Do NOT
- Do not modify `splits.py`, `cohort.py`, the raw CSV, or the target definition.
- Do not do feature engineering: no categorical encoding, no missingness imputation beyond
  dropping rows with NaN *within the 8 chosen numeric columns* if any exist (report how
  many), no interaction terms, no diagnosis-code grouping. That is the next spec.
- Do not tune hyperparameters or try to maximize the score. This is a floor.
- Do not commit without showing me the plan, `reports/baseline_results.md`, and the test
  results.

## Process
1. Restate understanding + short plan. Wait for approval.
2. Implement A, B, tests. Keep ruff-clean.
3. Run `python -m sentinel.models.baseline` + `pytest`. Show me the baseline results table,
   the CV mean±std, and confirm the MLflow runs + figures were written.
4. On approval: update CLAUDE.md "Current state" with the baseline numbers, then commit
   `feat(models): evaluation metrics layer + baseline models`.
```

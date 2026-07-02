# CLAUDE.md — Sentinel

This file is the working brief for any Claude Code session in this repo. Read it
fully at the start of every session. It encodes decisions that are **locked** — do
not silently re-litigate them. If a change would contradict something here, stop and
flag it to the user first.

## What Sentinel is

An explainable, fairness-audited, calibrated **30-day hospital readmission-risk
decision-support system**. B2B tool: gives hospital care teams a ranked patient
worklist with SHAP reason codes, DiCE counterfactual interventions, subgroup fairness
monitoring, and drift detection. It is decision support, **not** a diagnostic device.

The project is a flagship portfolio piece for master's applications (target: applied
ML / data science programs) and industry interviews. It must be finishable in ~5–6
months at ~25 hrs/week, and its credibility rests on **rigor of integration and honest
evaluation**, not on a novel algorithm or an impressive headline metric.

## Locked decisions — do not change without explicit user sign-off

- **Target definition:** `readmitted == "<30"` → 1 (positive); `">30"` and `"NO"` → 0.
  Positive rate ≈ 11.2%. This is a *30-day* readmission product. Folding `>30` into the
  positive class is a DIFFERENT product and is out of scope.
- **Honest performance ceiling:** On UCI Diabetes-130 the realistic benchmark is
  **AUROC ≈ 0.68**. Anything above ≈ 0.85 is a **data-leakage bug, not a win.** Treat a
  suspiciously high score as a defect to investigate, never as success. Never inflate,
  never tune toward a leaky number.
- **Patient-level split is mandatory.** 71,518 patients produced 101,766 encounters
  (~30k repeat encounters). Splits MUST be grouped by `patient_nbr` so a patient is in
  train OR test, never both. A naive row-wise split is forbidden — it is leakage.
- **Week-2 leakage audit is a hard gate** before any modeling. No model training is
  considered valid until the leakage checks pass and the split is locked.
- **Evaluation harness is locked once written.** Every model is judged through the same
  harness (same grouped split, same seed=42, same metrics). No per-model bespoke eval.

## Verified figures — use ONLY these

- ~$15,200 per readmission; ~$26B/year for Medicare (2018).
- DO NOT cite the $41.3B figure (outdated 2011 AHRQ framing).
- Kansagara et al. 2011 is the canonical model-performance reference: 26 models, pooled
  c-statistic 0.55–0.70.
- DO NOT use the ">350 published models" claim (no traceable support).
- DO NOT cite vendor performance comparisons as established evidence.

## Stack

- ML: LightGBM / XGBoost, SHAP, DiCE
- Serving: FastAPI, PostgreSQL
- Frontend: Next.js, Tailwind, shadcn/ui
- MLOps: MLflow (tracking), DVC (data versioning), Evidently **0.7.x** (drift — note the
  0.4→0.7 API change; target the installed 0.7 API)
- Infra: Docker; deploy backend to Render, frontend to Vercel
- Python 3.10.11 in `.venv`. Reproducibility seed = 42.

## Data

- Primary: UCI Diabetes-130 (id=296), fetched via `ucimlrepo`, loaded by
  `src/sentinel/data/load.py`, validated (101,766 rows × 50 cols), DVC-tracked at
  `data/raw/diabetes_130.csv`. The raw CSV is git-ignored; the `.dvc` pointer is tracked.
- MIMIC-IV external validation is **CUT for now** (no PhysioNet reference available).
  Revisit only if a reference appears. If reopened: MIMIC data must NEVER be pasted into
  any LLM/online service — DUA prohibits it.

## Repo layout

```
src/sentinel/   core lib: data, features, models, evaluation, fairness, explain, monitoring
api/            FastAPI service (later phase)
frontend/       Next.js app (later phase)
data/           DVC-tracked datasets (raw CSV not in git)
tests/          test suite (CI runs ruff + pytest on push)
docs/           competitive_landscape.md and design notes
```

## Conventions

- Lint/format: `ruff` (config in `pyproject.toml`). Pre-commit hooks enforce it; keep all
  code lint- and format-clean.
- Commits: conventional style (`feat:`, `fix:`, `chore:`, `docs:`). Keep commits narrowly
  scoped. Show the plan before committing anything.
- Tests must stay green; CI runs on every push to `main`.
- Windows + PowerShell environment. Repo root: the folder containing this file.

## Current state (update as work progresses)

- Phase 0 (Weeks 1–2), foundations & anti-leakage. DONE: repo scaffold, tooling, CI,
  locked env, DVC init, dataset loaded + validated + DVC-tracked.
- Week-2 leakage audit: **DONE.** Locked patient-grouped evaluation harness
  (`src/sentinel/evaluation/splits.py`, frozen — `GroupShuffleSplit` holdout +
  `StratifiedGroupKFold` folds, seed=42, patient-disjoint asserted structurally). Read-only
  leakage audit (`src/sentinel/data/leakage_checks.py`) run clean → report at
  `reports/leakage_audit.md`. Smell test honest: top single-feature AUROC 0.607, nothing
  >0.70 (no leakage). Death/hospice discharge dispositions {11,13,14,19,20} flagged
  (~2,423 encounters) plus 15 near-constant drug columns — reported only, NOT dropped.
- Cohort definition: **DONE.** Decisions locked and implemented in
  `src/sentinel/data/cohort.py` (`build_cohort`, pure raw → modeling-cohort transform,
  runs before the split harness). (1) Removed death/hospice encounters
  (`discharge_disposition_id` ∈ {11,13,14,19,20,21}); (2) dropped only zero-variance
  columns (`examide`, `citoglipton`) — near-constant columns kept; (3) all eligible
  encounters is the primary cohort, `first_encounter_only=True` is an optional
  sensitivity mode. `discharge_disposition_id` retained as a feature. Default cohort:
  **99,343 rows / 69,990 patients / 11.39% `<30` prevalence** (2,423 removed, only 43 of
  them positives). Summary at `reports/cohort_summary.md`.
- Baseline + metrics layer: **DONE.** Reusable, model-agnostic metrics
  (`src/sentinel/evaluation/metrics.py`: AUROC/AUPRC/Brier, quantile-binned ECE +
  reliability points, prevalence, plot helpers) reused by every later model. Two floor
  baselines (`src/sentinel/models/baseline.py`) through the frozen harness on the default
  cohort, MLflow-logged to `./mlruns`. Honest floor confirmed — **no leakage alarm**:
  - trivial constant: AUROC 0.500, AUPRC 0.112 (= prevalence).
  - logistic (8 safe numeric features, balanced): test **AUROC 0.6267**, AUPRC 0.1937;
    CV **AUROC 0.6336 ± 0.0075**, AUPRC 0.1986 ± 0.0086. In the expected ~0.60-0.66 band.
  - Calibration poor by design (ECE 0.366) — `class_weight="balanced"` inflates
    probabilities; expected for a floor, motivates the later calibration layer.
  - Results: `reports/baseline_results.md`; figures in `reports/figures/`.
  - **0.627 is the number every future model must beat.** Anything near/above ~0.85 is a
    leakage alarm, not a win (the runner enforces this).
- Feature engineering + engineered-feature leakage re-audit: **DONE.** Deterministic,
  target-free transform in `src/sentinel/features/build.py` (`build_features`,
  `load_and_build`): ICD-9 Strack disease grouping, age→midpoint, medication-activity
  counts, `total_prior_visits`, missingness→explicit categories (`weight` dropped),
  coded ids as nominal categoricals. **47 features = 35 categorical + 12 numeric**
  (reconciled structurally against the frozen harness; row count preserved; label-blind).
  Leakage re-audit (`audit_engineered_features` in `leakage_checks.py`) →
  `reports/leakage_audit_features.md`: verdict **CLEAN**, top single-feature CV AUROC
  `number_inpatient` 0.607 (nothing >0.70). See the Phase-1 section below for the
  LightGBM gate that follows.

## How to work in this repo

- Default to small, reviewable changes. Propose a plan, get sign-off, then implement.
- When unsure about a clinical/ML modeling choice, prefer the honest/conservative option
  and surface the tradeoff rather than optimizing a metric.

## Current state (Phase 1, step 3 done — gate passed)
- Untuned LightGBM through frozen harness: CV AUROC 0.672±0.005, AUPRC 0.227,
  Brier 0.208, ECE 0.320. Clears LR baseline 0.634 by +0.038. GATE PASS.
- 0.672 sits at the honest ~0.68 ceiling. Tuning chases ~0.005–0.01, NOT a step
  change. Anything >=0.72 = leakage/overfit signal, stop.
- Importances clinically coherent; no identifier-like feature dominating.
  WATCH: discharge_disposition_id is rank-2 by gain — interrogate in Phase 3 SHAP
  to confirm it's legit risk signal, not an administrative proxy.
- ECE 0.320 is expected (class_weight=balanced); calibration deferred to trust layer.
- HOLDOUT DISCIPLINE (locked): the 20% grouped holdout may be IDENTIFIED via
  make_holdout_split to exclude it, but is NEVER scored until the single final
  Phase 1 evaluation.
- Step 4 — tuned LightGBM: **DONE.** Focused Optuna search (`src/sentinel/models/lgbm_tuned.py`,
  60 TPE trials, 6 params, inner 3-fold grouped CV optimizing AUPRC; outer 5 frozen folds
  scored once; holdout scored exactly once; n_estimators via early stopping → 193 trees;
  `class_weight=balanced` fixed). Result: tuned **CV AUROC 0.676±0.005**, **HOLDOUT 0.677**
  (gap −0.001 → no CV-overfit, sealed-holdout discipline held). Only **+0.004** over the
  untuned gate — marginal by design, confirming the honest ~0.68 ceiling. **GATE PASS**,
  well under 0.72. `first_encounter_only` sensitivity: AUROC 0.657 (−0.019 vs all-encounters)
  — judged benign (less data + loss of prior-visit history; grouped splits already prevent
  within-patient contamination), flagged for discussion not as leakage. ECE ~0.33 carried
  forward (balanced weighting), motivating calibration. deps: `optuna>=4.9` pinned, lock
  re-generated. Report: `reports/lgbm_tuned_results.md`. **0.677 holdout is the locked
  Phase-1 performance number** — performance-chasing ends here.

## Current state — PHASE 1 COMPLETE

- Production model REGISTERED: sentinel-readmission v1, alias @phase1.
  Refit on full 80% train, locked best params, 193 trees, seed=42. Round-trip
  load-by-alias verified. Phase 4 API loads models:/sentinel-readmission@phase1.
- Locked Phase-1 holdout (recorded, NOT recomputed): AUROC 0.677 / AUPRC 0.235 /
  Brier 0.213 / ECE 0.334. Holdout now spent — do not re-score.
- Provenance attached to the run: feature_contract.json, locked_holdout_metrics.json,
  optuna_trials.csv. LOCKED_BEST_PARAMS / LOCKED_HOLDOUT_METRICS are single-source
  constants in lgbm_tuned.py.
- Full arc: 0.634 logistic → 0.672 untuned LGBM → 0.677 tuned holdout, at the honest
  ~0.68 ceiling, CV–holdout gap −0.001. Discrimination LOCKED; probabilities NOT yet
  trustworthy (ECE 0.334).

## PHASE 4 PREREQUISITES (captured now, do NOT solve before then)

- [serving] MLflow JSON serving drops pandas 'category' dtype → LightGBM categorical
  spec mismatch. Model is fine (typed-DataFrame predict works). Fix: wrap booster in a
  pyfunc whose predict casts incoming cols to declared categories before model call.
- [registry] mlruns/ is gitignored / local file store only. Deployed API needs a
  persistent backend (sqlite local, remote for deploy). Binary reproduces via register.py.

## Current state — PHASE 2, calibration step DONE

- Calibrator FIT + SELECTED + REGISTERED. Code: `src/sentinel/calibration/`
  (`calibration_splits.py` helper — does NOT touch frozen `splits.py`; `fit_calibrators.py`
  entrypoint, `python -m sentinel.calibration.fit_calibrators`). Report:
  `reports/calibration_results.md`; figure `reports/figures/reliability_before_after.png`.
- Phase-2 surfaces carved from INSIDE the 80% train (holdout spent, never loaded — only
  its patients identified to ASSERT exclusion). Patient-grouped, seed=42:
  S_eval (20% of 80%, FROZEN, fits nothing, looked at once), S_fit (80%) → S_cal (25%) /
  S_train (75%). Pairwise patient-disjoint across {S_train,S_cal,S_eval,holdout}; per-surface
  patient sha256 hashes locked in the manifest.
- OOF on S_fit (5-fold grouped, LOCKED params): AUROC 0.6713 / AUPRC 0.2229 — honest, no
  tripwire (<0.72). Sibling CVs: cal_booster_Sfit 0.6713, cal_booster_Strain 0.6659.
- OOF-vs-prod score GATE **PASS** (KS 0.0157 ≤ 0.05, max decile |Δ| 0.0090 ≤ 0.02) →
  deployment base = @phase1 booster; calibrator fit on S_fit OOF, applied on @phase1 scores.
  (@phase1 is in-sample on S_eval — gate is distributional, recorded verbatim.)
- Selection pre-registered (lowest S_eval ECE; Brier tie-break; reject non-monotone). Both
  reported. **Winner: isotonic.** S_eval ECE **0.3305 → 0.0264**, Brier 0.2039 → 0.0917
  (Platt 0.0272 / 0.0919 — close second). Discrimination unchanged (calibration is monotone).
- class_weight diagnostic (read-only, registers nothing): unweighted LGBM native S_eval ECE
  **0.0076** vs balanced 0.334, CV AUROC 0.6722 (ranking unchanged) — confirms the BALANCED
  weighting, not the features, inflates probabilities; calibration is the principled fix.
- Persisted (committed): `models/calibration/calibrator.joblib` (added a `.gitignore`
  negation — `*.joblib` is globally ignored) + `models/calibration/calibrator_manifest.json`
  (PORTABLE isotonic knots so Phase 4 never unpickles, gate, both methods' S_eval ECE/Brier,
  split hashes, seed, sklearn/lightgbm versions, git sha). Registered
  `sentinel-readmission@phase2-calibrated` (new version) with joblib+manifest as artifacts.
- Phase-4 loader (intended): pyfunc reads the manifest, reconstructs the map from the
  portable form, asserts sklearn version, applies AFTER the booster (and casts category
  dtypes — the existing serving prereq below).

## Current state — PHASE 2, clinical-utility step (W6) DONE

- Read-only utility eval of the SHIPPED calibrated path on FROZEN S_eval. Code:
  `src/sentinel/clinical_utility/` (`calibrated.py` = Phase-4 loader path, `dca.py`,
  `ranking.py`, `evaluate.py` entrypoint `python -m sentinel.clinical_utility.evaluate`).
  Report `reports/clinical_utility_results.md`; figures decision_curve / alert_burden_curve
  / precision_recall_at_k. Trains/registers NOTHING; holdout never loaded.
- `get_calibrated_proba` dogfoods the Phase-4 path: @phase1 booster + isotonic map
  RECONSTRUCTED FROM THE COMMITTED MANIFEST portable knots (np.interp), NOT the joblib.
  Matches committed joblib on S_eval to 0.0e+00 (re-confirms W5 CHECK 2 at use). S_eval
  patient sha256 asserted == W5 manifest, so it is provably the identical surface.
- S_eval: N=15,734, prevalence 0.1145. DCA on calibrated probs (p_t is an odds axis):
  NB_model beats treat-all AND treat-none across the ENTIRE grid p_t∈[0.01,0.50]
  (patient-grouped bootstrap 95% band, B=1000, seed=42).
- Operating points (pre-declared budgets, no snooping): 5% → P 0.408 / R 0.178 / lift 3.56;
  10% → P 0.360 / R 0.314; 20% → P 0.280 / R 0.488. Implied p_t 0.247 / 0.187 / 0.160.
  Honest: ~0.67 AUROC caps recall (top-10% catches 31%, top-20% catches 49%).
- CALIBRATION-INVARIANCE proven (becomes a test): precision@k(p_cal) == precision@k(p_raw)
  to 0.0e+00 — isotonic monotone ⇒ identical worklist. Cal ties broken by raw score.
  "Calibration changes displayed confidence and DCA, NOT who is on the worklist."
- SURFACE-REUSE (documented, not hidden): S_eval selected the calibrator in W5; DCA inherits
  negligible selection optimism (two-way near-tie 0.0264 vs 0.0272); precision@k/recall@k are
  ranking-based ⇒ zero optimism. Reuse-with-documentation, not re-fragmenting.
- Committed source of truth: `models/clinical/operating_points.json` (3 operating points,
  full k-grid curve, DCA grid w/ CI, prevalence, S_eval hash, seed, versions, git sha,
  surface-reuse note). NOT gitignored (`models/*.json` is one-level only). Phase-5 worklist
  UI + alert-burden control read this; mlruns/ not load-bearing.
- BRANCH NOTE: W5 (PR #1) is SQUASH-MERGED to main (`4654e4e`). W6 lives on
  `feat/phase2-clinical-utility` (rebased `--onto main`), pushed, open as PR #2 against main.

## NEXT: PHASE 2 / PHASE 3

- Phase 2 Trust Layer I (calibration + clinical utility) COMPLETE. Cleanest performance
  number still owed from an EXTERNAL set in Phase 8, not any internal surface.
- Phase 3 (explainability/fairness): SHAP — interrogate `discharge_disposition_id`
  (Phase-1 rank-2 by gain) to confirm legit risk signal, not an administrative proxy.

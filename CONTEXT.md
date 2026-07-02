# SENTINEL — LIVING CONTEXT FILE
# Maintained by: strategy/cowork session (Claude Desktop)
# Updated: end of every session. DO NOT edit manually mid-session.
# Companion files: CLAUDE.md (for Claude Code), SESSION_LOG.md (audit trail)

================================================================
PROJECT IDENTITY
================================================================
Name: Sentinel
Type: Clinical 30-day hospital readmission risk decision-support system
Primary user: Care managers (B2B)
Core differentiator: Rigor of integration — honest evaluation, calibration,
  fairness auditing, drift monitoring, and full deployment. NOT algorithmic novelty.
Portfolio role: Flagship project for 2027 MS applied ML/data science applications
  and industry interviews.
Builder: Soumya (solo), stronger in ML/DS than full-stack engineering.
Repo: github.com/SoumyaSinha2603/sentinel
Local: C:\Users\KIIT0001\Projects\sentinel
GitHub user: SoumyaSinha2603

================================================================
WORKFLOW
================================================================
- Strategy, spec generation, phase review: Claude Desktop (Cowork tab) — reads
  this file at session open, updates it at session close.
- Implementation: Claude Code in VS Code — reads CLAUDE.md in repo root.
- Specs flow from Cowork -> Claude Code -> results return to Cowork for review
  before committing.
- Session open ritual: "Read CONTEXT.md and SESSION_LOG.md and tell me what
  state we're in."
- Session close ritual: "Update CONTEXT.md with today's outcomes and append a
  SESSION_LOG.md entry."
- Commit-then-verify: Claude Code confirms origin sync via live git ls-remote
  before declaring a push complete.

================================================================
ENVIRONMENT & STACK
================================================================
Language: Python 3.10.11
OS: Windows / PowerShell (all terminal commands must use PowerShell syntax)
Virtual env: .venv (project-local)
Packaging: src-layout, pip install -e . required in CI (not PYTHONPATH crutch)
Seed: 42 everywhere

ML stack:
  - LightGBM (native categoricals, no target encoding)
  - Optuna (TPE, 60 trials used in Phase 1)
  - MLflow (experiment tracking + model registry, local ./mlruns)
  - DVC (data versioning)
  - SHAP (planned: explanation layer)
  - DiCE (planned: counterfactuals)
  - Evidently 0.7.x (planned: drift monitoring — note 0.4->0.7 API change)

Serving / deployment stack:
  FastAPI, PostgreSQL, Next.js, Tailwind, shadcn/ui, Docker, Render + Vercel

Dev tools:
  VS Code + Claude Code, pre-commit hooks (ruff), GitHub Actions CI (ruff + pytest)

================================================================
DATA
================================================================
Dataset: UCI Diabetes-130 (id=296), fetched via ucimlrepo
Raw file: data/raw/diabetes_130.csv (gitignored, DVC-tracked pointer committed)
Loader: src/sentinel/data/load.py

Raw shape: 101,766 rows x 50 cols, ~30,248 repeat encounters
After cohort eligibility filter: 99,343 rows / 69,990 patients
Prevalence: 11.39% positive (readmitted < 30 days)
Target: readmitted == "<30" -> 1; ">30" and "NO" -> 0

Cohort exclusions applied:
  - Removed death/hospice (discharge_disposition_id in {11,13,14,19,20,21}): 2,423 rows
  - Dropped zero-variance cols: examide, citoglipton

Split strategy:
  - Grouped by patient_nbr (NO row-wise splits — patient leakage prevention)
  - 80% train / 20% holdout via GroupShuffleSplit, seed=42
  - 5-fold CV via StratifiedGroupKFold on the 80% train
  - Holdout is SPENT after Phase 1 — never re-scored, never used for dev decisions
  - All post-Phase-1 evaluation uses OOF / calibration split from inside the 80% train

MIMIC-IV external validation: CUT permanently (no PhysioNet reference as solo applicant).

================================================================
FEATURES (47 total, from src/sentinel/features/build.py)
================================================================
35 categorical + 12 numeric. Deterministic, target-free.

Key transforms:
  - diag_1/2/3 ICD-9 codes -> Strack 9-group mapping (Circulatory, Respiratory,
    Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other;
    V/E/? -> Other)
  - age buckets -> ordinal midpoints
  - medication counts: n_meds_on, n_meds_changed
  - total_prior_visits = outpatient + emergency + inpatient
  - explicit "Missing" categories (weight dropped at 96.9% missing)
  - coded IDs cast as nominal categoricals

LightGBM: native categoricals used. NO target/mean encoding in model features
  (silent leak vector for trees). Target encoding allowed only fold-internal,
  only in leakage audit smell tests.

================================================================
LOCKED DECISIONS (do not re-litigate without explicit sign-off)
================================================================
1. Target: readmitted == "<30" -> 1 (">30" and "NO" -> 0). 30-day product.
2. Honest AUROC ceiling ~0.68 on this dataset. Anything >= ~0.72 = STOP and
   investigate leakage, do not celebrate.
3. Patient-level grouped splits MANDATORY. Row-wise split = leakage.
4. Evaluation harness (src/sentinel/evaluation/splits.py) is FROZEN. Every model
   judged through it identically. Never modify.
5. Holdout is SPENT after Phase 1. Never re-score. Never use for dev decisions.
6. Native LightGBM categoricals. No target encoding in model features.
7. Best params + locked metrics live as constants in lgbm_tuned.py (LOCKED_BEST_PARAMS,
   LOCKED_HOLDOUT_METRICS). Import them; do not hand-retype.
8. Novelty = rigor of integration, NOT a new algorithm.
9. Low-yield work gets cut (DL benchmark and MIMIC-IV already dropped deliberately).
10. Phase 4 loader must use portable manifest knots, NOT joblib (dogfooded in W6).
11. Verified figures only: ~$15,200/readmission, ~$26B/yr Medicare (2018);
    Kansagara 2011 c-stat range 0.55-0.70.
    BANNED: $41.3B figure, ">350 models" claim.
12. Papers reporting AUROC > 0.85 on this dataset almost certainly reflect target leakage.
13. Pre-registration discipline: selection rules, operating points, and evaluation gates
    are declared BEFORE running, not after seeing results.
14. CI requires explicit editable install (pip install -e .). PYTHONPATH crutch only
    works locally.

================================================================
PHASE COMPLETION STATUS
================================================================

PHASE 0 — Foundations & Anti-Leakage [COMPLETE]
  Key outputs:
  - Repo scaffold (src/sentinel/{data,features,models,evaluation,fairness,explain,monitoring})
  - Frozen evaluation harness: splits.py
  - Leakage audit: CLEAN (top single-feature AUROC 0.607, number_inpatient)
  - Cohort builder: 99,343 rows / 69,990 patients / 11.39% prevalence
  - Metrics layer: discrimination (AUROC, AUPRC, Brier) + calibration (ECE)
  - Logistic baseline: AUROC 0.6267, AUPRC 0.1937, Brier 0.2330, ECE 0.3663
  - CI green (run #8, commit a795927)
  - 14 tests passing

PHASE 1 — Feature Engineering + First Real Model [COMPLETE]
  Key outputs:
  - 47 engineered features (build.py), leakage re-audit CLEAN
  - Untuned LightGBM honesty gate passed: CV AUROC 0.672 +/- 0.005
  - Tuned LightGBM (Optuna, 60 TPE trials):
      Best params: num_leaves=41, min_child_samples=68, lr=0.0273,
                   trees=193, reg_alpha=0.00162, reg_lambda=0.0357,
                   class_weight=balanced
  LOCKED HOLDOUT METRICS (recorded once, never recomputed):
      AUROC 0.677 | AUPRC 0.235 | Brier 0.213 | ECE 0.334
  - CV vs holdout gap: -0.001 (holdout fractionally higher) — no overfit, no leakage
  - first_encounter_only sensitivity: AUROC 0.657 (-0.019), benign
  - Model registered: sentinel-readmission@phase1
  - 35 tests passing, CI green (run #11)

PHASE 2 — Trust Layer I: Calibration + Clinical Utility [COMPLETE]
  W5 — Calibration:
  - Patient-grouped OOF calibration workflow executed
  - Isotonic calibrator selected over Platt (lower ECE, monotone)
  - OOF-vs-production KS + decile gate: PASSED
  - Calibrated metrics on S_eval:
      ECE: 0.3305 -> 0.0264 | Brier: 0.2039 -> 0.0917
  - Calibrator persisted: models/calibration/calibrator.joblib
    AND portable manifest: models/calibration/calibrator_manifest.json
    (isotonic knots stored as JSON — Phase 4 pyfunc uses this, NOT joblib)
  - Registered: sentinel-readmission@phase2-calibrated

  W6 — Clinical Utility (DCA):
  - DCA with grouped bootstrap 95% CIs on frozen S_eval surface
  - Model shows weak dominance over treat-all/treat-none across the grid
  - Strict dominance in p_t ≈ 0.05–0.15 band (~+0.035 net benefit over treat-all)
  - Pre-declared operating points committed to models/clinical/operating_points.json:
      5% / 10% / 20% capacity thresholds
  - Phase 4 loader path (portable manifest knots) dogfooded and verified:
      delta to joblib path = 0.0e+00. Phase 4 pyfunc de-risked.

  NOTE on original ECE 0.334: This was a mechanical artifact of class_weight='balanced'
  (unweighted native ECE was 0.0076), NOT true miscalibration of the model's underlying
  scores. This informed the calibration design — the model was already well-ranked;
  calibration fixed the scale.

  DCA framing precision: "beats treat-all everywhere" is overstated. Above prevalence,
  treat-all goes negative so beating it is trivial. Correct framing: weak dominance
  everywhere, strict dominance in the clinically actionable low-threshold band.

================================================================
CURRENT STATE
================================================================
Active phase: PHASE 3 — Fairness Auditing Layer (next up)
Previous phase: Phase 2 COMPLETE
Holdout: SPENT (never touch)
Registry state: sentinel-readmission@phase2-calibrated is current production model

================================================================
PHASE 3 PREVIEW — Fairness Auditing
================================================================
Goal: Audit model performance disaggregated by sensitive attributes. Demonstrate
that Sentinel does not systematically harm subgroups — this is the capability that
separates serious clinical ML from notebook demos.

Attributes to audit (at minimum):
  - race (present in dataset, known missingness)
  - gender
  - age group (already an ordinal feature)
  - payer_code (proxy for socioeconomic status)

Metrics to compute per subgroup:
  - AUROC, AUPRC, ECE (calibrated)
  - Equalized odds components: TPR parity, FPR parity
  - Demographic parity (flag, not necessarily enforce)
  - Net benefit (DCA) per subgroup if sample sizes allow

Key constraints:
  - All evaluation on S_eval (calibration validation set from inside 80% train)
  - Holdout remains SPENT
  - Pre-register fairness criteria and thresholds BEFORE running
  - Report findings honestly — if disparities exist, document them with context,
    do not hide them. The honest documentation IS the differentiator.
  - Bootstrap CIs required for subgroup metrics (small n risk)

Output artifacts:
  - reports/fairness_audit.md (comprehensive, honest)
  - models/fairness/ directory with subgroup metric tables
  - Visualizations: subgroup ROC curves, calibration curves, metric comparison plots

================================================================
PHASE 4 PREREQUISITES (do not solve early — capture only)
================================================================
1. [serving] MLflow JSON serving drops pandas 'category' dtype -> LightGBM categorical
   spec mismatch. Fix: wrap booster in a pyfunc whose predict() casts incoming columns
   to declared categories before the model call.
2. [registry] mlruns/ is gitignored / local file store only. Deployed API needs a
   persistent backend (sqlite local, remote for deploy). Binary reproduces via register.py.
3. [loader] Phase 4 pyfunc MUST use calibrator_manifest.json knots (portable),
   NOT calibrator.joblib. Already dogfooded and verified in Phase 2 (delta = 0.0e+00).

================================================================
KEY ARTIFACTS (path reference)
================================================================
src/sentinel/
  data/load.py                         ← data loader
  data/cohort.py                       ← cohort builder
  data/leakage_checks.py               ← leakage audit
  features/build.py                    ← feature engineering (47 features)
  evaluation/splits.py                 ← FROZEN evaluation harness
  evaluation/metrics.py                ← metrics layer (reused by all models)
  models/lgbm_tuned.py                 ← tuned LightGBM + locked constants
  models/baseline.py                   ← logistic baseline
  models/register.py                   ← MLflow model registry

models/
  calibration/calibrator.joblib        ← calibrator (local use only)
  calibration/calibrator_manifest.json ← PORTABLE calibrator (Phase 4 uses this)
  clinical/operating_points.json       ← pre-declared operating points (5/10/20%)

reports/
  leakage_audit.md
  leakage_audit_features.md
  cohort_summary.md
  baseline_results.md
  lgbm_baseline_results.md
  lgbm_tuned_results.md
  model_registry.md
  fairness_audit.md                    ← to be created in Phase 3

================================================================
OPEN ITEMS / LOOSE ENDS
================================================================
1. competitive_landscape.md (docs/): seeded but NOT finished. Close before Phase 4.
   Calibration + fairness auditing are exactly the capabilities this doc needs to argue
   Sentinel has and median projects don't.
2. Agentic evaluation layer (strategic question open since Phase 0): still deferred.
   Revisit only after core product (Phases 3 + 4) is complete.

================================================================
RIGOR PRINCIPLES (apply in every phase)
================================================================
- Honest evaluation is the differentiator. Credibility comes from rigorous integration,
  not impressive numbers.
- Pre-registration discipline: criteria declared before running, never after.
- Holdout discipline: SPENT after Phase 1. Non-negotiable.
- Leakage tripwire: AUROC >= ~0.72 on this dataset = STOP and investigate.
- CI requires pip install -e . (not PYTHONPATH crutch).
- Prose over heavy formatting in strategy responses.
- New chat instances for new phases — handoff files and kickoff prompts generated
  at phase boundaries. Claude Code does not carry strategic context automatically.
- All terminal commands: PowerShell syntax (Windows environment).
- Verified figures only (see locked decisions #11).
- Rigor interventions are expected and welcomed. Brutally honest guidance preferred.

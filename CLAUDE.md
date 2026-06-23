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
- NEXT: Week-2 leakage audit — the locked grouped-by-`patient_nbr` evaluation harness and
  the leakage-check script (`src/sentinel/data/leakage_checks.py`, currently a stub).

## How to work in this repo

- Default to small, reviewable changes. Propose a plan, get sign-off, then implement.
- When unsure about a clinical/ML modeling choice, prefer the honest/conservative option
  and surface the tradeoff rather than optimizing a metric.

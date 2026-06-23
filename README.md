# Sentinel

Explainable, fairness-audited, calibrated clinical **readmission-risk decision support**.

Sentinel gives hospital care teams a ranked patient worklist with SHAP-based reason
codes, counterfactual intervention suggestions, subgroup fairness monitoring, and
drift detection. It is a B2B decision-support tool, **not** a diagnostic device.

## Why this exists

30-day hospital readmissions are costly and partly preventable (~\$15,200 per
readmission; ~\$26B/year for Medicare, 2018 figures). The problem is crowded with
models, but most are poorly calibrated, unaudited for bias, and never deployed or
monitored. Sentinel's contribution is **rigor of integration**, not a new algorithm:
honest evaluation, calibration, conformal uncertainty, and continuous fairness and
drift monitoring shipped as live product features.

## Honest performance expectation

On the UCI Diabetes-130 dataset the realistic benchmark is **AUROC ~0.68**. Anything
above ~0.85 on this dataset is a data-leakage bug, not a result. A Week-2 leakage
audit is a hard gate before any modeling.

## Tech stack

- **ML:** LightGBM / XGBoost, SHAP, DiCE
- **Serving:** FastAPI, PostgreSQL
- **Frontend:** Next.js, Tailwind, shadcn/ui
- **MLOps:** MLflow (tracking), DVC (data versioning), Evidently (drift)
- **Infra:** Docker, deployed to Render (backend) + Vercel (frontend)

## Repository layout

```
src/sentinel/      core library (data, models, evaluation, fairness, explain, monitoring)
api/               FastAPI service
frontend/          Next.js app
data/              DVC-tracked datasets (not in git)
notebooks/         exploratory analysis
tests/             test suite
docs/              design notes, competitive landscape
reports/           generated figures and results
```

## Status

Phase 0 — foundations and anti-leakage. See `docs/` for the project roadmap.

## Data & ethics

Primary dataset: UCI Diabetes-130 (public). Built for research and portfolio
purposes; not validated for clinical use.

## License

MIT — see `LICENSE`.

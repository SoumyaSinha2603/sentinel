# SENTINEL — SESSION LOG
# Append-only. One entry per session. Never delete entries.
# Format: ## [YYYY-MM-DD] | Phase | What was decided or built

---

## [Phase 0 Summary] | Phase 0 — Foundations & Anti-Leakage | COMPLETE
Built: repo scaffold, frozen evaluation harness (splits.py), leakage audit (CLEAN),
cohort builder (99,343 rows / 69,990 patients / 11.39% prevalence), metrics layer,
logistic baseline (AUROC 0.6267). 14 tests passing. CI green (run #8, commit a795927).
Key decision: MIMIC-IV external validation cut permanently (no PhysioNet reference).

---

## [Phase 1 Summary] | Phase 1 — Feature Engineering + First Real Model | COMPLETE
Built: 47 engineered features (build.py), leakage re-audit (CLEAN), untuned LightGBM
honesty gate (CV AUROC 0.672), tuned LightGBM via Optuna (60 TPE trials).
LOCKED HOLDOUT: AUROC 0.677 | AUPRC 0.235 | Brier 0.213 | ECE 0.334.
CV-holdout gap: -0.001. Holdout SPENT. Model registered: sentinel-readmission@phase1.
35 tests passing. CI green (run #11).
Key cut: DL benchmark (FT-Transformer/TabNet) dropped — low yield vs ~0.01 headroom.

---

## [Phase 2 Summary] | Phase 2 — Trust Layer I: Calibration + Clinical Utility | COMPLETE
W5 — Calibration: isotonic calibrator selected (lower ECE than Platt, monotone).
OOF-vs-production KS + decile gate passed. Calibrated metrics on S_eval:
ECE 0.3305->0.0264, Brier 0.2039->0.0917. Calibrator persisted as both joblib and
portable manifest (calibrator_manifest.json). Registered: sentinel-readmission@phase2-calibrated.
W6 — DCA: weak model dominance over treat-all/treat-none across grid; strict dominance
in p_t ≈ 0.05-0.15 band (~+0.035 net benefit). Operating points committed to
models/clinical/operating_points.json (5/10/20% capacity). Phase 4 pyfunc loader
dogfooded via manifest knots, delta = 0.0e+00. Phase 4 de-risked.

---

## [Workflow Migration] | Meta | Migrated to Claude Desktop Cowork workflow
Moved strategy sessions from claude.ai chat to Claude Desktop Cowork tab.
Created CONTEXT.md (living project state) and SESSION_LOG.md (audit trail).
CONTEXT.md replaces per-phase handoff .txt files as single source of truth for
strategy sessions. CLAUDE.md in repo root remains source of truth for Claude Code.
Session open ritual: "Read CONTEXT.md and SESSION_LOG.md and tell me what state we're in."
Session close ritual: "Update CONTEXT.md with today's outcomes and append a SESSION_LOG.md entry."

---

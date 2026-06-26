# Competitive Landscape & Novelty Positioning

> Status: Sections 2–4 + comparison table drafted from shipped artifacts. Sections 1 and 5
> remain [AUTHOR: ...] stubs. External cells marked [author: verify] need a source check.

## 1. The problem space

[AUTHOR: one short paragraph — what 30-day readmission prediction is and why it matters
clinically + financially. Use the verified cost/volume figures from the citations block.]

## 2. Where readmission models actually sit (field reality)

Thirty-day readmission prediction is a crowded, mature problem, and the honest summary of
the field is that most models are only modestly discriminative. The canonical review,
Kansagara et al. (2011), examined 26 distinct models and found c-statistics mostly in the
0.55–0.70 range, with only six exceeding 0.70 [2]. More than a decade later the picture has
not materially changed: a 2024 systematic review and meta-analysis reports a pooled AUC of
roughly 0.71, while flagging that a large share of the underlying models carry a high risk
of bias [3]. Discrimination, in other words, has a low ceiling on routinely available data,
and incremental AUROC is not where the problem is won or lost.

This field reality also sets a leakage tripwire. On the UCI Diabetes 130-US hospitals
dataset specifically [4], a realistic, leakage-free model lands near AUROC ~0.68; Sentinel's
own tuned model reaches **0.677** on a sealed, patient-grouped holdout. Reported AUROCs of
0.85+ — and the 0.95+ figures common in public notebooks — on this dataset almost always
indicate target leakage or a row-wise (rather than patient-grouped) split, not a genuinely
better predictor. We treat any such number as a defect to investigate, not a result to
celebrate.

### Comparison of representative approaches

External cells are characterizations of the cited literature/segment; cells tagged
[author: verify] still need a direct source check. Sentinel cells distinguish **shipped**
(with artifact) from **planned** (roadmap, not yet built) — we do not credit unbuilt work.

| Approach | Typical AUROC/c-stat | Calibration reported? | Fairness audit? | Monitoring/drift? | Deployed product? | Clinician validation? |
|---|---|---|---|---|---|---|
| Published academic models (Kansagara-era) | 0.55–0.70; six >0.70 [2] | Rarely [author: verify] | Rarely [author: verify] | No | Rarely | Rarely [author: verify] |
| 2024 pooled meta-analysis | pooled AUC ~0.71; many high-bias [3] | Inconsistent [3] | Rarely assessed [author: verify] | n/a (review) | n/a (review) | n/a (review) |
| Typical public/student GitHub project | Frequently 0.9+ reported (often leakage) [author: verify] | No (typical) | No (typical) | No | No | No |
| Vendor products (claimed) | Vendor-claimed, not independently verified | Claimed, unverified | Claimed, unverified | Claimed, unverified | Yes (commercial) | Claimed, unverified |
| Sentinel | 0.677 sealed holdout [4] | **Shipped** — ECE 0.33→0.026 (isotonic) + DCA | Planned (Phase 3) | Planned (Evidently 0.7) | Partial — model registered; API Phase 4 | No — deferred (Phase 8) |

## 3. The gap the literature itself names

The recurring critique of this field is not that the predictors are too weak — it is that
the surrounding rigor is missing. Calibration is inconsistently reported [3]; subgroup
fairness is rarely audited; prospective monitoring and drift detection are seldom shipped as
live features rather than one-off plots; and clinician validation of the actual decision
artifact — the worklist and its reason codes — is the exception, not the rule. Reviews of
the area repeatedly call for better-calibrated, less-biased, externally validated models
[2][3], yet these elements are almost never delivered *together* in one maintained system.
That is the integration gap: each ingredient exists somewhere, but the assembled,
honestly-evaluated decision-support product does not.

## 4. Sentinel's position

Sentinel does not claim a better predictor. Its discrimination is deliberately honest —
**AUROC 0.677** on a sealed, patient-grouped holdout, squarely at the field's ~0.68 ceiling
[2][3][4]. The contribution is the integration the literature keeps asking for, built and
evidenced rather than asserted. Three layers are already shipped with artifacts:

- **Calibration as a first-class output.** Isotonic calibration cuts expected calibration
  error from **0.3305 to 0.0264** and Brier from **0.2039 to 0.0917** on a frozen evaluation
  surface, selected under a pre-registered rule and persisted as a portable, version-pinned
  artifact (`reports/calibration_results.md`, `models/calibration/calibrator_manifest.json`).
- **Clinical-utility validation, not just metrics.** Decision-curve analysis shows the
  calibrated model weakly dominates treat-all/treat-none across the grid and strictly
  dominates in the meaningful band (p_t ≈ 0.05–0.15, ~**+0.035** net benefit over treat-all),
  with operating points at fixed 5/10/20% alert budgets and a verified calibration-invariance
  result — calibration changes displayed confidence and net benefit, not who is on the
  worklist (`reports/clinical_utility_results.md`, `models/clinical/operating_points.json`).
- **Honest evaluation that refuses leakage-driven numbers.** A locked, patient-grouped
  harness, a Week-2 leakage audit (top single-feature AUROC 0.607, nothing >0.70), and an
  explicit tripwire treating >0.72 as a defect keep every reported figure trustworthy.

The remaining differentiators from the original pitch — conformal/uncertainty quantification
on individual predictions, continuous fairness **and** drift monitoring as live features, and
clinician validation of reason codes and worklist utility — are on the roadmap (Phases 3–8),
held to the same evidence standard and **not** claimed as done here. The thesis is restraint
plus completeness: a calibrated, utility-validated, leakage-disciplined decision-support
product, backed by committed artifacts rather than promises, in a field that rarely ships the
whole package.

## 5. Honest limitations

[AUTHOR: single public dataset, modest discrimination caps recall, external validation
deferred to Phase 8, counterfactuals heuristic without clinician sign-off. State plainly.]

## References (verified figures only)

1. AHRQ HCUP Statistical Briefs (readmissions) — ~$15,200 average cost per 30-day
   readmission (2018); ~3.8M 30-day all-cause adult readmissions in the US, ~14% rate
   (2018); readmissions cost Medicare ~$26B/yr (2018), ~$52B all-payer (some sources).
   hcup-us.ahrq.gov/reports/statbriefs. [author: complete cite — exact statbrief number(s) + date]
2. Kansagara D, et al. JAMA 2011 — reviewed 26 models; c-statistics mostly 0.55–0.70,
   only six above 0.70. pubmed.ncbi.nlm.nih.gov/22009101. [author: complete cite — full title, volume, pages]
3. 2024 systematic review / meta-analysis — pooled AUC ~0.71; many models at high risk of
   bias. Int. J. Nursing Studies (sciencedirect.com). [author: complete cite — authors, title, DOI]
4. UCI Diabetes 130-US hospitals dataset (1999–2008). archive.ics.uci.edu.
   [author: complete cite — Strack et al. 2014]

# Competitive Landscape & Novelty Positioning

> Status: SCAFFOLD. Argument prose to be written by author. Stubs marked [AUTHOR: ...].

## 1. The problem space

[AUTHOR: one short paragraph — what 30-day readmission prediction is and why it matters
clinically + financially. Use the verified cost/volume figures from the citations block.]

## 2. Where readmission models actually sit (field reality)

[AUTHOR: argue the field is crowded with modest, often-biased predictors. Anchor on
Kansagara c-stats 0.55–0.70 and the 2024 pooled AUC ~0.71 / high-bias finding. Make the
point that 0.95+ AUROC reports on this dataset almost always indicate leakage.]

### Comparison of representative approaches

[AUTHOR: fill every "[AUTHOR]" cell; row labels are pre-filled. Keep claims traceable to
the references block; mark vendor figures "vendor-claimed, not independently verified".]

| Approach | Typical AUROC/c-stat | Calibration reported? | Fairness audit? | Monitoring/drift? | Deployed product? | Clinician validation? |
|---|---|---|---|---|---|---|
| Published academic models (Kansagara-era) | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] |
| 2024 pooled meta-analysis | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] |
| Typical public/student GitHub project | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] |
| Vendor products (claimed) | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] |
| Sentinel | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] |

## 3. The gap the literature itself names

[AUTHOR: argue that calibration, fairness auditing, monitoring, and clinician validation
are repeatedly called for and rarely shipped TOGETHER. This is the integration gap.]

## 4. Sentinel's position

[AUTHOR: the thesis. NOT a better predictor — honestly ~0.677 — but the integrated,
calibrated, utility-validated decision-support product the field asks for. Cite your OWN
shipped evidence: ECE 0.33→0.026, DCA net-benefit band, calibration-invariance. The
differentiator is restraint + completeness, backed by artifacts, not promises.]

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

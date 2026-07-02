# Phase 3 — Age-disparity robustness check (OOF): spec for Claude Code

> Appendix to the pre-registered fairness audit. **Does NOT replace the pre-registered
> `S_eval` audit** — it is the "investigate before believing" follow-through that prereg
> §9 explicitly demands for any subgroup AUROC outside the honest ~0.62–0.68 band. The
> `S_eval` audit stands; this adds an out-of-fold robustness section that tells us how much
> of the age gap is real vs an in-sample-optimism artifact.
>
> Branch: `feat/phase3-fairness`. Read-only. Holdout stays SPENT. seed = 42. ruff + pytest green.

## Why this exists (the problem being tested)

The `S_eval` fairness audit scored `@phase1` on rows it was **trained on** (`S_eval` ⊂ the
80% train that `@phase1` was refit on). So every subgroup AUROC is in-sample-inflated
(bands read 0.70–0.84 vs honest holdout 0.677 / OOF 0.671). The headline age finding —
80+ AUROC ~0.70 vs <40 ~0.84, Δ up to −0.13 — is therefore suspect on two counts:

1. **Differential optimism.** In-sample memorization is not uniform across subgroups; a
   small band can be overfit harder, manufacturing an AUROC gap that is an evaluation
   artifact, not a real-world reliability gap.
2. **Fragile anchor.** The <40 band is the smallest powered group (~1,130 rows / 164
   positives) and pools the anomalous [20-30) bucket (prevalence 0.227). Its 0.84 is the
   least trustworthy number in the table, and it is exactly what the headline Δ leans on.

The clean test: recompute the age-band discrimination gap on the **S_fit out-of-fold**
scores — genuinely out-of-fold (each row scored by a model not trained on it), ~64k rows
so every band is well-powered, and disjoint from both `S_eval` and the spent holdout.

## Surface and scoring (reuse existing code, don't reinvent)

- Recover **S_fit** the same way W5 did: `make_holdout_split(df)` → 80% train →
  `make_calibration_splits(df_train)` → `indices["S_fit"]` (= `S_train ∪ S_cal`, disjoint
  from `S_eval` by construction, disjoint from holdout as a subset of the 80% train).
- Row-aligned frames, same pattern the audit already uses: score the **built** feature
  frame; slice subgroup labels from the **cohort** frame (raw `age` string survives there;
  `build_features` collapses it to `age_midpoint`). Assert row alignment.
- **OOF raw scores:** reuse `sentinel.calibration.fit_calibrators._oof_scores(df_sub,
  x_sub, y_sub, factory=lgbm_tuned._model, n_folds=N_CAL_FOLDS)`. This is the *identical*
  function and locked booster W5 used — it must reproduce pooled OOF AUROC **0.6713** on
  S_fit (assert within ~1e-6 as a provenance check). These are the honest, out-of-fold
  scores.
- **Calibrated OOF (for ECE only):** apply the committed portable isotonic map to the OOF
  raw scores via `clinical_utility.calibrated.apply_calibrator_portable(manifest, p_oof_raw)`.
  Caveat to record verbatim in the report: the isotonic map was *fit* on these S_fit OOF
  scores in W5, so per-band ECE here is mildly optimistic *for the calibrator* (a low-
  capacity monotone map — small effect). AUROC is calibration-invariant, so it is
  unaffected. Discrimination is the primary deliverable of this check; ECE is secondary.

## What to compute

For the **age** attribute only (this is a targeted robustness check, not a re-audit of all
four attributes), on S_fit OOF:

- Per age band (`<40`, `40-60`, `60-80`, `80+`): `subgroup_metrics` — AUROC, AUPRC, ECE
  (on calibrated OOF), n, positives, prevalence — reusing `fairness.metrics`. All bands
  should clear the powered rule on S_fit; assert and report it.
- Patient-grouped bootstrap 95% CIs (2,000 resamples, seed 42) via `fairness.bootstrap`,
  on the S_fit OOF surface.
- Pairwise ΔAUROC with paired-resample CIs for the clinically meaningful pairs, anchored on
  the **well-powered** comparisons — **80+ vs 60-80** and **80+ vs 40-60** as the primary
  contrasts. Report 80+ vs <40 too, but explicitly label <40 as the fragile anchor.
- **Extra diagnostic:** report the [20-30) decade sub-bucket on its own (n / positives /
  OOF AUROC with CI) so the reader can see how much the <40 band is driven by that
  anomalous, high-prevalence slice.

## The comparison that answers the question

Produce a side-by-side table, per age band: **S_eval (in-sample) AUROC** vs **S_fit (OOF)
AUROC**, plus the per-band optimism gap (in-sample − OOF). This makes differential optimism
visible directly. Then judge:

- **Gap survives** — if 80+ still discriminates materially worse than the powered mid bands
  out-of-fold (pairwise ΔAUROC CI excludes 0 **and** |Δ| > 0.05, prereg §6): the age
  finding is **real**. Keep it as the headline, re-anchored on 80+ vs mid bands, with the
  case-mix mechanism noted (elderly risk distributions are compressed → ranking is
  intrinsically harder; check whether per-band ECE is also worse or whether 80+ is merely
  less *rank-separable* but still well-calibrated).
- **Gap collapses** — if out-of-fold the bands converge (Δ shrinks below threshold or CIs
  cover 0): the S_eval age disparity was **substantially an in-sample artifact**, and the
  differential optimism (esp. an inflated small <40 band) explains the headline. Reframe
  the report accordingly — that reframing is itself the rigor result.

Either outcome is a legitimate, reportable finding. Do not tune toward either.

## Guardrails

- Read-only. Trains nothing persistent (the OOF CV fits are transient, in-memory, exactly
  as W5 did). Registers nothing. Holdout never loaded for scoring.
- No new splitting logic; reuse frozen primitives. Assert S_fit ∩ S_eval = ∅ and
  S_fit ∩ holdout = ∅.
- seed = 42; 2,000 bootstrap resamples; same powered rule and practical thresholds as the
  prereg (ΔAUROC > 0.05, ΔECE > 0.03).
- This is a robustness *appendix*; it must not silently mutate the pre-registered S_eval
  verdicts. The S_eval numbers and verdicts stay in the report as-run; this adds a clearly
  labeled "OOF robustness (age)" section and, if warranted, a revised *interpretation* of
  the age headline — with the original S_eval result still shown.

## Outputs

- Extend `reports/fairness_audit.md` with an **"Age disparity — OOF robustness"** section:
  the S_eval-vs-S_fit side-by-side table, pairwise ΔAUROC (OOF) with CIs, the [20-30)
  diagnostic, the survives/collapses verdict, and the calibrator-in-sample caveat.
- `models/fairness/age_oof_robustness.json` — per-band S_fit OOF metrics + CIs, the
  in-sample-vs-OOF gaps, pairwise diffs, provenance (S_fit patient sha256, reproduced OOF
  AUROC, seed, n_boot, versions, git sha).
- One figure: per-age-band AUROC with CI error bars, S_eval vs S_fit overlaid.

## Tests

- S_fit recovery reproduces W5 OOF AUROC 0.6713 (provenance assert).
- S_fit disjoint from S_eval and from holdout (structural assert).
- Determinism: same seed → identical OOF scores and identical CIs.
- The [20-30) diagnostic is computed on the raw cohort `age` string, not `age_midpoint`.

---

### Kickoff line for Claude Code

> Read `docs/phase3_oof_robustness_spec.md`. On `feat/phase3-fairness`, add a read-only
> OOF robustness check for the age disparity: recover S_fit, reproduce W5 OOF scores via
> `fit_calibrators._oof_scores` with `lgbm_tuned._model` (assert OOF AUROC 0.6713), compute
> per-age-band AUROC/ECE with patient-grouped bootstrap CIs, produce the S_eval-vs-S_fit
> side-by-side and the 80+ vs mid-band pairwise ΔAUROC, plus the [20-30) diagnostic. Judge
> survives/collapses per prereg §6 thresholds. Extend `reports/fairness_audit.md`, emit
> `models/fairness/age_oof_robustness.json` + one figure, add tests, keep ruff + pytest
> green. Holdout stays SPENT. Show the plan before committing.

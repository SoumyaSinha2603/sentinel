# Fairness Audit — Phase 3 PRE-REGISTRATION

> **Status: PRE-REGISTRATION. Written and frozen BEFORE any subgroup score is computed.**
> This document declares the sensitive attributes, subgroup definitions, metrics,
> statistical method, numeric decision thresholds, and analyst degrees of freedom for
> the Phase 3 fairness audit. Once committed, it is not edited in response to results.
> Any deviation forced by reality is recorded as a dated amendment at the bottom, never
> a silent rewrite. (Pre-registration discipline — CONTEXT.md locked decision #13.)

Provenance of the grounding numbers below: reproduced from `data/raw/diabetes_130.csv`
→ `build_cohort` → `make_holdout_split` (80% train) → `make_calibration_splits` →
`S_eval`, seed=42, sklearn 1.7.2. The reconstructed `S_eval` is **15,734 rows /
11,199 patients / prevalence 0.1145 / 1,802 positives**, matching the patient sha256
`71a7a8b69e3b` recorded in `reports/calibration_results.md`. The surface is therefore
the same frozen `S_eval`, not a re-draw.

---

## 0. Blocking prerequisites (resolve before running the audit)

These are recorded here because the audit cannot run reproducibly until they are fixed.
They are **not** part of the pre-registered analysis; they are gating engineering facts.

1. **Operating points are not on the working branch.** `models/clinical/operating_points.json`
   and the entire `src/sentinel/clinical_utility/` source exist only on the unmerged
   branch `feat/phase2-clinical-utility` (verified via `git ls-tree`). The currently
   checked-out branch (`docs/competitive-landscape-scaffold`) and `main` do **not** contain
   them, despite CONTEXT.md/SESSION_LOG.md asserting Phase 2 W6 is "COMPLETE" with the file
   "committed". The TPR/FPR-parity section below evaluates parity **at the pre-declared
   operating points**; those thresholds must be present on the branch the audit runs from.
   → Merge/rebase `feat/phase2-clinical-utility` (or cherry-pick `operating_points.json`)
   before scoring. Until then, treat the operating-point thresholds as unavailable.
2. **Dirty working tree.** ~30 files are modified-uncommitted on the current branch. Start
   Phase 3 from a clean, named branch (`feat/phase3-fairness`) off a base that actually
   contains the Phase 2 artifacts, so the audit is reproducible and attributable.

*(This pre-registration itself is valid regardless of the above — the criteria do not
depend on results. But the run is blocked until #1 is resolved.)*

---

## 1. Purpose and scope

Audit whether the calibrated production model
(`sentinel-readmission@phase2-calibrated`) performs **equitably across sensitive
subgroups**, and document any disparities honestly. This is the capability that
separates serious clinical ML from a notebook demo, and the honest documentation of
disparities — not their absence — is the portfolio differentiator.

**This phase measures and documents. It does not mitigate.** No thresholding,
reweighting, or post-processing is applied in Phase 3. If a material disparity is found,
mitigation is a *conditional, separately pre-registered* follow-up, never an ad-hoc
in-phase reaction.

Scope guards inherited from locked decisions:

- All scoring on the **frozen `S_eval`** surface. The Phase-1 20% holdout stays **SPENT**
  — never loaded, never scored (locked decision #5).
- Model under audit is the calibrated model; scores are post-isotonic-calibration
  probabilities via the portable manifest knots.
- seed = 42 everywhere.

---

## 2. Sensitive attributes and subgroup definitions

Four attributes, defined on the raw cohort columns. Subgroup collapsing is declared
**now**, from the real `S_eval` marginal counts, so it cannot be reverse-engineered from
results. Counts and per-subgroup positive counts are the reproduced `S_eval` values.

### 2.1 `race` — PRIMARY

| subgroup | n | positives | subgroup prev | tier |
|---|---:|---:|---:|---|
| Caucasian | 11,709 | 1,348 | 0.115 | full metrics |
| AfricanAmerican | 3,066 | 370 | 0.121 | full metrics |
| Hispanic | 316 | 27 | 0.085 | descriptive-only |
| Other | 244 | 29 | 0.119 | descriptive-only |
| Missing (NaN) | 312 | 22 | 0.071 | descriptive-only |
| Asian | 87 | 6 | 0.069 | descriptive-only |

Decision: **do not merge** the small race groups into a synthetic "non-white" bucket —
that would hide heterogeneity and is scientifically dishonest. Report each as-is.
`Missing` is kept as its own group (missingness in `race` can itself be biased and is
worth surfacing). Consequence to state plainly: race parity is effectively a **two-group
comparison (Caucasian vs AfricanAmerican)** with well-powered estimates; the other four
groups are reported descriptively with wide CIs and an explicit UNDERPOWERED label.

### 2.2 `gender` — PRIMARY

| subgroup | n | positives | tier |
|---|---:|---:|---|
| Female | 8,442 | 977 | full metrics |
| Male | 7,291 | 825 | full metrics |
| Unknown/Invalid | 1 | 0 | **dropped** (n=1) |

Decision: drop `Unknown/Invalid` (single row) from parity computation; report that it
was dropped and why.

### 2.3 `age` — PRIMARY (collapsed)

Raw `age` is ten decade buckets; the extremes are underpowered ([0-10) has 31 rows / 0
positives; [10-20) has 106 / 6). Pre-declared collapse into four clinically sensible
bands (bands fixed **before** seeing per-band model performance):

| band | source buckets | n | positives | tier |
|---|---|---:|---:|---|
| < 40 | [0-10),[10-20),[20-30),[30-40) | 1,130 | 164 | full metrics |
| 40–60 | [40-50),[50-60) | 4,169 | 417 | full metrics |
| 60–80 | [60-70),[70-80) | 7,481 | 868 | full metrics |
| 80+ | [80-90),[90-100) | 2,954 | 353 | full metrics |

Note for transparency: the `< 40` band pools the anomalous [20-30) bucket (prev 0.227,
well above base rate). This is flagged for interpretation, not excluded.

### 2.4 `payer_code` — SECONDARY / EXPLORATORY

`payer_code` is 39% missing on `S_eval` (6,198 rows) and otherwise fragmented across 16
codes, most with < 300 rows and several with 0 positives. It is a **noisy proxy** for
socioeconomic status, not a clean sensitive attribute. Pre-declared collapse:

| group | source codes | approx n | tier |
|---|---|---:|---|
| Medicare | MC | 4,988 | full metrics |
| Medicaid | MD | 555 | descriptive-only |
| Private | BC, HM, CP, CM, OG, DM, CH | ~3,105 | full metrics (pooled) |
| Self-pay | SP | 709 | descriptive-only |
| Missing | NaN | 6,198 | full metrics |
| Other/Unknown | UN, PO, WC, OT, MP, SI | ~558 | descriptive-only |

Decision: `payer_code` findings are reported as **exploratory** and never used to fail
the model on their own — the missingness and fragmentation make strong parity claims
unwarranted. The pooling above is a pre-registered convenience, and its arbitrariness is
disclosed.

---

## 3. Minimum-power inclusion rule (pre-registered)

A subgroup receives **full quantitative metrics** (AUROC, AUPRC, calibrated ECE,
TPR/FPR at operating points, and pairwise parity tests) only if it has, on `S_eval`:

- **≥ 100 total rows**, AND
- **≥ 30 positive (`<30`) cases.**

Subgroups below either bound are **descriptive-only**: reported with counts and point
estimates carrying explicit bootstrap CIs, labeled **UNDERPOWERED**, and **excluded from
pass/fail parity judgments**. Rationale: below ~30 positives, subgroup AUROC and TPR
estimates have CIs too wide to support any parity claim; forcing a verdict there would be
false precision. This threshold is a project choice, declared before scoring.

Groups that qualify for full metrics under this rule: race {Caucasian, AfricanAmerican};
gender {Female, Male}; age {all four collapsed bands}; payer {Medicare, Private, Missing}.

---

## 4. Metrics computed per qualifying subgroup

**Discrimination:** AUROC, AUPRC (report alongside subgroup prevalence, since AUPRC
scales with base rate — cross-subgroup AUPRC differences are largely prevalence artifacts
and will be interpreted as such, not as bias).

**Calibration:** calibrated ECE (same quantile-binned estimator as
`evaluation/metrics.py`) plus a reliability curve per subgroup. Calibration parity is a
first-class concern: a model can rank equally well across groups yet be systematically
over/under-confident for one.

**Error-rate parity (equalized-odds components; Hardt, Price & Srebro, 2016):** TPR
(sensitivity) and FPR per subgroup, evaluated at fixed decision thresholds — the
**pre-declared operating points** (5% / 10% / 20% capacity) from
`models/clinical/operating_points.json`. Parity is assessed by pairwise differences
between qualifying subgroups within each attribute. The 10% capacity point is the
**primary** operating point for the headline parity verdict; 5% and 20% are reported for
sensitivity.

**Demographic parity:** flagged and reported (selection rate per subgroup at each
operating point) but **not** treated as a fairness requirement — with differing subgroup
base rates, enforcing equal selection rates would degrade calibrated risk estimates.
Reported for completeness and explicitly not used as a pass/fail gate.

**Per-subgroup net benefit (DCA):** computed only for the two best-powered attributes
(race: Caucasian vs AfricanAmerican; gender) if the qualifying subgroup sizes support
stable bootstrap CIs; otherwise omitted and the omission noted.

---

## 5. Statistical method (pre-registered)

- **Patient-grouped bootstrap.** Resample **patients** (by `patient_nbr`), not rows, so
  repeat encounters do not inflate effective sample size — consistent with the project's
  grouped-split discipline. **2,000** bootstrap resamples, seed = 42. Report **95%
  percentile CIs** for every subgroup metric and for every pairwise difference.
- **Disparity is judged on the difference, with two conditions.** A disparity between two
  qualifying subgroups is called **material** only if **both** hold:
  1. **Statistical:** the 95% bootstrap CI of the pairwise difference **excludes 0**.
  2. **Practical:** the point difference exceeds the pre-declared practical threshold for
     that metric (Section 6).
  Meeting only the statistical condition → **FLAG (monitor)**, not material. This guards
  against calling a trivially small but tight difference a fairness failure.
- **No multiplicity fishing.** The comparisons are pre-enumerated in Sections 2–4. No
  post-hoc subgroup discovery. If an unplanned subgroup looks interesting, it is logged as
  a hypothesis for a future audit, not scored into this verdict.

---

## 6. Numeric decision thresholds (pre-registered)

Practical-significance thresholds per metric. These are project choices with stated
rationale, frozen before scoring.

| metric | practical threshold for "material" | rationale |
|---|---|---|
| ΔAUROC (pairwise) | > 0.05 absolute | below ~0.05, AUROC gaps are within this dataset's own CV noise band (±0.005–0.02 seen in Phases 1–2); 0.05 is a conservative, visible gap |
| ΔTPR (pairwise, at operating point) | > 0.10 absolute | equalized-odds sensitivity gap; 0.10 is a clinically noticeable difference in who gets flagged |
| ΔFPR (pairwise, at operating point) | > 0.10 absolute | symmetric to TPR |
| TPR/FPR ratio (four-fifths screen) | outside [0.80, 1.25] | EEOC four-fifths rule as a secondary screen, reported alongside the absolute-difference test |
| ECE (per subgroup) | > 0.05 absolute | matches the calibration bar already cleared in aggregate (aggregate calibrated ECE was 0.0264); any subgroup > 0.05 means calibration didn't hold there |
| ΔECE (pairwise) | > 0.03 absolute | cross-subgroup calibration spread |

The **four-fifths (80%) rule** is used only as a secondary, reported screen — not the
primary verdict — because at low selection rates ratio tests are unstable.

---

## 7. Verdict rules (pre-registered)

Per attribute, per metric family:

- **PASS** — no qualifying pairwise comparison is *material* (Section 5 definition).
- **FLAG (monitor)** — a statistically-significant-but-sub-threshold difference exists,
  or a material difference exists **only** in an underpowered/descriptive group.
- **MATERIAL DISPARITY** — at least one qualifying pairwise comparison is material.

A MATERIAL DISPARITY verdict does **not** fail the project. It triggers: (a) honest
documentation with clinical context and plausible mechanism (e.g., differential
missingness, base-rate differences, feature availability), and (b) a decision — recorded,
not automatic — on whether a separately pre-registered mitigation phase is warranted. The
audit's credibility comes from reporting whatever it finds, including uncomfortable
findings, with context.

---

## 8. What this audit will NOT do

- Will not touch or re-score the spent holdout.
- Will not apply any mitigation, per-group threshold, or reweighting in Phase 3.
- Will not merge small subgroups to manufacture power or to smooth a disparity away.
- Will not optimize any threshold to improve a parity number.
- Will not compute equalized-odds thresholds that require access to the sensitive
  attribute at inference time (the deployed model must not consume `race`/`gender` as a
  decision input); parity is measured, not enforced via group-conditional thresholds.
- Will not report demographic parity as a pass/fail requirement.

---

## 9. Output artifacts

- `reports/fairness_audit.md` — comprehensive, honest write-up: per-subgroup tables,
  pairwise difference tables with CIs, verdicts per attribute/metric, and a plain-language
  findings section including any material disparities and their context.
- `models/fairness/` — machine-readable subgroup metric tables (JSON/CSV) with the
  bootstrap CIs, seed, `S_eval` patient sha256, and package versions for provenance.
- `reports/figures/` — per-subgroup ROC curves, per-subgroup reliability (calibration)
  curves, and a metric-comparison plot (AUROC / TPR / FPR with CI bars across subgroups).
- Code under `src/sentinel/fairness/` (currently only `__init__.py`), reusing
  `evaluation/metrics.py` and the frozen split primitives. The `S_eval` surface is
  obtained the same way as in Phase 2 — no new splitting logic.

---

## 10. Analyst degrees of freedom — frozen list

Everything a later "flexible" analysis could quietly change, fixed here:

- Surface: `S_eval` only; holdout spent. Seed 42.
- Subgroup definitions and collapses: Section 2 (exact).
- Inclusion rule: ≥100 rows AND ≥30 positives (Section 3).
- Primary operating point: 10% capacity; 5%/20% sensitivity only.
- Bootstrap: patient-grouped, 2,000 resamples, 95% percentile CIs, seed 42.
- Material-disparity definition: statistical AND practical, both required (Section 5).
- Practical thresholds: Section 6 (exact numbers).
- Verdict labels and their consequences: Section 7.

---

## Amendments (append-only; dated; reason required)

*(none yet)*

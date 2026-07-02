# Phase 3 — Fairness Audit: implementation spec for Claude Code

> Hand this to Claude Code in the `sentinel` repo. It implements `src/sentinel/fairness/`
> exactly as pre-registered in `reports/fairness_prereg.md`. **Read the pre-registration
> first — it is the source of truth for every subgroup definition, threshold, and verdict
> rule. This spec is the *how*; the prereg is the *what and why*, and it is frozen.**
>
> Branch: `feat/phase3-fairness` (already cut from clean `main`). Windows/PowerShell,
> Python 3.10.11 `.venv`. seed = 42 everywhere. `ruff` + `pytest` must stay green.

## 0. Non-negotiable guardrails (from CONTEXT.md + prereg)

- **Read-only audit.** Trains nothing, registers nothing, fits nothing. No calibration,
  no thresholds learned from data.
- **Holdout stays SPENT.** Never call anything that loads the 20% Phase-1 holdout for
  scoring. The only surface touched is the frozen `S_eval`.
- **Do not modify frozen code.** `evaluation/splits.py` and `calibration/calibration_splits.py`
  are frozen — import them, never edit. Reuse `evaluation/metrics.py`, `clinical_utility/
  calibrated.py`, and `clinical_utility/ranking.py` rather than reimplementing.
- **No new splitting logic.** `S_eval` is recovered via the *existing* pattern (§3).
- **Sensitive attributes are NEVER model inputs.** `race`/`gender`/`payer_code` are read
  from the raw `df_eval` columns only as *grouping labels* for slicing metrics. They are
  not added to any feature frame. (The model already does not consume them as decision
  inputs; keep it that way.)
- **One global decision threshold**, not per-group thresholds. TPR/FPR parity is measured
  at the *same* operating-point threshold for every subgroup (that is what deployment
  does, and it is the correct equalized-odds measurement). Prereg §8 forbids
  group-conditional thresholds.
- **Assert the `S_eval` patient sha256** equals the manifest/operating-points value
  (`71a7a8b69e3b…`) before computing anything — same drift check `evaluate.py` uses.
- seed = 42; patient-grouped bootstrap; 2,000 resamples (prereg §5).

## 1. Module layout

Create under `src/sentinel/fairness/` (currently only `__init__.py`):

```
src/sentinel/fairness/
  __init__.py
  subgroups.py     # subgroup definitions + assignment (pure, target-free)
  metrics.py       # per-subgroup metrics + threshold confusion + pairwise diffs
  bootstrap.py     # patient-grouped bootstrap CIs for subgroup + difference metrics
  audit.py         # entrypoint: recover S_eval, score, evaluate verdicts, emit artifacts
```

Keep `subgroups.py` and `metrics.py` free of MLflow/training imports so they stay unit-
testable without loading the booster.

## 2. `subgroups.py` — subgroup definitions (pre-registered, exact)

Encode prereg §2 as data + pure functions. Do **not** invent groupings; copy these.

```python
# Attribute -> ordered mapping of subgroup label -> predicate over a raw column value.
# "descriptive-only" tier is determined at runtime by the power rule (§3 of prereg),
# NOT hard-coded here — this module only DEFINES groups, power gating lives in metrics.

RACE_GROUPS = ("Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian", "Missing")
# Missing = NaN in the raw `race` column. Do NOT merge small groups.

GENDER_GROUPS = ("Female", "Male")          # drop "Unknown/Invalid" (n=1) explicitly.

# age: raw decade buckets collapsed into 4 bands (prereg §2.3).
AGE_BANDS = {
    "<40":   {"[0-10)", "[10-20)", "[20-30)", "[30-40)"},
    "40-60": {"[40-50)", "[50-60)"},
    "60-80": {"[60-70)", "[70-80)"},
    "80+":   {"[80-90)", "[90-100)"},
}

# payer_code collapsed (prereg §2.4). EXPLORATORY tier — never a standalone fail.
PAYER_GROUPS = {
    "Medicare": {"MC"},
    "Medicaid": {"MD"},
    "Private":  {"BC", "HM", "CP", "CM", "OG", "DM", "CH"},
    "Self-pay": {"SP"},
    "Missing":  {"__NAN__"},
    "Other":    {"UN", "PO", "WC", "OT", "MP", "SI"},
}

ATTRIBUTE_TIER = {"race": "primary", "gender": "primary",
                  "age": "primary", "payer_code": "exploratory"}
```

```python
def assign_subgroups(df_eval: pd.DataFrame, attribute: str) -> pd.Series:
    """Return a categorical Series of subgroup label per row for the given attribute.

    Pure. Reads raw columns only (`race`, `gender`, `age`, `payer_code`). NaN -> "Missing"
    for race/payer; "Unknown/Invalid" rows in gender are labeled and later dropped by the
    caller. Raises on an unknown attribute. Never touches the target or the model.
    """
```

Requirement: for each attribute, the assigned labels must partition every row (assert no
row is unlabeled). Log a warning if any raw value is unmapped (schema drift guard).

## 3. Recovering `S_eval` + calibrated scores (reuse, don't reinvent)

`evaluate.py._recover_s_eval` already does exactly this and asserts the manifest hash.
**Preferred:** factor that private function into a small shared, read-only helper both
modules import — e.g. `src/sentinel/evaluation/surface.py::recover_s_eval(df, manifest)` —
and have `evaluate.py` import it (a pure refactor, no behavior change; keep its tests
green). **Acceptable fallback if you want zero churn in `evaluate.py`:** replicate the
same recovery in `fairness/audit.py`, but it MUST assert the identical
`manifest["split_hashes"]["S_eval"]` and the `S_eval ∩ holdout == ∅` check.

The scoring path is fixed and already dogfooded — reuse verbatim:

```python
from sentinel.clinical_utility import calibrated
booster = calibrated.load_phase1_booster()          # @phase1 base
p_raw   = calibrated.raw_proba(df_eval, booster)     # uncalibrated
p_cal   = calibrated.get_calibrated_proba(df_eval, booster=booster)  # portable manifest map
```

Decision thresholds come from the committed file, not recomputed:

```python
ops = json.load(open(MODELS_DIR / "clinical" / "operating_points.json"))
# implied_threshold per budget: 0.05->0.2471, 0.10->0.1875, 0.20->0.1604
PRIMARY_BUDGET = 0.10   # prereg: 10% is the headline parity point; 5/20% sensitivity only
```

## 4. `metrics.py` — per-subgroup metrics + confusion at threshold + pairwise diffs

Reuse `evaluation.metrics.summarize` for discrimination + calibration; add threshold-based
rates and pairwise differences.

```python
def confusion_at_threshold(y_true, y_prob, threshold) -> dict:
    """TP/FP/TN/FN, TPR (=recall/sensitivity), FPR, selection_rate at a FIXED threshold.
    y_hat = (y_prob >= threshold). Pure. Used for equalized-odds components + demographic
    parity (selection_rate). Threshold is the GLOBAL operating-point threshold, same for
    all subgroups."""

def subgroup_metrics(y_true, y_prob, threshold) -> dict:
    """Merge summarize() (auroc, auprc, brier, ece, n, positives, prevalence) with
    confusion_at_threshold() (tpr, fpr, selection_rate). Guards: if positives==0 or
    the subgroup has a single class, set auroc/auprc to None (undefined) rather than
    raising — the power rule will mark it descriptive-only anyway."""

def is_powered(m: dict) -> bool:
    """Prereg §3 inclusion rule: m['n'] >= 100 AND m['positives'] >= 30."""

def pairwise_difference(m_a: dict, m_b: dict, key: str) -> float:
    """Point difference m_a[key] - m_b[key] for a metric key (auroc/tpr/fpr/ece/...)."""

def four_fifths_ratio(m_a: dict, m_b: dict, key: str) -> float:
    """Ratio m_a[key]/m_b[key] for the 80%-rule secondary screen (tpr/fpr/selection_rate)."""
```

## 5. `bootstrap.py` — patient-grouped bootstrap (2,000 resamples, seed 42)

Mirror the resampling machinery of `ranking.bootstrap_pr_ci` (resample **patients**, gather
their rows), but compute *subgroup* and *pairwise-difference* metrics inside each resample.

```python
def bootstrap_subgroup_metrics(
    y, p_prob, patient_ids, subgroup_labels, threshold, *,
    metrics=("auroc","auprc","ece","tpr","fpr","selection_rate"),
    n_boot=2000, seed=42,
) -> dict:
    """For each subgroup label, return {metric: (lo, hi)} 95% percentile CIs.
    Resample unique patients with replacement -> concat their rows -> for each subgroup,
    subset the bootstrap sample to that subgroup's rows and recompute metrics.
    NaN-guard: if a resampled subgroup has <2 classes or 0 positives, record NaN for
    auroc/auprc/ece for that draw and drop NaNs before taking percentiles (report the
    effective draw count). This is why underpowered groups stay descriptive-only."""

def bootstrap_pairwise_diff(
    y, p_prob, patient_ids, subgroup_labels, group_a, group_b, threshold, *,
    keys=("auroc","tpr","fpr","ece"), n_boot=2000, seed=42,
) -> dict:
    """95% CI of (metric_a - metric_b) per key, using the SAME resampled patients for both
    groups each draw (paired) so the difference CI captures their correlation. Returns
    {key: {"diff": point, "ci": (lo, hi), "ci_excludes_zero": bool}}."""
```

Determinism: one `np.random.default_rng(seed)`, draw the full 2,000-patient-index matrix
once and reuse across subgroups so subgroup CIs and pairwise diffs are computed on the
*same* resamples (required for the paired difference to be valid).

## 6. `audit.py` — entrypoint and verdict logic

`python -m sentinel.fairness.audit`. Steps:

1. Env assert (`sklearn == 1.7.2`), load manifest + operating points.
2. Recover `S_eval` (§3), assert hash `71a7a8b69e3b…`, assert `S_eval ∩ holdout == ∅`.
3. Score once: `p_raw`, `p_cal`. Use `p_cal` for calibration/DCA-style metrics; TPR/FPR use
   `p_cal >= implied_threshold` (calibration-invariant for ranking, but the *threshold* is a
   calibrated probability, so apply on `p_cal`).
4. For each attribute (race, gender, age, payer_code):
   - assign subgroups; drop gender "Unknown/Invalid" (log it).
   - per subgroup: `subgroup_metrics` at the primary threshold (0.1875) + at 5%/20% for
     TPR/FPR sensitivity; mark `powered = is_powered(...)`.
   - bootstrap CIs for all subgroups (§5).
   - for every pair of **powered** subgroups within the attribute: `bootstrap_pairwise_diff`;
     also `four_fifths_ratio` as a secondary screen.
5. **Verdict per attribute/metric family** (prereg §6–7), applied only to powered pairs:
   - `material` iff `ci_excludes_zero` **AND** `abs(diff) >` practical threshold
     (ΔAUROC 0.05, ΔTPR 0.10, ΔFPR 0.10, ΔECE 0.03; per-subgroup ECE flag at 0.05).
   - `PASS` if no powered pair is material; `FLAG` if significant-but-sub-threshold, or
     material only in a descriptive/underpowered group; `MATERIAL DISPARITY` otherwise.
   - `payer_code` verdicts are reported as **exploratory** and cannot, alone, produce a
     project-level MATERIAL verdict (prereg §2.4).
   - demographic-parity selection rates are reported, never gated (prereg §4, §8).
6. Emit artifacts (§7). Print a concise console summary (verdict per attribute).

Verdict thresholds live as named constants at the top of `audit.py` imported/echoed from
the prereg values — do not scatter magic numbers.

## 7. Output artifacts

- `models/fairness/subgroup_metrics.json` — per attribute → per subgroup: n, positives,
  prevalence, auroc/auprc/ece/tpr/fpr/selection_rate with CIs, `powered` flag, threshold
  used. Plus provenance block: `s_eval_patient_sha256`, `seed`, `n_boot`, `sklearn_version`,
  `lightgbm_version`, `git_commit`, primary/secondary thresholds, and a verbatim note that
  sensitive attributes were used only as slicing labels, never as model inputs.
- `models/fairness/pairwise_disparities.json` — per attribute → each powered pair → per
  metric: diff, CI, ci_excludes_zero, practical_threshold, four_fifths_ratio, verdict.
- `reports/fairness_audit.md` — honest write-up: methods recap (pointing to the prereg),
  per-attribute subgroup tables (with CIs and the powered/underpowered tier called out),
  pairwise-difference tables, verdicts, and a plain-language findings section that states
  any material disparities with plausible mechanism (differential missingness, base-rate
  differences, feature availability) — and states plainly where the audit is underpowered
  (race is effectively Caucasian vs AfricanAmerican; payer is exploratory).
- `reports/figures/` — per-subgroup ROC curves and reliability curves (reuse
  `metrics.plot_roc_pr` / `plot_reliability` per subgroup, or a combined overlay), plus a
  metric-comparison plot (AUROC/TPR/FPR with CI error bars across subgroups per attribute).

`models/fairness/*.json` must not be gitignored (check `.gitignore`; add a negation like
the calibrator `.joblib` fix if needed). Figures follow the existing `reports/figures/`
convention.

## 8. Tests (`tests/test_fairness_*.py`, keep CI green)

- `subgroups`: assignment partitions all rows; NaN→Missing for race/payer; age bands map
  every decade bucket exactly once; unknown attribute raises; an injected unmapped value
  triggers the warning.
- `metrics`: `confusion_at_threshold` matches a hand-computed small example; `is_powered`
  boundary (n=100/pos=30 inclusive); `subgroup_metrics` returns None auroc on a single-class
  slice instead of raising.
- `bootstrap`: determinism (same seed → identical CIs); paired diff uses the same resamples
  for both groups; NaN-guard drops degenerate draws and reports effective count.
- `audit` (integration, may be marked slow): runs end-to-end on `S_eval`, asserts the hash
  guard fires on a tampered manifest, asserts holdout is never loaded (e.g. patch/monitor
  `make_holdout_split` is only used for exclusion), and that `payer_code` alone cannot yield
  a project MATERIAL verdict.

## 9. Sanity expectations (NOT targets — do not tune toward these)

From the reproduced `S_eval` marginals, subgroup base rates sit near the 11.5% overall
(Caucasian 11.5%, AfricanAmerican 12.1%, Female 11.6%, Male 11.3%). If any subgroup AUROC
lands far outside the honest ~0.62–0.68 band, or a powered pairwise ΔAUROC exceeds ~0.05,
**investigate before believing it** — small-subgroup AUROC is high-variance, and a
suspicious gap is more likely a power/variance artifact than a real effect. Report what
the data says with its CI; do not massage it.

---

### One-line kickoff for Claude Code

> Read `reports/fairness_prereg.md` and `docs/phase3_fairness_spec.md`. On branch
> `feat/phase3-fairness`, implement `src/sentinel/fairness/` (subgroups, metrics, bootstrap,
> audit) per the spec — read-only audit on the frozen `S_eval`, holdout untouched, seed 42,
> reusing `evaluation/metrics.py`, `clinical_utility/calibrated.py`, and
> `clinical_utility/ranking.py`. Add tests, keep ruff + pytest green, and emit
> `models/fairness/*.json`, `reports/fairness_audit.md`, and figures. Show the plan before
> committing.

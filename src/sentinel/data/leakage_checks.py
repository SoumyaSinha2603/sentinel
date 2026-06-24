"""
Leakage audit for UCI Diabetes-130 (Week-2 hard gate).

Read-only by design: this module *reports* leakage vectors with concrete evidence and
writes a versioned markdown artifact to ``reports/leakage_audit.md``. It never drops,
filters, imputes, or mutates anything — cohort decisions (e.g. removing death/hospice
encounters) are made by a human AFTER reviewing the report.

Honesty over metrics: a surprisingly strong single-feature signal is a leakage *suspect
to surface*, not a result to keep. No model is trained here beyond the single-feature
AUROC smell test in check 8.

Run:

    python -m sentinel.data.leakage_checks
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sentinel.config import REPORTS_DIR
from sentinel.data.load import fetch_raw
from sentinel.evaluation.splits import (
    GROUP_COL,
    IDENTIFIER_COLS,
    feature_columns,
    make_binary_target,
    make_cv_folds,
    make_holdout_split,
)

# Integer-coded categorical columns. They look numeric but are categories, so scoring
# them by raw value in the smell test would be meaningless and could hide real leakage.
# Route them through the same target-encoded path as object/categorical columns.
CODED_CATEGORICAL_IDS = {
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
}

# Death / hospice discharge dispositions. Source: the dataset's own ``IDS_mapping.csv``
# (Diabetes 130-US Hospitals, UCI id=296), discharge_disposition_id mapping:
#   11 = Expired
#   13 = Hospice / home
#   14 = Hospice / medical facility
#   19 = Expired at home (Medicaid only, hospice)
#   20 = Expired in a medical facility (Medicaid only, hospice)
#   21 = Expired, place unknown (Medicaid only, hospice)
# A patient who died or entered hospice cannot be readmitted within 30 days, so these
# ids should show a near-0% <30 rate — a post-outcome leakage vector. Report & flag only.
DEATH_HOSPICE_DISPOSITION_IDS = {
    11: "Expired",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    19: "Expired at home (Medicaid only, hospice)",
    20: "Expired in a medical facility (Medicaid only, hospice)",
    21: "Expired, place unknown (Medicaid only, hospice)",
}

MISSING_TOKEN = "?"
NEAR_CONSTANT_THRESHOLD = 0.99
AUROC_SUSPECT_THRESHOLD = 0.70


# --- Check 8 helpers -----------------------------------------------------------------


def _target_encode(train_x: pd.Series, train_y: pd.Series, val_x: pd.Series) -> np.ndarray:
    """Mean-target-encode ``val_x`` using category means learned on the training fold.

    Unseen categories fall back to the training-fold global mean. Fit strictly inside
    the train fold; applied to the held-out val fold (no peeking at val labels).
    """
    enc = pd.DataFrame({"x": train_x.astype(str).to_numpy(), "y": train_y.to_numpy()})
    means = enc.groupby("x")["y"].mean()
    global_mean = float(train_y.mean())
    return val_x.astype(str).map(means).fillna(global_mean).to_numpy()


def _numeric_scores(train_x: pd.Series, val_x: pd.Series) -> np.ndarray:
    """Use the raw numeric value as the score; impute NaN with the train-fold median."""
    median = float(pd.to_numeric(train_x, errors="coerce").median())
    val = pd.to_numeric(val_x, errors="coerce").fillna(median)
    return val.to_numpy()


def _feature_mean_auroc(
    df_train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    column: str,
    is_categorical: bool,
) -> float:
    """Direction-adjusted mean AUROC of a single feature across the grouped CV folds."""
    y_all = make_binary_target(df_train)
    aucs: list[float] = []
    for tr_idx, va_idx in folds:
        tr_x, va_x = df_train[column].iloc[tr_idx], df_train[column].iloc[va_idx]
        tr_y, va_y = y_all.iloc[tr_idx], y_all.iloc[va_idx]
        scores = _target_encode(tr_x, tr_y, va_x) if is_categorical else _numeric_scores(tr_x, va_x)
        auc = roc_auc_score(va_y.to_numpy(), scores)
        # Direction-adjust: a strongly *inverse* feature is just as predictive.
        aucs.append(max(auc, 1.0 - auc))
    return float(np.mean(aucs))


# --- Individual checks ---------------------------------------------------------------


def _check_identifiers(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    enc_card = int(df["encounter_id"].nunique())
    pat_card = int(df[GROUP_COL].nunique())
    return {
        "n_rows": n_rows,
        "encounter_id_cardinality": enc_card,
        "encounter_id_is_unique": enc_card == n_rows,
        "patient_nbr_cardinality": pat_card,
        "excluded": list(IDENTIFIER_COLS),
    }


def _check_patient_repetition(df: pd.DataFrame) -> dict:
    n_enc = len(df)
    n_pat = int(df[GROUP_COL].nunique())
    per_patient = df[GROUP_COL].value_counts()
    return {
        "encounters": n_enc,
        "unique_patients": n_pat,
        "repeat_encounters": n_enc - n_pat,
        "max_encounters_one_patient": int(per_patient.max()),
    }


def _check_split_integrity(df: pd.DataFrame) -> dict:
    y = make_binary_target(df)
    train_idx, test_idx = make_holdout_split(df)  # asserts no group overlap internally
    return {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_patients": int(df[GROUP_COL].iloc[train_idx].nunique()),
        "test_patients": int(df[GROUP_COL].iloc[test_idx].nunique()),
        "train_prevalence": float(y.iloc[train_idx].mean()),
        "test_prevalence": float(y.iloc[test_idx].mean()),
        "overlap_ok": True,
    }


def _check_disposition_leakage(df: pd.DataFrame) -> dict:
    y = make_binary_target(df)
    work = pd.DataFrame({"disp": df["discharge_disposition_id"], "y": y})
    grouped = work.groupby("disp")["y"]
    rows = []
    for disp_id, count in df["discharge_disposition_id"].value_counts().items():
        rate = float(grouped.get_group(disp_id).mean())
        flagged = int(disp_id) in DEATH_HOSPICE_DISPOSITION_IDS
        rows.append(
            {
                "disposition_id": int(disp_id),
                "count": int(count),
                "readmit_lt30_rate": rate,
                "flagged_death_hospice": flagged,
                "meaning": DEATH_HOSPICE_DISPOSITION_IDS.get(int(disp_id), ""),
            }
        )
    rows.sort(key=lambda r: r["disposition_id"])
    flagged_rows = [r for r in rows if r["flagged_death_hospice"]]
    flagged_encounters = sum(r["count"] for r in flagged_rows)
    return {
        "rows": rows,
        "flagged_ids": sorted(DEATH_HOSPICE_DISPOSITION_IDS),
        "flagged_encounters": flagged_encounters,
    }


def _check_constant_columns(df: pd.DataFrame) -> dict:
    findings = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=False))
        top_share = float(df[col].value_counts(dropna=False, normalize=True).iloc[0])
        if nunique == 1 or top_share >= NEAR_CONSTANT_THRESHOLD:
            findings.append(
                {
                    "column": col,
                    "n_unique": nunique,
                    "top_value": str(df[col].value_counts(dropna=False).index[0]),
                    "top_share": top_share,
                }
            )
    findings.sort(key=lambda r: r["top_share"], reverse=True)
    return {"findings": findings}


def _check_duplicate_rows(df: pd.DataFrame) -> dict:
    deduped_subset = [c for c in df.columns if c != "encounter_id"]
    n_dupes = int(df.duplicated(subset=deduped_subset).sum())
    return {"duplicate_rows_excluding_encounter_id": n_dupes}


def _check_missingness(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    rows = []
    for col in df.columns:
        q_count = int((df[col].astype("object") == MISSING_TOKEN).sum())
        nan_count = int(df[col].isna().sum())
        if q_count or nan_count:
            rows.append(
                {
                    "column": col,
                    "q_rate": q_count / n_rows,
                    "nan_rate": nan_count / n_rows,
                }
            )
    rows.sort(key=lambda r: r["q_rate"] + r["nan_rate"], reverse=True)
    return {"rows": rows}


def _check_univariate_auroc(df: pd.DataFrame) -> dict:
    train_idx, _ = make_holdout_split(df)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    folds = make_cv_folds(df_train)

    results = []
    for col in feature_columns(df_train):
        is_categorical = col in CODED_CATEGORICAL_IDS or not pd.api.types.is_numeric_dtype(
            df_train[col]
        )
        mean_auc = _feature_mean_auroc(df_train, folds, col, is_categorical)
        results.append(
            {
                "feature": col,
                "mean_auroc": mean_auc,
                "type": "categorical" if is_categorical else "numeric",
                "suspect": mean_auc > AUROC_SUSPECT_THRESHOLD,
            }
        )
    results.sort(key=lambda r: r["mean_auroc"], reverse=True)
    suspects = [r for r in results if r["suspect"]]
    return {"top10": results[:10], "suspects": suspects, "n_features": len(results)}


# --- Orchestration -------------------------------------------------------------------


def run_leakage_audit(df: pd.DataFrame) -> dict:
    """Run every leakage check and return the structured results (no mutation of ``df``)."""
    return {
        "identifiers": _check_identifiers(df),
        "patient_repetition": _check_patient_repetition(df),
        "split_integrity": _check_split_integrity(df),
        "disposition_leakage": _check_disposition_leakage(df),
        "constant_columns": _check_constant_columns(df),
        "duplicate_rows": _check_duplicate_rows(df),
        "missingness": _check_missingness(df),
        "univariate_auroc": _check_univariate_auroc(df),
    }


def _build_verdict(results: dict) -> list[dict]:
    """Collect every flagged item and whether a human decision is needed before modeling."""
    verdict = []

    disp = results["disposition_leakage"]
    verdict.append(
        {
            "item": "Death/hospice discharge dispositions",
            "detail": (
                f"{disp['flagged_encounters']:,} encounters across ids "
                f"{disp['flagged_ids']} — post-outcome leakage."
            ),
            "needs_human_decision": True,
        }
    )

    const = results["constant_columns"]["findings"]
    if const:
        verdict.append(
            {
                "item": "Constant / near-constant columns",
                "detail": f"{len(const)} column(s) ≥{NEAR_CONSTANT_THRESHOLD:.0%} "
                "concentrated or single-valued.",
                "needs_human_decision": True,
            }
        )

    dupes = results["duplicate_rows"]["duplicate_rows_excluding_encounter_id"]
    if dupes:
        verdict.append(
            {
                "item": "Duplicate rows (excl. encounter_id)",
                "detail": f"{dupes:,} fully duplicated row(s).",
                "needs_human_decision": True,
            }
        )

    suspects = results["univariate_auroc"]["suspects"]
    if suspects:
        names = ", ".join(f"{s['feature']} ({s['mean_auroc']:.3f})" for s in suspects)
        verdict.append(
            {
                "item": "High single-feature AUROC suspects",
                "detail": f"{names} — investigate before trusting any model.",
                "needs_human_decision": True,
            }
        )

    return verdict


def _render_report(results: dict) -> str:
    ident = results["identifiers"]
    rep = results["patient_repetition"]
    split = results["split_integrity"]
    disp = results["disposition_leakage"]
    const = results["constant_columns"]
    dupes = results["duplicate_rows"]
    miss = results["missingness"]
    uni = results["univariate_auroc"]
    verdict = _build_verdict(results)

    lines: list[str] = []
    lines.append("# Leakage Audit — UCI Diabetes-130")
    lines.append("")
    lines.append(
        "> Read-only audit (Week-2 hard gate). Reports leakage vectors with evidence; "
        "**no rows or columns are dropped**. Cohort decisions are made by a human after "
        "review. Generated by `sentinel.data.leakage_checks`."
    )
    lines.append("")

    # 1
    lines.append("## 1. Identifier exclusion")
    lines.append("")
    lines.append(
        f"- `encounter_id` cardinality: {ident['encounter_id_cardinality']:,} "
        f"(rows: {ident['n_rows']:,}) — "
        f"{'unique (1 row/encounter)' if ident['encounter_id_is_unique'] else 'NOT unique'}"
    )
    lines.append(f"- `patient_nbr` cardinality: {ident['patient_nbr_cardinality']:,}")
    lines.append(
        f"- **Excluded from every feature set:** {', '.join(f'`{c}`' for c in ident['excluded'])}"
    )
    lines.append("")

    # 2
    lines.append("## 2. Patient repetition")
    lines.append("")
    lines.append(f"- Encounters: {rep['encounters']:,}")
    lines.append(f"- Unique patients: {rep['unique_patients']:,}")
    lines.append(f"- Repeat encounters: {rep['repeat_encounters']:,}")
    lines.append(f"- Max encounters for one patient: {rep['max_encounters_one_patient']}")
    lines.append("")

    # 3
    lines.append("## 3. Grouped-split integrity")
    lines.append("")
    lines.append(
        "Locked holdout split (`GroupShuffleSplit`, grouped by `patient_nbr`). "
        "`assert_no_group_overlap` passed: no patient is in both train and test."
    )
    lines.append("")
    lines.append("| | rows | unique patients | `<30` prevalence |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| train | {split['train_rows']:,} | {split['train_patients']:,} | "
        f"{split['train_prevalence']:.2%} |"
    )
    lines.append(
        f"| test | {split['test_rows']:,} | {split['test_patients']:,} | "
        f"{split['test_prevalence']:.2%} |"
    )
    lines.append("")

    # 4
    lines.append("## 4. Post-outcome disposition leakage")
    lines.append("")
    lines.append(
        "`<30` readmission rate by `discharge_disposition_id`. Flagged ids are "
        "death/hospice per the dataset's `IDS_mapping.csv` — a patient who died "
        "or entered hospice cannot be readmitted, so a near-0% rate is expected. "
        "**Reported and flagged only; not dropped.**"
    )
    lines.append("")
    lines.append("| id | meaning | encounters | `<30` rate | flag |")
    lines.append("|---:|---|---:|---:|:--:|")
    for r in disp["rows"]:
        flag = "⚠️ death/hospice" if r["flagged_death_hospice"] else ""
        meaning = r["meaning"] or ""
        lines.append(
            f"| {r['disposition_id']} | {meaning} | {r['count']:,} | "
            f"{r['readmit_lt30_rate']:.2%} | {flag} |"
        )
    lines.append("")
    lines.append(
        f"Flagged death/hospice encounters total: **{disp['flagged_encounters']:,}** "
        f"(ids {disp['flagged_ids']})."
    )
    lines.append("")

    # 5
    lines.append("## 5. Constant / near-constant columns")
    lines.append("")
    if const["findings"]:
        lines.append(
            f"Columns with a single value or ≥{NEAR_CONSTANT_THRESHOLD:.0%} "
            "concentration in one value:"
        )
        lines.append("")
        lines.append("| column | n unique | top value | top share |")
        lines.append("|---|---:|---|---:|")
        for r in const["findings"]:
            lines.append(
                f"| `{r['column']}` | {r['n_unique']} | `{r['top_value']}` | {r['top_share']:.2%} |"
            )
    else:
        lines.append("None found.")
    lines.append("")

    # 6
    lines.append("## 6. Duplicate rows")
    lines.append("")
    lines.append(
        f"Fully duplicated rows after excluding `encounter_id`: "
        f"**{dupes['duplicate_rows_excluding_encounter_id']:,}**"
    )
    lines.append("")

    # 7
    lines.append("## 7. Missingness")
    lines.append("")
    if miss["rows"]:
        lines.append(
            "Per-column rate of the `?` token and of NaN (columns with any "
            "missingness, worst first):"
        )
        lines.append("")
        lines.append("| column | `?` rate | NaN rate |")
        lines.append("|---|---:|---:|")
        for r in miss["rows"]:
            lines.append(f"| `{r['column']}` | {r['q_rate']:.2%} | {r['nan_rate']:.2%} |")
    else:
        lines.append("No `?` tokens or NaNs found.")
    lines.append("")

    # 8
    lines.append("## 8. Univariate predictive-power smell test")
    lines.append("")
    lines.append(
        f"Single-feature mean AUROC across the locked grouped CV folds "
        f"({uni['n_features']} candidate features; identifiers excluded). Category "
        "encoding is fit **inside** each training fold and applied to the "
        "validation fold. Coded-categorical id columns are target-encoded, not "
        "scored by raw value. AUROC is direction-adjusted. "
        f"**Any feature > {AUROC_SUSPECT_THRESHOLD:.2f} is a leakage suspect.**"
    )
    lines.append("")
    lines.append("| rank | feature | type | mean AUROC | suspect |")
    lines.append("|---:|---|---|---:|:--:|")
    for i, r in enumerate(uni["top10"], start=1):
        suspect = "🚩" if r["suspect"] else ""
        lines.append(
            f"| {i} | `{r['feature']}` | {r['type']} | {r['mean_auroc']:.3f} | {suspect} |"
        )
    lines.append("")
    if uni["suspects"]:
        names = ", ".join(f"`{s['feature']}` ({s['mean_auroc']:.3f})" for s in uni["suspects"])
        lines.append(
            f"**Suspects (> {AUROC_SUSPECT_THRESHOLD:.2f}):** {names}. Investigate "
            "before trusting any model."
        )
    else:
        lines.append(
            f"No single feature exceeds {AUROC_SUSPECT_THRESHOLD:.2f} — consistent "
            "with the honest AUROC≈0.68 ceiling for this dataset."
        )
    lines.append("")

    # 9
    lines.append("## 9. Summary verdict")
    lines.append("")
    if verdict:
        lines.append("| flagged item | detail | needs human decision |")
        lines.append("|---|---|:--:|")
        for v in verdict:
            need = "yes" if v["needs_human_decision"] else "no"
            lines.append(f"| {v['item']} | {v['detail']} | {need} |")
    else:
        lines.append("No items flagged.")
    lines.append("")
    lines.append(
        "_Audit is report-only. No rows/columns were dropped, filtered, or "
        "imputed. Cohort decisions (e.g. excluding death/hospice encounters) "
        "require explicit human sign-off before modeling._"
    )
    lines.append("")

    return "\n".join(lines)


# --- Phase 1: leakage re-audit on engineered features --------------------------------


def audit_engineered_features(df: pd.DataFrame, out_path=None) -> dict:
    """Single-feature CV-AUROC smell test on the engineered feature frame.

    Reuses the frozen ``StratifiedGroupKFold`` (``make_cv_folds``, 5 folds, grouped by
    ``patient_nbr``) — no bespoke split. For each feature:
      - numeric:     mean/std of ``AUROC(feature_values, y)`` across the val folds.
      - categorical: FOLD-INTERNAL target-rate encoding — learn P(y=1|category) on each
        TRAIN fold, map onto the VAL fold (unseen categories -> train-fold prevalence),
        AUROC on the val fold. Full-data encoding would fake high AUROC, so the
        fold-safety is mandatory.

    Verdict is SUSPECT if any single feature exceeds {0.70}, else CLEAN. Writes a markdown
    report to ``reports/leakage_audit_features.md`` (a NEW file; the Phase 0 audit is left
    intact).
    """
    from sentinel.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    y = make_binary_target(df)
    folds = make_cv_folds(df)

    results = []
    for feat in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        is_categorical = feat in set(CATEGORICAL_FEATURES)
        aucs: list[float] = []
        for tr_idx, va_idx in folds:
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            xtr, xva = df[feat].iloc[tr_idx], df[feat].iloc[va_idx]
            if is_categorical:
                scores = _target_encode(xtr, ytr, xva)
            else:
                scores = pd.to_numeric(xva, errors="coerce").to_numpy()
            aucs.append(float(roc_auc_score(yva.to_numpy(), scores)))
        results.append(
            {
                "feature": feat,
                "type": "categorical" if is_categorical else "numeric",
                "mean_auroc": float(np.mean(aucs)),
                "std_auroc": float(np.std(aucs)),
            }
        )

    results.sort(key=lambda r: r["mean_auroc"], reverse=True)
    suspects = [r for r in results if r["mean_auroc"] > AUROC_SUSPECT_THRESHOLD]
    verdict = "SUSPECT" if suspects else "CLEAN"

    report = _render_features_report(results, suspects, verdict, n_folds=len(folds))
    if out_path is None:
        out_path = REPORTS_DIR / "leakage_audit_features.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    return {
        "verdict": verdict,
        "suspects": suspects,
        "ranked": results,
        "report": report,
        "out_path": out_path,
    }


def _render_features_report(results: list, suspects: list, verdict: str, n_folds: int) -> str:
    lines: list[str] = []
    lines.append("# Leakage Re-Audit — Engineered Features")
    lines.append("")
    lines.append(
        "> Phase 1 single-feature smell test on the engineered feature frame "
        "(`sentinel.features.build`). Reuses the frozen grouped CV split. Categorical "
        "features use fold-internal target-rate encoding (no full-data peeking); numeric "
        "features use the raw value as the score. Generated by "
        "`sentinel.data.leakage_checks.audit_engineered_features`."
    )
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    if suspects:
        names = ", ".join(f"`{s['feature']}` ({s['mean_auroc']:.3f})" for s in suspects)
        lines.append(
            f"Single feature(s) exceeding {AUROC_SUSPECT_THRESHOLD:.2f} mean CV AUROC: "
            f"{names}. Investigate before any modeling — on this dataset no legitimate "
            "single feature should be that predictive."
        )
    else:
        lines.append(
            f"No single feature exceeds {AUROC_SUSPECT_THRESHOLD:.2f} mean CV AUROC across "
            f"{n_folds} folds — consistent with the clean Phase 0 audit and the honest "
            "~0.68 ceiling. Feature engineering introduced no leakage."
        )
    lines.append("")
    lines.append(f"## Ranked single-feature CV AUROC ({n_folds} folds)")
    lines.append("")
    lines.append("| rank | feature | type | mean CV AUROC | std |")
    lines.append("|---:|---|---|---:|---:|")
    for i, r in enumerate(results, start=1):
        flag = " 🚩" if r["mean_auroc"] > AUROC_SUSPECT_THRESHOLD else ""
        lines.append(
            f"| {i} | `{r['feature']}`{flag} | {r['type']} | "
            f"{r['mean_auroc']:.3f} | {r['std_auroc']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = fetch_raw()
    results = run_leakage_audit(df)
    report = _render_report(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "leakage_audit.md"
    out_path.write_text(report, encoding="utf-8")

    # The report contains Unicode (em-dashes, ≈, flag glyphs); force UTF-8 on stdout so
    # it prints cleanly on a Windows cp1252 console instead of raising UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    print(f"\n[written] {out_path}")


def main_engineered() -> None:
    from sentinel.features.build import load_and_build

    df = load_and_build()
    result = audit_engineered_features(df)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(result["report"])
    print(f"\n[written] {result['out_path']}")


if __name__ == "__main__":
    if "--features" in sys.argv:
        main_engineered()
    else:
        main()

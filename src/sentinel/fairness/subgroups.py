"""Subgroup definitions for the Phase-3 fairness audit (pre-registered, exact).

Encodes `reports/fairness_prereg.md` §2 as data + a single pure assignment function.
This module only *defines* groups; the minimum-power gating (prereg §3) lives in
`fairness.metrics.is_powered`, and pass/fail verdicts live in `fairness.audit`.

Pure and target-free: it reads the raw cohort columns (`race`, `gender`, `age`,
`payer_code`) only as *grouping labels* for slicing metrics. Sensitive attributes are
NEVER model inputs — this module never touches the feature frame, the target, or the model.
"""

from __future__ import annotations

import warnings

import pandas as pd

# --- Pre-registered subgroup definitions (prereg §2 — copy, do not invent) -------------

# race (prereg §2.1). "Missing" = NaN in the raw column. Small groups are NOT merged.
RACE_GROUPS = ("Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian", "Missing")

# gender (prereg §2.2). "Unknown/Invalid" (n=1) is labeled here and dropped by the caller.
GENDER_GROUPS = ("Female", "Male")
GENDER_DROP_LABEL = "Unknown/Invalid"

# age (prereg §2.3): raw decade buckets collapsed into four clinically sensible bands.
AGE_BANDS = {
    "<40": {"[0-10)", "[10-20)", "[20-30)", "[30-40)"},
    "40-60": {"[40-50)", "[50-60)"},
    "60-80": {"[60-70)", "[70-80)"},
    "80+": {"[80-90)", "[90-100)"},
}
AGE_BAND_ORDER = ("<40", "40-60", "60-80", "80+")

# payer_code (prereg §2.4) — EXPLORATORY tier. NaN carried as the "__NAN__" sentinel.
PAYER_GROUPS = {
    "Medicare": {"MC"},
    "Medicaid": {"MD"},
    "Private": {"BC", "HM", "CP", "CM", "OG", "DM", "CH"},
    "Self-pay": {"SP"},
    "Missing": {"__NAN__"},
    "Other": {"UN", "PO", "WC", "OT", "MP", "SI"},
}

ATTRIBUTE_TIER = {
    "race": "primary",
    "gender": "primary",
    "age": "primary",
    "payer_code": "exploratory",
}

# Reverse lookups (built once).
_AGE_BUCKET_TO_BAND = {bucket: band for band, buckets in AGE_BANDS.items() for bucket in buckets}
_PAYER_CODE_TO_GROUP = {code: group for group, codes in PAYER_GROUPS.items() for code in codes}


def _warn_unmapped(attribute: str, values: set) -> None:
    """Schema-drift guard: warn (never silently reclassify) on unexpected raw values."""
    warnings.warn(
        f"assign_subgroups({attribute!r}): unmapped raw value(s) "
        f"{sorted(str(v) for v in values)} — possible schema drift",
        stacklevel=3,
    )


def _as_ordered_categorical(labels: pd.Series, declared_order, index, name: str) -> pd.Series:
    """Categorical with the declared order first, then any observed extras (never dropped)."""
    observed = list(pd.unique(labels))
    categories = [c for c in declared_order if c in observed]
    categories += [o for o in observed if o not in categories]
    cat = pd.Categorical(labels, categories=categories, ordered=False)
    return pd.Series(cat, index=index, name=name)


def assign_subgroups(df_eval: pd.DataFrame, attribute: str) -> pd.Series:
    """Return a categorical Series of subgroup label per row for the given attribute.

    Pure. Reads raw columns only. NaN -> "Missing" for race/payer_code; a NaN or
    "Unknown/Invalid" `gender` is labeled `GENDER_DROP_LABEL` and later dropped by the
    caller. Raises ``ValueError`` on an unknown attribute. Every row is labeled (the
    output partitions the frame); unmapped raw values trigger a warning (schema drift).
    """
    if attribute not in ATTRIBUTE_TIER:
        raise ValueError(
            f"unknown sensitive attribute: {attribute!r} (known: {list(ATTRIBUTE_TIER)})"
        )

    col = df_eval[attribute]

    if attribute == "race":
        labels = col.where(col.notna(), "Missing").astype(object)
        unmapped = set(pd.unique(labels)) - set(RACE_GROUPS)
        if unmapped:
            _warn_unmapped(attribute, unmapped)
        declared = RACE_GROUPS

    elif attribute == "gender":
        # NaN gender is treated like "Unknown/Invalid" (labeled, then dropped by caller).
        labels = col.where(col.notna(), GENDER_DROP_LABEL).astype(object)
        unmapped = set(pd.unique(labels)) - (set(GENDER_GROUPS) | {GENDER_DROP_LABEL})
        if unmapped:
            _warn_unmapped(attribute, unmapped)
        declared = (*GENDER_GROUPS, GENDER_DROP_LABEL)

    elif attribute == "age":
        labels = col.astype(object).map(_AGE_BUCKET_TO_BAND)
        unmapped_mask = labels.isna()
        if unmapped_mask.any():
            _warn_unmapped(attribute, set(pd.unique(col[unmapped_mask])))
            # keep the raw bucket as its own (extra) group rather than dropping the row.
            labels = labels.where(~unmapped_mask, col.astype(object))
        declared = AGE_BAND_ORDER

    elif attribute == "payer_code":
        raw = col.where(col.notna(), "__NAN__").astype(object)
        labels = raw.map(_PAYER_CODE_TO_GROUP)
        unmapped_mask = labels.isna()
        if unmapped_mask.any():
            _warn_unmapped(attribute, set(pd.unique(raw[unmapped_mask])))
            # pool unrecognized codes into the pre-registered "Other/Unknown" bucket.
            labels = labels.where(~unmapped_mask, "Other")
        declared = tuple(PAYER_GROUPS.keys())

    else:  # pragma: no cover - guarded above
        raise ValueError(attribute)

    if labels.isna().any():
        raise AssertionError(
            f"assign_subgroups({attribute!r}) left {int(labels.isna().sum())} rows unlabeled"
        )

    return _as_ordered_categorical(labels, declared, df_eval.index, attribute)

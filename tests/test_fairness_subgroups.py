"""Tests for subgroup assignment (sentinel.fairness.subgroups)."""

import numpy as np
import pandas as pd
import pytest

from sentinel.fairness import subgroups


def _df(**cols):
    return pd.DataFrame(cols)


def test_race_nan_maps_to_missing_and_partitions():
    df = _df(race=["Caucasian", "AfricanAmerican", np.nan, "Asian"])
    s = subgroups.assign_subgroups(df, "race")
    assert list(s) == ["Caucasian", "AfricanAmerican", "Missing", "Asian"]
    assert s.notna().all()  # every row labeled (partition)


def test_payer_nan_and_pooling():
    df = _df(payer_code=["MC", "BC", "HM", np.nan, "SP", "UN"])
    s = subgroups.assign_subgroups(df, "payer_code")
    assert list(s) == ["Medicare", "Private", "Private", "Missing", "Self-pay", "Other"]


def test_age_bands_map_every_decade_bucket_exactly_once():
    buckets = list(subgroups.AGE_BANDS.values())
    flat = [b for group in buckets for b in group]
    assert len(flat) == len(set(flat)) == 10  # all ten decade buckets, no overlap
    df = _df(age=["[0-10)", "[30-40)", "[40-50)", "[70-80)", "[90-100)"])
    s = subgroups.assign_subgroups(df, "age")
    assert list(s) == ["<40", "<40", "40-60", "60-80", "80+"]


def test_gender_unknown_labeled_not_dropped_here():
    df = _df(gender=["Female", "Male", "Unknown/Invalid"])
    s = subgroups.assign_subgroups(df, "gender")
    assert list(s) == ["Female", "Male", subgroups.GENDER_DROP_LABEL]


def test_unknown_attribute_raises():
    with pytest.raises(ValueError):
        subgroups.assign_subgroups(_df(race=["Caucasian"]), "ethnicity")


def test_unmapped_value_triggers_warning_and_still_partitions():
    df = _df(race=["Caucasian", "Klingon"])  # injected unmapped value
    with pytest.warns(UserWarning, match="unmapped"):
        s = subgroups.assign_subgroups(df, "race")
    assert s.notna().all()
    assert "Klingon" in set(s)  # surfaced as its own group, not silently dropped

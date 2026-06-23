"""
Load the UCI Diabetes-130 dataset (UCI id=296) into ``data/raw/``.

The dataset is "Diabetes 130-US Hospitals for Years 1999-2008": 101,766 inpatient
encounters, 50 columns (including the ``encounter_id`` / ``patient_nbr`` identifiers
and the ``readmitted`` target).

Run as a script to fetch + validate + write the raw CSV:

    python -m sentinel.data.load            # fetch (skips if file already present)
    python -m sentinel.data.load --force    # force a re-fetch and overwrite

The integrity checks exist so that a truncated or malformed download fails loudly
here, rather than silently corrupting every downstream step.
"""

from __future__ import annotations

import sys

import pandas as pd

from sentinel.config import RAW_DIR

UCI_ID = 296
RAW_FILENAME = "diabetes_130.csv"

EXPECTED_ROWS = 101_766
EXPECTED_COLS = 50
REQUIRED_COLUMNS = {"encounter_id", "patient_nbr", "readmitted"}
VALID_TARGET_VALUES = {"<30", ">30", "NO"}


def _assemble_full_frame(dataset) -> pd.DataFrame:
    """Reconstruct the complete 50-column frame regardless of how ucimlrepo splits it."""
    data = dataset.data
    original = getattr(data, "original", None)
    if original is not None:
        return original.copy()
    parts = [p for p in (data.ids, data.features, data.targets) if p is not None]
    return pd.concat(parts, axis=1)


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError with all problems found, or return None if the frame is sound."""
    problems: list[str] = []

    if df.shape[0] != EXPECTED_ROWS:
        problems.append(f"row count {df.shape[0]:,} != expected {EXPECTED_ROWS:,}")
    if df.shape[1] != EXPECTED_COLS:
        problems.append(
            f"column count {df.shape[1]} != expected {EXPECTED_COLS}; "
            f"got columns: {list(df.columns)}"
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        problems.append(f"missing required columns: {sorted(missing)}")

    if "encounter_id" in df.columns and df["encounter_id"].duplicated().any():
        n_dupe = int(df["encounter_id"].duplicated().sum())
        problems.append(f"encounter_id not unique ({n_dupe} duplicates); expected 1 row/encounter")

    if "readmitted" in df.columns:
        unexpected = set(df["readmitted"].dropna().unique()) - VALID_TARGET_VALUES
        if unexpected:
            problems.append(f"unexpected 'readmitted' values: {sorted(unexpected)}")

    if problems:
        raise ValueError("Dataset integrity check FAILED:\n  - " + "\n  - ".join(problems))


def fetch_raw(force: bool = False) -> pd.DataFrame:
    """Fetch (or reload) the raw dataset, validate it, and persist to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / RAW_FILENAME

    if out_path.exists() and not force:
        df = pd.read_csv(out_path)
        validate(df)
        return df

    from ucimlrepo import fetch_ucirepo  # imported lazily so tests don't need network

    dataset = fetch_ucirepo(id=UCI_ID)
    df = _assemble_full_frame(dataset)
    validate(df)
    df.to_csv(out_path, index=False)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    n_encounters = len(df)
    n_patients = df["patient_nbr"].nunique()
    dupe_patients = n_encounters - n_patients

    print("\n" + "=" * 60)
    print("UCI Diabetes-130 loaded and validated")
    print("=" * 60)
    print(f"  rows (encounters) : {n_encounters:,}")
    print(f"  columns           : {df.shape[1]}")
    print(f"  unique patients   : {n_patients:,}")
    print(f"  repeat encounters : {dupe_patients:,}  <-- leakage risk (same patient, many rows)")
    print("\n  raw 'readmitted' distribution:")
    counts = df["readmitted"].value_counts(dropna=False)
    for value, count in counts.items():
        print(f"    {str(value):>4} : {count:>7,}  ({count / n_encounters:6.2%})")
    print(f"\n  saved to: {RAW_DIR / RAW_FILENAME}")
    print("=" * 60)


def main() -> None:
    force = "--force" in sys.argv
    df = fetch_raw(force=force)
    _print_summary(df)


if __name__ == "__main__":
    main()

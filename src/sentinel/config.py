"""Central configuration: paths and constants. Dependency-free on purpose."""

from __future__ import annotations

from pathlib import Path

# Repository root (this file lives at <root>/src/sentinel/config.py)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Reproducibility
RANDOM_SEED = 42

# UCI Diabetes-130 specifics
TARGET_COLUMN = "readmitted"  # will be binarized in the data pipeline

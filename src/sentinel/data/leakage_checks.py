"""
Leakage detection harness (Week-2 hard gate).

This is a STUB. The full implementation lands in the Week-2 leakage-audit step.
Planned checks:
  - target-correlated identifiers / index leakage
  - features unavailable at prediction time (post-outcome columns)
  - train/test contamination via duplicate encounters per patient
  - suspiciously high single-feature AUROC (a leakage smell test)
"""

from __future__ import annotations


def run_leakage_checks() -> None:
    raise NotImplementedError("Implemented in the Week-2 leakage-audit step.")

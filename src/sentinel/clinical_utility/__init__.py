"""Phase 2 — Trust Layer I: clinical utility (DCA, precision@k, operating points).

Read-only evaluation of the shipped calibrated-probability path (@phase1 booster + the
committed isotonic manifest) on the FROZEN Phase-2 eval surface `S_eval`. No model is
trained or registered here; the Phase-1 holdout is never loaded.
"""

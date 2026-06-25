"""Calibration layer (Phase 2, Trust Layer I).

Carves its own evaluation surfaces from *inside* the 80% train (the Phase-1 holdout is
spent), fits probability calibrators on out-of-fold scores, and selects between isotonic
and Platt under a pre-registered rule. Never touches the frozen split harness
(``sentinel.evaluation.splits``) and never scores the 20% holdout.
"""

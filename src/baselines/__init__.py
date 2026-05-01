"""Reference baselines for LOB direction prediction.

These models are kept self-contained so they can be trained independently
from the FinDrama world-model stack and report direction-prediction metrics
on the same Polymarket validation split.
"""
from sub_models.lob_encoder import K_LEVELS  # noqa: F401  re-export for convenience.

"""Reference baselines for LOB direction prediction.

These models are kept self-contained so they can be trained independently
from the FinMamba3 world-model stack and report direction-prediction metrics
on the same Polymarket validation split.
"""
# region imports
from finmamba3.models.lob_encoder import K_LEVELS  # noqa: F401  re-export for convenience.
# endregion

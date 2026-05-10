"""FI-2010 limit order book loader (Ntakaris et al. 2018).

FI-2010 is the canonical public benchmark for LOB mid-price forecasting,
used by DeepLOB and many follow-ups. Files are space-separated text in
shape (149, N_events):

  rows  0-39   : 10 levels x (ask_price, ask_size, bid_price, bid_size)
  rows  40-143 : derived hand-crafted features (ignored; we recompute ours)
  rows 144-148 : 5 horizon labels (k = 10, 20, 30, 50, 100 events)

The NoAuction ZScore variant is already per-stock per-day z-score normalized,
so the downstream replay buffer can consume it without a second normalization
pass. We still clip to the BasicSettings.NormClip window to match the
Polymarket pipeline's safety net.

Public entry point: load_fi2010_split(data_dir, split, horizon).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from envs.lob_features import (
    LOBSequence,
    NormalizationStats,
    compute_basic_tick_features,
)
logger = logging.getLogger(__name__)
FI2010_K_LEVELS = 10
FI2010_F_LEVEL = 4
FI2010_F_TICK = 6
FI2010_FEATURE_DIM = FI2010_K_LEVELS * FI2010_F_LEVEL + FI2010_F_TICK
LEVEL_FEATURE_NAMES_FI2010 = ("ask_price", "ask_size", "bid_price", "bid_size")
TICK_FEATURE_NAMES_FI2010 = (
    "mid", "spread", "log_spread", "imbalance", "microprice", "log_total_vol",
)
FLAT_FEATURE_NAMES_FI2010 = (
    tuple(
        f"level{k}.{name}"
        for k in range(FI2010_K_LEVELS)
        for name in LEVEL_FEATURE_NAMES_FI2010
    )
    + tuple(f"tick.{name}" for name in TICK_FEATURE_NAMES_FI2010)
)
# FI-2010 label encoding from the published files: 1 = up, 2 = stationary, 3 = down.
# Our direction head expects {0=down, 1=flat, 2=up} so we remap on load.
_LABEL_ROW_BY_HORIZON = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
FILENAME_BY_SPLIT = {
    "train": "Train_Dst_NoAuction_ZScore_CF_7.txt",
    "validation": "Test_Dst_NoAuction_ZScore_CF_7.txt",
}


@dataclass
class FI2010Sequence:
    sequence: LOBSequence
    direction_labels: np.ndarray  # Shape (T,), int64 in {0, 1, 2}.
    horizon: int


def _load_raw_matrix(path: Path) -> np.ndarray:
    # FI-2010 files are space-separated 149 x N text. np.loadtxt handles either
    # transposed orientation; we normalize to (N_events, 149) below.
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"FI-2010 file {path} has unexpected ndim={arr.ndim}")
    if arr.shape[0] == 149:
        arr = arr.T
    if arr.shape[1] != 149:
        raise ValueError(
            f"FI-2010 file {path} must have 149 rows or columns; got shape {arr.shape}"
        )
    return arr


def _remap_labels(raw: np.ndarray) -> np.ndarray:
    # Published encoding is {1=up, 2=stationary, 3=down}; map to {0=down, 1=flat, 2=up}.
    rounded = np.rint(raw).astype(np.int64)
    out = np.full(rounded.shape, 1, dtype=np.int64)
    out[rounded == 1] = 2
    out[rounded == 3] = 0
    return out


def load_fi2010_split(
    data_dir: Path,
    split: str,
    horizon: int = 10,
    clip_value: float = 8.0,
) -> FI2010Sequence:
    """Load one FI-2010 split as an LOBSequence plus direction labels.

    The level features come straight from the pre-normalized FI-2010 file; the
    six per-tick aggregates are derived on the fly via compute_basic_tick_features
    so downstream code does not have to special-case the dataset.
    """
    if split not in FILENAME_BY_SPLIT:
        raise ValueError(f"split must be one of {sorted(FILENAME_BY_SPLIT)}, got {split!r}")
    if horizon not in _LABEL_ROW_BY_HORIZON:
        raise ValueError(
            f"horizon must be one of {sorted(_LABEL_ROW_BY_HORIZON)}, got {horizon}"
        )
    path = Path(data_dir) / FILENAME_BY_SPLIT[split]
    if not path.exists():
        raise FileNotFoundError(
            f"FI-2010 file not found: {path}. Download the NoAuction ZScore CF "
            "files from the FairData Etsin release (Ntakaris et al. 2018) or "
            "the project HF Hub mirror."
        )
    logger.info(f"loading FI-2010 split={split} horizon={horizon} from {path}")
    matrix = _load_raw_matrix(path)
    n_events = matrix.shape[0]
    lob_flat = matrix[:, :40]
    per_level = lob_flat.reshape(n_events, FI2010_K_LEVELS, FI2010_F_LEVEL).astype(np.float32)
    per_tick = compute_basic_tick_features(per_level)
    # Clip both tensors to the same window used for Polymarket features.
    per_level = np.clip(per_level, -clip_value, clip_value).astype(np.float32)
    per_tick = np.clip(per_tick, -clip_value, clip_value).astype(np.float32)
    if not np.isfinite(per_level).all() or not np.isfinite(per_tick).all():
        raise ValueError(f"Non-finite values after clipping in {path}")
    label_row = _LABEL_ROW_BY_HORIZON[horizon]
    direction_labels = _remap_labels(matrix[:, label_row])
    # Synthetic midprice for downstream metrics: average of best ask and best bid
    # in z-scored space. Not a true price, just a stable reference signal.
    mid_norm = 0.5 * (per_level[:, 0, 0] + per_level[:, 0, 2])
    sequence = LOBSequence(
        market_slug=f"fi2010_h{horizon}",
        per_level=per_level,
        per_tick=per_tick,
        midprice=mid_norm.astype(np.float32),
        ts_sec=np.arange(n_events, dtype=np.int64),
        yes_outcome=None,
    )
    logger.info(
        f"fi2010 {split}: {n_events} events, per_level={per_level.shape}, "
        f"per_tick={per_tick.shape}, labels {{0,1,2}} -> "
        f"{np.bincount(direction_labels, minlength=3).tolist()}"
    )
    return FI2010Sequence(sequence=sequence, direction_labels=direction_labels, horizon=horizon)


def build_identity_normalization() -> NormalizationStats:
    """Identity normalization stats for the pre-normalized FI-2010 files.

    apply_normalization() expects mean and std arrays; passing zeros and ones
    leaves the data untouched so the same downstream code path works.
    """
    return NormalizationStats(
        per_level_mean=np.zeros(FI2010_F_LEVEL, dtype=np.float32),
        per_level_std=np.ones(FI2010_F_LEVEL, dtype=np.float32),
        per_tick_mean=np.zeros(FI2010_F_TICK, dtype=np.float32),
        per_tick_std=np.ones(FI2010_F_TICK, dtype=np.float32),
        clip_value=8.0,
    )

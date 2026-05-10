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
from envs.lob_features import LOBSequence, compute_basic_tick_features
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
# Candidate filenames in priority order. The first one that exists wins.
# DecPre is the decimal-precision normalized variant (used by DeepLOB's vendored
# data.zip); ZScore is the per-stock per-day z-scored variant from the FairData
# release. Either works because the trainer fits its own normalization stats.
FILENAMES_BY_SPLIT = {
    "train": (
        "Train_Dst_NoAuction_DecPre_CF_7.txt",
        "Train_Dst_NoAuction_ZScore_CF_7.txt",
    ),
    "validation": (
        "Test_Dst_NoAuction_DecPre_CF_7.txt",
        "Test_Dst_NoAuction_ZScore_CF_7.txt",
    ),
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


def _resolve_split_path(data_dir: Path, split: str) -> Path:
    candidates = FILENAMES_BY_SPLIT[split]
    for name in candidates:
        path = Path(data_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"FI-2010 file not found under {data_dir}. Looked for: {', '.join(candidates)}. "
        "Download from the FairData Etsin release (Ntakaris et al. 2018) or the "
        "project HF Hub mirror at ruuudy/FinDrama under data/fi2010/<split>/."
    )


def load_fi2010_split(
    data_dir: Path,
    split: str,
    horizon: int = 10,
    max_events: int | None = None,
) -> FI2010Sequence:
    """Load one FI-2010 split as an LOBSequence plus direction labels.

    Returns raw (un-normalized) per_level and per_tick arrays so the trainer
    can fit its own z-score stats. The DecPre variant is decimal-scaled; the
    ZScore variant is already per-stock-day z-scored - both work because
    fit_normalization handles either input.
    """
    if split not in FILENAMES_BY_SPLIT:
        raise ValueError(f"split must be one of {sorted(FILENAMES_BY_SPLIT)}, got {split!r}")
    if horizon not in _LABEL_ROW_BY_HORIZON:
        raise ValueError(
            f"horizon must be one of {sorted(_LABEL_ROW_BY_HORIZON)}, got {horizon}"
        )
    path = _resolve_split_path(Path(data_dir), split)
    logger.info(f"loading FI-2010 split={split} horizon={horizon} from {path.name}")
    matrix = _load_raw_matrix(path)
    n_events = matrix.shape[0]
    if max_events is not None and 0 < max_events < n_events:
        # Keep the most recent slice; LOB statistics drift, and the tail is
        # closest to the validation split.
        matrix = matrix[-max_events:]
        n_events = matrix.shape[0]
    lob_flat = matrix[:, :40]
    per_level = lob_flat.reshape(n_events, FI2010_K_LEVELS, FI2010_F_LEVEL).astype(np.float32)
    per_tick = compute_basic_tick_features(per_level)
    if not np.isfinite(per_level).all() or not np.isfinite(per_tick).all():
        raise ValueError(f"Non-finite values in {path}")
    label_row = _LABEL_ROW_BY_HORIZON[horizon]
    direction_labels = _remap_labels(matrix[:, label_row])
    # Synthetic midprice for downstream metrics: average of best ask and best bid.
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

"""The class-balanced direction gap came back positive while FiLM was inert, so the suspicion is that
the "robustness" is a label-distribution artifact: if the high-vol windows simply carry more up/down
ticks, an over-predicting balanced head scores a higher macro-F1 there regardless of conditioning.
This dumps the next-tick label balance inside the low-vol vs high-vol windows to test that directly.
"""
# region imports
from pathlib import Path
import numpy as np
from finmamba3.envs.fi2010_loader import load_fi2010_split
from finmamba3.envs.lob_features import apply_normalization, load_normalization
from finmamba3.eval.compare_direction import _label_directions
from finmamba3.eval.regime_split import window_volatility_split
# endregion


def _regime_label_fractions(midprice: np.ndarray, windows: list, level_width: int) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    for start, end in windows:
        labels = _label_directions(midprice[start:end], threshold=0.0)
        counts += np.bincount(labels[labels >= 0], minlength=3)
    return counts


def main() -> int:
    bundle = load_fi2010_split(Path("data/fi2010/validation"), split="validation", horizon=10)
    stats = load_normalization(Path("saved_models/lob/fi2010_norm_real.json"))
    val_norm = apply_normalization(bundle.sequence, stats)
    split = window_volatility_split(val_norm.midprice, window_len=512, quantile=0.5)
    # Labels use the normalized flat mid channel (tick mid = level_width + 0), exactly as _global_labels.
    mid = val_norm.to_flat()[:, 40]
    low_counts = _regime_label_fractions(mid, split.low_vol_windows, level_width=40)
    high_counts = _regime_label_fractions(mid, split.high_vol_windows, level_width=40)
    print(f"low-vol  label counts (down/flat/up) = {low_counts.tolist()}  frac = {(low_counts / low_counts.sum()).round(4).tolist()}")
    print(f"high-vol label counts (down/flat/up) = {high_counts.tolist()}  frac = {(high_counts / high_counts.sum()).round(4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

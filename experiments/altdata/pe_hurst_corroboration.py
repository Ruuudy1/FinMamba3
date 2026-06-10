"""Corroborate the Efficiency-Ratio regime split with permutation entropy and Hurst (goal: 'ER primary;
permutation entropy / Hurst reported'). For each dataset's val stream, split windows at the ER median and
report mean ER / PE / Hurst of the reference (high-ER) vs shifted (low-ER) buckets. The reference bucket
should be more forecastable: higher ER, lower permutation entropy, and a Hurst farther from the 0.5
random-walk value.
"""
# region imports
import numpy as np
from pathlib import Path
from finmamba3.envs.kaggle_lob_loader import load_kaggle_lob
from finmamba3.envs.fi2010_loader import load_fi2010_split
from finmamba3.eval.predictability import (
    efficiency_ratio,
    hurst_exponent,
    permutation_entropy,
    window_predictability_split,
)
# endregion


def bucket_stats(mid, windows):
    ers = np.array([efficiency_ratio(mid[s:e]) for s, e in windows])
    pes = np.array([permutation_entropy(mid[s:e]) for s, e in windows])
    hursts = np.array([hurst_exponent(mid[s:e]) for s, e in windows])
    return ers.mean(), pes.mean(), hursts.mean(), len(windows)


def report(name, mid, window_len):
    split = window_predictability_split(mid, window_len, 0.5)
    ref = bucket_stats(mid, split.reference_windows)
    shi = bucket_stats(mid, split.shifted_windows)
    print(f"=== {name} ({split.description}) ===")
    print(f"  reference (high-ER, n={ref[3]}): ER={ref[0]:.4f}  PE={ref[1]:.4f}  Hurst={ref[2]:.4f}")
    print(f"  shifted   (low-ER,  n={shi[3]}): ER={shi[0]:.4f}  PE={shi[1]:.4f}  Hurst={shi[2]:.4f}")
    print(f"  contrast: dER={ref[0]-shi[0]:+.4f}  dPE={ref[1]-shi[1]:+.4f} (ref should be lower)  "
          f"dHurst={ref[2]-shi[2]:+.4f}")


def main():
    kaggle = load_kaggle_lob(Path("data/BTC_1min.csv"), "BTC", "1min", max_rows=14400)
    kaggle_val_mid = kaggle.sequence.midprice[10080:14400]
    report("Kaggle BTC 1min val", kaggle_val_mid, 128)
    fi = load_fi2010_split(Path("data/fi2010/validation"), "validation", 10)
    report("FI-2010 val", fi.sequence.midprice, 512)


if __name__ == "__main__":
    main()

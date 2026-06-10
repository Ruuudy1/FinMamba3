"""Aggregate per-seed generalization gaps into mean +/- sd, SE, one-sample t, p, and a 95% CI.

Usage: python gap_stats.py <label> <gap_seed0> <gap_seed1> [<gap_seed2> ...]
The null hypothesis is gap == 0 (FiLM no more robust than baseline); a CI excluding 0 with a
consistent sign is the WIN condition's statistical half (the FiLM-activation half is judged separately).
"""
# region imports
import sys
import numpy as np
from scipy import stats
# endregion


def main() -> int:
    label = sys.argv[1]
    gaps = np.array([float(x) for x in sys.argv[2:]], dtype=np.float64)
    n = gaps.shape[0]
    mean = float(gaps.mean())
    sd = float(gaps.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    sign_pattern = "".join("+" if g > 0 else ("-" if g < 0 else "0") for g in gaps)
    if n > 1 and se > 0:
        t_stat = mean / se
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low, ci_high = mean - crit * se, mean + crit * se
        excludes_zero = ci_low > 0.0 or ci_high < 0.0
    else:
        t_stat = p_value = ci_low = ci_high = float("nan")
        excludes_zero = False
    print(f"[{label}] n={n} gaps={gaps.tolist()}")
    print(f"  mean={mean:+.5f} sd={sd:.5f} SE={se:.5f} sign={sign_pattern}")
    print(f"  t={t_stat:+.4f} p={p_value:.4f} 95%CI=[{ci_low:+.5f}, {ci_high:+.5f}] excludes_0={excludes_zero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

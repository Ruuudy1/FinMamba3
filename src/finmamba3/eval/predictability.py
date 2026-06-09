"""Predictability regime axis (newgoal-2 Phase 0.7).

The campaign's whole problem was finding a regime axis a FiLM router has a genuine reason to
encode. "How forecastable is this window" is exactly that latent: a selectively-betting policy
profits by trading the predictable windows and sitting out the random-walk ones. These three
cheap estimators on the underlying spot series quantify it; the Efficiency Ratio is the most
interpretable scalar and is the primary split axis, with permutation entropy and the Hurst
exponent as corroborating measures. Thresholds are fit on the train split, never hardcoded.
"""
# region imports
from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
from finmamba3.backtester.strategy import MarketLifecycle
from finmamba3.eval.regime_split import market_spot_series
# endregion
if TYPE_CHECKING:
    from finmamba3.backtester.data_loader import TickData


def efficiency_ratio(series: np.ndarray) -> float:
    """Kaufman efficiency ratio: net move over total path length, in [0, 1].

    Near 1 means a clean directional trend (the net displacement used the whole path); near 0
    means the path wandered back on itself (a random or mean-reverting window). It is the single
    most interpretable predictability scalar, so the per-market split is taken on it first.
    """
    spot = np.asarray(series, dtype=np.float64)
    if spot.shape[0] < 2:
        return 0.0
    total_path = float(np.abs(np.diff(spot)).sum())
    if total_path <= 0.0:
        return 0.0
    return float(abs(spot[-1] - spot[0]) / total_path)


def permutation_entropy(series: np.ndarray, order: int = 4) -> float:
    """Normalized permutation entropy of the ordinal patterns of length `order`, in [0, 1].

    Near 0 means one ordinal pattern dominates (a structured, predictable series); near 1 means
    every pattern is equally likely (a maximally random series). Normalizing by log(order!) makes
    it comparable across windows. A series shorter than order+1 has no pattern and returns 0.
    """
    spot = np.asarray(series, dtype=np.float64)
    num_values = spot.shape[0]
    if num_values < order + 1:
        return 0.0
    count_by_pattern: dict[tuple, int] = {}
    for start in range(num_values - order + 1):
        pattern = tuple(np.argsort(spot[start : start + order]).tolist())
        count_by_pattern[pattern] = count_by_pattern.get(pattern, 0) + 1
    counts = np.asarray(list(count_by_pattern.values()), dtype=np.float64)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / math.log(math.factorial(order))


def hurst_exponent(series: np.ndarray) -> float:
    """Hurst exponent via the structure-function (variance-of-lagged-differences) estimator.

    The std of lag-k differences scales as lag^H, so the slope of log(std) against log(lag) is H.
    H > 0.5 is trending (persistent), H < 0.5 mean-reverting (anti-persistent), H ~ 0.5 a random
    walk. Crypto at fine timeframes tends to run mean-reverting. A short window defaults to 0.5
    (random walk) so the estimator is always finite.
    """
    spot = np.asarray(series, dtype=np.float64)
    num_values = spot.shape[0]
    if num_values < 20:
        return 0.5
    lags = np.arange(2, min(num_values // 2, 20))
    tau = np.asarray([float(np.std(spot[lag:] - spot[:-lag])) for lag in lags], dtype=np.float64)
    positive = tau > 0.0
    if int(positive.sum()) < 2:
        return 0.5
    slope = np.polyfit(np.log(lags[positive]), np.log(tau[positive]), 1)[0]
    return float(slope)


_METRIC_BY_NAME = {
    "efficiency_ratio": efficiency_ratio,
    "permutation_entropy": permutation_entropy,
    "hurst_exponent": hurst_exponent,
}


def predictability_from_timeline(
    timeline: list["TickData"],
    lifecycles: list[MarketLifecycle],
    metric: str = "efficiency_ratio",
) -> dict[str, float]:
    """Per-market predictability scalar over each market's underlying spot window.

    The default Efficiency Ratio is the primary split axis: bucketing markets at the train-fit ER
    median separates the forecastable (high-ER) regime, where the model should trade, from the
    random-walk (low-ER) regime, where it should sit out.
    """
    if metric not in _METRIC_BY_NAME:
        raise ValueError(f"Unknown predictability metric {metric!r}; expected one of {sorted(_METRIC_BY_NAME)}.")
    metric_func = _METRIC_BY_NAME[metric]
    score_by_slug: dict[str, float] = {}
    for slug, series in market_spot_series(timeline, lifecycles).items():
        score_by_slug[slug] = metric_func(series)
    return score_by_slug

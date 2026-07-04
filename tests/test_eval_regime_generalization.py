"""Unit tests for the cross-regime generalization-gap evaluator.

Model-free: covers the volatility slug partition (with the min-ticks qualifier), the
gap sign convention, the markdown verdict, and the per-market sequence bridge. The
same-checkpoint gap-approx-0 plumbing check runs on the GPU in the A/B step, not here,
because the world model's forward pass uses CUDA autocast.
"""
# region imports
from __future__ import annotations
import numpy as np
import pytest
from finmamba3.backtester.data_loader import BacktestData, TickData
from finmamba3.backtester.strategy import MarketLifecycle, OrderBookLevel, OrderBookSnapshot, StoredBook
from finmamba3.envs.lob_features import extract_features, fit_normalization
from finmamba3.eval.eval_regime_generalization import (
    _degradation,
    _format_table,
    _regime_sequences,
    _split_slugs,
    generalization_gap,
)
from finmamba3.eval.regime_split import realized_vol_from_timeline
# endregion
_CALM = "btc-updown-5m-0"
_WILD = "btc-updown-5m-300"
_CALM_MIDS = [0.50, 0.50, 0.51, 0.50, 0.51, 0.50, 0.50, 0.51]
_WILD_MIDS = [0.50, 0.70, 0.40, 0.80, 0.30, 0.75, 0.35, 0.70]


def _book(mid):
    return OrderBookSnapshot(bids=(OrderBookLevel(mid - 0.01, 100.0),), asks=(OrderBookLevel(mid + 0.01, 100.0),))


def _stored(mid, book_ts):
    return StoredBook(yes_book=_book(mid), no_book=_book(1.0 - mid), book_ts=book_ts)


def _two_market_bt():
    ticks = []
    for ts in range(len(_CALM_MIDS)):
        tick = TickData(ts_sec=ts)
        tick.order_books[_CALM] = _stored(_CALM_MIDS[ts], ts)
        tick.order_books[_WILD] = _stored(_WILD_MIDS[ts], ts)
        ticks.append(tick)
    return BacktestData(
        timeline=ticks,
        lifecycles=[
            MarketLifecycle(_CALM, "5m", start_ts=0, end_ts=300),
            MarketLifecycle(_WILD, "5m", start_ts=300, end_ts=600),
        ],
        settlements={},
        start_ts=0,
        end_ts=len(_CALM_MIDS) - 1,
    )


def test_split_slugs_routes_high_vol_to_the_shifted_regime():
    bt = _two_market_bt()
    realized_vol = realized_vol_from_timeline(bt.timeline)
    low_slugs, high_slugs, desc = _split_slugs(bt, realized_vol, max_markets=10, min_ticks=2, quantile=0.5)
    assert low_slugs == [_CALM]
    assert high_slugs == [_WILD]
    assert desc.startswith("vol>")


def test_split_slugs_drops_markets_below_min_ticks():
    bt = _two_market_bt()
    realized_vol = realized_vol_from_timeline(bt.timeline)
    # Requiring more than the eight ticks each synthetic market has must fail fast, not return a silently empty split.
    with pytest.raises(RuntimeError):
        _split_slugs(bt, realized_vol, max_markets=10, min_ticks=100, quantile=0.5)


def test_generalization_gap_positive_when_treatment_degrades_less():
    # Baseline loses 0.30 macro-F1 across the regime shift; the treatment loses only 0.10.
    gap = generalization_gap(degradation_baseline=0.30, degradation_treatment=0.10)
    assert gap == pytest.approx(0.20)
    assert gap > 0


def test_generalization_gap_negative_when_treatment_degrades_more():
    gap = generalization_gap(degradation_baseline=0.05, degradation_treatment=0.20)
    assert gap < 0


def test_format_table_reports_supported_verdict_for_positive_gap():
    rows = [
        {"arm": "baseline", "low": 0.60, "high": 0.30, "degradation": 0.30},
        {"arm": "treatment", "low": 0.58, "high": 0.48, "degradation": 0.10},
    ]
    table = _format_table(rows, gap_value=0.20, regime_desc="vol>0.12345", metric="direction_macro_f1")
    assert "thesis supported" in table
    assert "+0.20000" in table
    assert "vol>0.12345" in table


def test_format_table_reports_not_robust_verdict_for_negative_gap():
    rows = [
        {"arm": "baseline", "low": 0.60, "high": 0.55, "degradation": 0.05},
        {"arm": "treatment", "low": 0.58, "high": 0.38, "degradation": 0.20},
    ]
    table = _format_table(rows, gap_value=-0.15, regime_desc="vol>0.1", metric="prediction_mse")
    assert "not more robust" in table
    assert "-0.15000" in table


def test_degradation_sign_is_consistent_across_metric_directions():
    # Macro-F1 is higher-is-better: dropping from 0.6 to 0.4 is a +0.2 degradation.
    assert _degradation(0.60, 0.40, higher_is_better=True) == pytest.approx(0.20)
    # Prediction MSE is lower-is-better: rising from 0.40 to 0.60 is a +0.2 degradation.
    assert _degradation(0.40, 0.60, higher_is_better=False) == pytest.approx(0.20)
    # A model that improves under the shift has negative degradation either way.
    assert _degradation(0.40, 0.60, higher_is_better=True) == pytest.approx(-0.20)


def test_regime_sequences_builds_one_normalized_sequence_per_slug():
    bt = _two_market_bt()
    # Fit normalization on the calm market so apply_normalization sees matching widths.
    stats = fit_normalization(extract_features(bt.timeline, _CALM))
    seqs = _regime_sequences(bt, [_CALM, _WILD], stats, aggregate_only=False, include_binary_features=False)
    assert len(seqs) == 2
    assert [s.market_slug for s in seqs] == [_CALM, _WILD]
    for seq in seqs:
        assert seq.to_flat().shape[0] == len(_CALM_MIDS)
        assert np.isfinite(seq.to_flat()).all()

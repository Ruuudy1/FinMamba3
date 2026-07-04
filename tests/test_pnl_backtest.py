"""Phase 1: WorldModelStrategy turns a divergence between the model's YES probability and the book
into an order, and is a no-op when they agree. The execution engine is vendored in this repo, so the
engine and its dataclasses import directly from finmamba3.backtester.
"""
# region imports
from finmamba3.backtester.engine import BacktestEngine
from finmamba3.backtester.strategy import MarketState, MarketView, OrderBookLevel, OrderBookSnapshot, PositionView, Side, Token
from finmamba3.backtester import BacktestData, OrderBookLevel as RepoLevel, OrderBookSnapshot as RepoBook, StoredBook, TickData
from finmamba3.backtester.strategy import MarketLifecycle, Settlement
from finmamba3.backtester.strategy import Token as RepoToken
from finmamba3.eval.pnl_backtest import (
    NaiveLagStrategy,
    WorldModelStrategy,
    _data_for_slugs,
    bootstrap_survivability,
    bootstrap_survivability_by_market,
    delay_prob_series,
    per_trade_pnls,
    per_trade_pnls_by_market,
    settlement_calibration,
)
from finmamba3.eval.strategies import CusumGate
from finmamba3.eval.survivability import kelly_shares
# endregion
_SLUG = "btc-updown-5m-0"


def _book(mid):
    return OrderBookSnapshot(
        bids=(OrderBookLevel(round(mid - 0.01, 4), 100.0),),
        asks=(OrderBookLevel(round(mid + 0.01, 4), 100.0),),
    )


def _state(yes_mid, ts=100, positions=None, spot=0.0):
    view = MarketView(
        market_slug=_SLUG, interval="5m", start_ts=0, end_ts=300,
        time_remaining_s=150.0, time_remaining_frac=0.5,
        yes_book=_book(yes_mid), no_book=_book(1.0 - yes_mid),
    )
    return MarketState(
        timestamp=ts, timestamp_utc="", markets={_SLUG: view},
        chainlink_btc=spot, cash=10_000.0, positions=positions or {}, total_portfolio_value=10_000.0,
    )


def test_buys_yes_when_model_above_book():
    strat = WorldModelStrategy({(_SLUG, 100): 0.60}, edge_threshold=0.05, order_size=50.0)
    orders = strat.on_tick(_state(0.40))
    assert len(orders) == 1
    assert orders[0].token == Token.YES
    assert orders[0].side == Side.BUY
    assert orders[0].size == 50.0


def test_buys_no_when_model_below_book():
    strat = WorldModelStrategy({(_SLUG, 100): 0.20}, edge_threshold=0.05, order_size=50.0)
    orders = strat.on_tick(_state(0.40))
    assert len(orders) == 1
    assert orders[0].token == Token.NO
    assert orders[0].side == Side.BUY


def test_no_trade_when_model_agrees_with_book():
    strat = WorldModelStrategy({(_SLUG, 100): 0.42}, edge_threshold=0.05, order_size=50.0)
    assert strat.on_tick(_state(0.40)) == []


def test_no_trade_without_a_probability():
    strat = WorldModelStrategy({}, edge_threshold=0.05)
    assert strat.on_tick(_state(0.40)) == []


def test_position_limit_blocks_at_cap_and_clips_below_it():
    strat = WorldModelStrategy({(_SLUG, 100): 0.90}, edge_threshold=0.05, order_size=50.0)
    # At the 500-share cap there is no remaining capacity, so no order is emitted.
    at_cap = {_SLUG: PositionView(market_slug=_SLUG, yes_shares=500.0)}
    assert strat.on_tick(_state(0.40, positions=at_cap)) == []
    # With 480 held the fixed 50 clips to the remaining 20 shares rather than being dropped.
    near_cap = {_SLUG: PositionView(market_slug=_SLUG, yes_shares=480.0)}
    orders = strat.on_tick(_state(0.40, positions=near_cap))
    assert len(orders) == 1
    assert orders[0].size == 20.0


def test_max_tte_frac_gate_skips_early_ticks():
    strat = WorldModelStrategy({(_SLUG, 100): 0.90}, edge_threshold=0.05, max_tte_frac=0.3)
    # With time_remaining_frac 0.5 > 0.3, the near-expiry gate skips this early, low-edge tick.
    assert strat.on_tick(_state(0.40)) == []


def _repo_stored(yes_mid):
    return StoredBook(
        yes_book=RepoBook(bids=(RepoLevel(round(yes_mid - 0.01, 4), 100.0),), asks=(RepoLevel(round(yes_mid + 0.01, 4), 100.0),)),
        no_book=RepoBook(bids=(RepoLevel(round(0.99 - yes_mid, 4), 100.0),), asks=(RepoLevel(round(1.01 - yes_mid, 4), 100.0),)),
        book_ts=0,
    )


def test_engine_settles_a_winning_yes_trade():
    # A tiny repo timeline drives the vendored engine directly (no converter): a YES contract that settles YES, with the model forced 0.9 vs a 0.40 book, must fill cheap YES and book a positive PnL.
    timeline = []
    for ts in range(7):
        tick = TickData(ts_sec=ts, btc_mid=50_000.0, chainlink_btc=50_000.0)
        tick.order_books[_SLUG] = _repo_stored(0.40)
        tick.book_timestamps[_SLUG] = ts
        timeline.append(tick)
    bt = BacktestData(
        timeline=timeline,
        lifecycles=[MarketLifecycle(_SLUG, "5m", start_ts=0, end_ts=5)],
        settlements={_SLUG: Settlement(_SLUG, "5m", RepoToken.YES, 0, 5, 50_000.0, 50_500.0)},
        start_ts=0,
        end_ts=6,
    )
    prob_by_slug_ts = {(_SLUG, ts): 0.90 for ts in range(7)}
    strat = WorldModelStrategy(prob_by_slug_ts, edge_threshold=0.05, order_size=50.0)
    engine_data = _data_for_slugs(bt, [_SLUG])
    result = BacktestEngine(engine_data, strat, starting_cash=10_000.0, snapshot_interval=1).run()
    assert result.total_trades >= 1
    assert result.total_settlements == 1
    assert result.total_pnl > 0.0
    # Per-trade PnL of a winning YES buy is size * (1 - avg_price) > 0, the bootstrap's input.
    trade_pnls = per_trade_pnls(result)
    assert trade_pnls and all(pnl > 0.0 for pnl in trade_pnls)


def test_cusum_fires_on_drift_and_is_silent_on_noise():
    quiet = CusumGate(threshold=0.003)
    noise_events = [quiet.update(r) for r in [0.0005, -0.0005] * 6]
    assert all(event == 0 for event in noise_events)
    drift = CusumGate(threshold=0.003)
    drift_events = [drift.update(0.0015) for _ in range(4)]
    assert 1 in drift_events


def test_kelly_shares_formula_cap_and_zero():
    bankroll, price = 10_000.0, 0.5
    # Uncapped quarter-Kelly of a modest YES edge equals 0.25 * (p_model-p_market)/(1-p_market) * bankroll / price.
    edge_fraction = (0.55 - 0.5) / (1.0 - 0.5)
    shares = kelly_shares(0.55, 0.5, Token.YES, bankroll, price, 0.25, 0.05)
    assert abs(shares - 0.25 * edge_fraction * bankroll / price) < 1e-6
    # A large edge clamps to the 5% cap.
    capped = kelly_shares(0.99, 0.5, Token.YES, bankroll, price, 0.25, 0.05)
    assert abs(capped - 0.05 * bankroll / price) < 1e-6
    # No edge or an adverse edge sizes to zero.
    assert kelly_shares(0.5, 0.5, Token.YES, bankroll, price, 0.25, 0.05) == 0.0
    assert kelly_shares(0.4, 0.5, Token.YES, bankroll, price, 0.25, 0.05) == 0.0


def test_naive_lag_emits_on_spot_move_and_noop_when_flat():
    mover = NaiveLagStrategy(cusum_threshold=0.003, order_size=50.0)
    spot = 50_000.0
    emitted = []
    for _ in range(6):
        spot *= 1.0015
        emitted += mover.on_tick(_state(0.40, spot=spot))
    assert emitted
    assert emitted[0].token == Token.YES and emitted[0].side == Side.BUY
    flat = NaiveLagStrategy(cusum_threshold=0.003, order_size=50.0)
    assert all(flat.on_tick(_state(0.40, spot=50_000.0)) == [] for _ in range(6))


def test_naive_predictability_gate_buys_the_spot_trend_direction():
    # With the same gate as the model but no model, the naive buys the observable trend: up -> YES, down -> NO.
    up = NaiveLagStrategy(use_predictability_gate=True, predictability_threshold=0.5, predictability_window=10)
    for t in range(12):
        up.on_tick(_state(0.40, ts=t, spot=50_000.0 + 10.0 * t))
    up_orders = up.on_tick(_state(0.40, ts=12, spot=50_130.0))
    assert up_orders and up_orders[0].token == Token.YES
    down = NaiveLagStrategy(use_predictability_gate=True, predictability_threshold=0.5, predictability_window=10)
    for t in range(12):
        down.on_tick(_state(0.60, ts=t, spot=50_000.0 - 10.0 * t))
    down_orders = down.on_tick(_state(0.60, ts=12, spot=49_870.0))
    assert down_orders and down_orders[0].token == Token.NO


def test_settlement_calibration_brier_and_reliability():
    yes_slug, no_slug = "btc-updown-5m-0", "btc-updown-5m-300"
    bt = BacktestData(
        timeline=[],
        lifecycles=[MarketLifecycle(yes_slug, "5m", 0, 300), MarketLifecycle(no_slug, "5m", 300, 600)],
        settlements={yes_slug: Settlement(yes_slug, "5m", RepoToken.YES, 0, 300), no_slug: Settlement(no_slug, "5m", RepoToken.NO, 300, 600)},
        start_ts=0, end_ts=600,
    )
    # A calibrated head: ~0.9 on the YES-settling market, ~0.1 on the NO-settling one -> low Brier.
    probs = {(yes_slug, t): 0.9 for t in range(50)}
    probs.update({(no_slug, 1000 + t): 0.1 for t in range(50)})
    cal = settlement_calibration(probs, bt)
    assert cal["n"] == 100
    assert cal["brier"] < 0.02
    by_bin = {b["bin"]: b for b in cal["bins"]}
    assert by_bin["0.9-1.0"]["realized_yes"] == 1.0
    assert by_bin["0.1-0.2"]["realized_yes"] == 0.0


def test_bootstrap_constant_positive_trades_are_fully_survivable():
    result = bootstrap_survivability([5.0, 5.0, 5.0, 5.0], bankroll=10_000.0, n_paths=1000, seed=0)
    assert result["frac_profitable"] == 1.0
    assert result["drawdown_p95"] == 0.0
    assert result["n_trades"] == 4


def test_per_trade_pnls_by_market_groups_fills_by_slug():
    # Two markets settle opposite ways; grouping must keep each market's fills together for block resampling.
    timeline = []
    other = "btc-updown-5m-300"
    for ts in range(7):
        tick = TickData(ts_sec=ts, btc_mid=50_000.0, chainlink_btc=50_000.0)
        tick.order_books[_SLUG] = _repo_stored(0.40)
        tick.order_books[other] = _repo_stored(0.40)
        tick.book_timestamps[_SLUG] = ts
        tick.book_timestamps[other] = ts
        timeline.append(tick)
    bt = BacktestData(
        timeline=timeline,
        lifecycles=[MarketLifecycle(_SLUG, "5m", 0, 5), MarketLifecycle(other, "5m", 0, 5)],
        settlements={
            _SLUG: Settlement(_SLUG, "5m", RepoToken.YES, 0, 5, 50_000.0, 50_500.0),
            other: Settlement(other, "5m", RepoToken.NO, 0, 5, 50_000.0, 49_500.0),
        },
        start_ts=0, end_ts=6,
    )
    probs = {(_SLUG, ts): 0.90 for ts in range(7)}
    probs.update({(other, ts): 0.90 for ts in range(7)})
    strat = WorldModelStrategy(probs, edge_threshold=0.05, order_size=50.0)
    result = BacktestEngine(_data_for_slugs(bt, [_SLUG, other]), strat, starting_cash=10_000.0, snapshot_interval=1).run()
    by_market = per_trade_pnls_by_market(result)
    # Both markets traded; the winning-YES market's fills are positive, the NO-settling market's negative.
    assert sorted(by_market.keys()) == sorted([_SLUG, other])
    assert all(pnl > 0.0 for pnl in by_market[_SLUG])
    assert all(pnl < 0.0 for pnl in by_market[other])
    # The flattened by-market PnLs match the ungrouped per-trade list.
    assert sorted(p for v in by_market.values() for p in v) == sorted(per_trade_pnls(result))


def test_market_block_bootstrap_counts_independent_markets_not_trades():
    # Ten trades inside ONE market are one independent bet: a losing market is never profitable, however many correlated trades it holds, where the iid bootstrap would see ten "independent" losers.
    pnls_by_market = {"m0": [-3.0] * 10}
    block = bootstrap_survivability_by_market(pnls_by_market, bankroll=10_000.0, n_paths=500, seed=0)
    assert block["n_markets"] == 1
    assert block["n_trades"] == 10
    assert block["frac_profitable"] == 0.0
    # A balanced book of one reliably-winning and one reliably-losing market straddles zero.
    mixed = bootstrap_survivability_by_market({"win": [4.0, 4.0], "lose": [-4.0, -4.0]}, n_paths=2000, seed=1)
    assert mixed["n_markets"] == 2
    assert 0.0 < mixed["frac_profitable"] < 1.0


def test_signal_delay_shifts_probabilities_later_in_time():
    probs = {(_SLUG, 100): 0.7, (_SLUG, 160): 0.8}
    delayed = delay_prob_series(probs, 30)
    assert delayed == {(_SLUG, 130): 0.7, (_SLUG, 190): 0.8}
    # A non-positive delay is the identity (same object), so the no-delay path costs nothing.
    assert delay_prob_series(probs, 0) is probs


def test_predictability_gate_blocks_chop_allows_trend():
    probs = {(_SLUG, 999): 0.90}
    # A clean uptrend fills the window with a high causal ER, so the gate lets the divergence trade.
    trend = WorldModelStrategy(probs, edge_threshold=0.05, use_predictability_gate=True, predictability_threshold=0.5, predictability_window=10)
    for t in range(12):
        trend.on_tick(_state(0.40, ts=t, spot=50_000.0 + 10.0 * t))
    assert trend.on_tick(_state(0.40, ts=999, spot=50_120.0))
    # A zigzag fills the window with a near-zero ER, so the gate sits the strategy out of the chop.
    chop = WorldModelStrategy(probs, edge_threshold=0.05, use_predictability_gate=True, predictability_threshold=0.5, predictability_window=10)
    for t in range(12):
        chop.on_tick(_state(0.40, ts=t, spot=50_000.0 + (50.0 if t % 2 else -50.0)))
    assert chop.on_tick(_state(0.40, ts=999, spot=50_000.0)) == []


def test_min_book_depth_gate_blocks_shallow_books():
    # The synthetic book has 100 + 100 = 200 two-sided depth, so a 1000 floor blocks it, a 100 floor lets it through.
    probs = {(_SLUG, 100): 0.90}
    assert WorldModelStrategy(probs, edge_threshold=0.05, min_book_depth=1000.0).on_tick(_state(0.40)) == []
    assert WorldModelStrategy(probs, edge_threshold=0.05, min_book_depth=100.0).on_tick(_state(0.40))


def test_naive_precomputed_gate_and_direction_per_market():
    # With precomputed per-market ER + direction the naive gates and follows each asset's own trend.
    down = NaiveLagStrategy(use_predictability_gate=True, predictability_threshold=0.5,
                            gate_er_by_slug_ts={(_SLUG, 100): 0.9}, gate_dir_by_slug_ts={(_SLUG, 100): -1.0})
    down_orders = down.on_tick(_state(0.60))
    assert down_orders and down_orders[0].token == Token.NO
    closed = NaiveLagStrategy(use_predictability_gate=True, predictability_threshold=0.5,
                              gate_er_by_slug_ts={(_SLUG, 100): 0.2}, gate_dir_by_slug_ts={(_SLUG, 100): 1.0})
    assert closed.on_tick(_state(0.40)) == []


def test_precomputed_gate_er_gates_per_market():
    # A precomputed per-market ER (asset-correct, enables multi-asset) drives the gate by lookup.
    probs = {(_SLUG, 100): 0.90}
    sits_out = WorldModelStrategy(probs, edge_threshold=0.05, use_predictability_gate=True,
                                  predictability_threshold=0.5, gate_er_by_slug_ts={(_SLUG, 100): 0.20})
    assert sits_out.on_tick(_state(0.40)) == []
    trades = WorldModelStrategy(probs, edge_threshold=0.05, use_predictability_gate=True,
                                predictability_threshold=0.5, gate_er_by_slug_ts={(_SLUG, 100): 0.90})
    assert trades.on_tick(_state(0.40))


def test_calibration_temperature_softens_overconfident_probability():
    # T = 1 leaves the over-confident 0.90 alone, so the 0.50 edge over a 0.40 book clears a 0.45 gate.
    raw = WorldModelStrategy({(_SLUG, 100): 0.90}, edge_threshold=0.45, calibration_temperature=1.0)
    assert raw.on_tick(_state(0.40))
    # T = 5 pulls 0.90 toward ~0.61, so the edge drops below 0.45 and the strategy sits out.
    calibrated = WorldModelStrategy({(_SLUG, 100): 0.90}, edge_threshold=0.45, calibration_temperature=5.0)
    assert calibrated.on_tick(_state(0.40)) == []


# Edge mode (--prob-source edge) tests.
def test_edge_mode_buys_yes_when_ev_yes_clears_threshold_and_dominates():
    # With ev_yes_hat=0.10 > threshold=0.05 AND ev_yes > ev_no → BUY_YES.
    ev = {(_SLUG, 100): (0.10, -0.05)}
    strat = WorldModelStrategy({}, edge_threshold=0.05, order_size=50.0, ev_by_slug_ts=ev)
    orders = strat.on_tick(_state(0.40))
    assert len(orders) == 1
    assert orders[0].token == Token.YES
    assert orders[0].side == Side.BUY
    assert orders[0].size == 50.0


def test_edge_mode_buys_no_when_ev_no_clears_threshold_and_dominates():
    # With ev_no_hat=0.10 > threshold=0.05 AND ev_no > ev_yes → BUY_NO.
    ev = {(_SLUG, 100): (-0.05, 0.10)}
    strat = WorldModelStrategy({}, edge_threshold=0.05, order_size=50.0, ev_by_slug_ts=ev)
    orders = strat.on_tick(_state(0.40))
    assert len(orders) == 1
    assert orders[0].token == Token.NO


def test_edge_mode_sits_when_neither_ev_clears_threshold():
    # Both EVs below 0.05 threshold → SIT.
    ev = {(_SLUG, 100): (0.02, 0.01)}
    strat = WorldModelStrategy({}, edge_threshold=0.05, order_size=50.0, ev_by_slug_ts=ev)
    assert strat.on_tick(_state(0.40)) == []


def test_edge_mode_sits_on_tied_ev():
    # Equal EVs both above threshold: neither dominates → SIT.
    ev = {(_SLUG, 100): (0.10, 0.10)}
    strat = WorldModelStrategy({}, edge_threshold=0.05, order_size=50.0, ev_by_slug_ts=ev)
    assert strat.on_tick(_state(0.40)) == []


def test_edge_mode_no_trade_without_ev_entry():
    # Missing (slug, ts) pair → no order.
    strat = WorldModelStrategy({}, edge_threshold=0.05, ev_by_slug_ts={})
    assert strat.on_tick(_state(0.40)) == []


def test_settlement_path_unchanged_when_ev_by_slug_ts_is_none():
    # With ev_by_slug_ts=None the existing settlement path runs unmodified.
    probs = {(_SLUG, 100): 0.80}
    strat = WorldModelStrategy(probs, edge_threshold=0.05, order_size=50.0, ev_by_slug_ts=None)
    orders = strat.on_tick(_state(0.40))
    assert len(orders) == 1
    assert orders[0].token == Token.YES


def test_world_model_kelly_sizing_scales_with_edge():
    strat = WorldModelStrategy({(_SLUG, 100): 0.90}, edge_threshold=0.05, sizing="kelly", bankroll=10_000.0)
    orders = strat.on_tick(_state(0.40))
    # Kelly sizing replaces the fixed 50 with an edge-proportional, capped share count.
    assert len(orders) == 1
    assert orders[0].token == Token.YES
    assert orders[0].size > 0.0
    assert orders[0].size != 50.0

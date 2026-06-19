"""PnL backtest adapter: judge a frozen world model by simulated trading PnL, not a forecasting proxy.

This module is the one place the torch world model meets the execution engine, now vendored in this
repo as ``finmamba3.backtester.engine`` and consuming this repo's timeline dataclasses directly. The
frozen model's spot-conditioned settlement YES probability is precomputed with this repo's exact
training feature pipeline, so the probability the strategy trades on is the one the model was trained
to produce (no train/serve skew). A thin WorldModelStrategy turns a divergence from the book's implied
probability into an order, which the engine matches T+1 and settles for the headline PnL / Sharpe per
spot-volatility regime.
"""
# region imports
from __future__ import annotations
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import yaml
from finmamba3.backtester import build_timeline
from finmamba3.backtester.data_loader import BacktestData, _asset_from_slug
from finmamba3.backtester.engine import BacktestEngine, MAX_SHARES_PER_TOKEN
from finmamba3.backtester.engine_cpp import (
    STRATEGY_NAIVE_LAG, STRATEGY_WORLD_MODEL, align_run_signals, marshal_timeline, run_marshalled,
)
from finmamba3.backtester.strategy import BaseStrategy, MarketLifecycle, Order, Settlement, Side, Token
from finmamba3.config import DotDict, parse_args_and_update_config
from finmamba3.envs.lob_features import apply_normalization, extract_features, load_normalization
from finmamba3.eval.compare_direction import load_world_model
from finmamba3.eval.predictability import efficiency_ratio, predictability_from_timeline
from finmamba3.eval.regime_split import spot_realized_vol_from_timeline, volatility_split
from finmamba3.sequence_builder import _append_cross_interval, _settlement_yes_outcome, _spot_feature_kwargs
# endregion
logger = logging.getLogger(__name__)


class CusumGate:
    """Symmetric CUSUM filter on the spot return (Lopez de Prado event sampling).

    Accumulates signed returns and fires +1 / -1 when the running positive / negative sum crosses
    the threshold, then resets that arm. This turns a per-second stream into a handful of genuine
    directional events per session, so the strategy participates selectively instead of trading
    forecasting noise. The threshold is a hyperparameter fit on the train split.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.pos = 0.0
        self.neg = 0.0
    def update(self, spot_return: float) -> int:
        self.pos = max(0.0, self.pos + spot_return)
        self.neg = min(0.0, self.neg + spot_return)
        if self.pos >= self.threshold:
            self.pos = 0.0
            return 1
        if self.neg <= -self.threshold:
            self.neg = 0.0
            return -1
        return 0


def kelly_shares(
    model_prob: float, book_prob: float, side: "Token", bankroll: float, price: float,
    kelly_fraction: float, cap_fraction: float,
) -> float:
    """Quarter-Kelly share size for one binary-contract trade, capped and zero with no edge.

    For YES the edge-over-breakeven fraction is (p_model - p_market) / (1 - p_market); for NO it is
    (p_market - p_model) / p_market. A non-positive edge sizes to zero (do not trade against the
    model). The bet is kelly_fraction of that edge, capped at cap_fraction of bankroll, so sizing
    scales with conviction while the convex downside stays bounded.
    """
    if side == Token.YES:
        edge_fraction = (model_prob - book_prob) / max(1.0 - book_prob, 1e-6)
    else:
        edge_fraction = (book_prob - model_prob) / max(book_prob, 1e-6)
    if edge_fraction <= 0.0:
        return 0.0
    bet_fraction = min(kelly_fraction * edge_fraction, cap_fraction)
    return (bet_fraction * bankroll) / max(price, 1e-6)


def per_trade_pnls(result, slippage_per_share: float = 0.0) -> list[float]:
    """Realized per-trade PnL for every fill, from the bought token and the market's settlement.

    The strategy only ever buys, so a fill of `size` at `avg_price` returns size * (1 - avg_price)
    when its token matches the settled outcome and -size * avg_price otherwise. This per-trade list
    is the input to the fat-tailed bootstrap, which the portfolio-level PnL alone cannot supply.
    slippage_per_share subtracts a per-share execution cost on top of the walk-the-book fill, a
    deployability stress test for the liquidity the backtest assumes is takeable.
    """
    outcome_by_slug = {settlement.market_slug: settlement.outcome for settlement in result.settlements}
    pnls = []
    for fill in result.fills:
        outcome = outcome_by_slug.get(fill.market_slug)
        if outcome is None:
            continue
        payout = 1.0 if fill.token == outcome else 0.0
        pnls.append(float(fill.size * (payout - fill.avg_price) - slippage_per_share * fill.size))
    return pnls


def bootstrap_survivability(trade_pnls: list[float], bankroll: float = 10_000.0, n_paths: int = 10_000, seed: int = 0) -> dict:
    """Monte Carlo bootstrap over the per-trade PnL list (the fat-tail-robust headline test).

    Resamples the realized trades with replacement into n_paths equity curves of the same length,
    and reports the fraction of profitable paths, the worst-5%-tail max drawdown as a fraction of
    bankroll, and the 95% CI on terminal PnL. Trade returns are non-normal, so this replaces the
    Gaussian t-test as the survivability gate. This treats every trade as an independent draw; when
    trades within a market are correlated, bootstrap_survivability_by_market is the honest test.
    """
    if not trade_pnls:
        return {"n_trades": 0, "frac_profitable": 0.0, "drawdown_p95": 0.0, "ci_low": 0.0, "ci_high": 0.0, "mean_terminal": 0.0}
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(pnls, size=(n_paths, pnls.shape[0]), replace=True)
    equity = bankroll + np.cumsum(sampled, axis=1)
    terminal = equity[:, -1] - bankroll
    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdown = (running_peak - equity).max(axis=1) / bankroll
    return {
        "n_trades": int(pnls.shape[0]),
        "frac_profitable": float(np.mean(terminal > 0.0)),
        "drawdown_p95": float(np.percentile(drawdown, 95.0)),
        "ci_low": float(np.percentile(terminal, 2.5)),
        "ci_high": float(np.percentile(terminal, 97.5)),
        "mean_terminal": float(np.mean(terminal)),
    }


def per_trade_pnls_by_market(result, slippage_per_share: float = 0.0) -> dict:
    """Realized per-trade PnL grouped by market slug, the input to the market-block bootstrap.

    Trades inside one market share its single settlement outcome and sit on overlapping holding
    windows, so they are not independent draws; grouping them lets the block bootstrap resample a
    market's entire trade vector at once and respect that dependence. Same per-fill PnL convention as
    per_trade_pnls (the strategy only ever buys), keyed by market_slug.
    """
    outcome_by_slug = {settlement.market_slug: settlement.outcome for settlement in result.settlements}
    pnls_by_slug = {}
    for fill in result.fills:
        outcome = outcome_by_slug.get(fill.market_slug)
        if outcome is None:
            continue
        payout = 1.0 if fill.token == outcome else 0.0
        pnls_by_slug.setdefault(fill.market_slug, []).append(float(fill.size * (payout - fill.avg_price) - slippage_per_share * fill.size))
    return pnls_by_slug


def bootstrap_survivability_by_market(pnls_by_market: dict, bankroll: float = 10_000.0, n_paths: int = 10_000, seed: int = 0) -> dict:
    """Market-block bootstrap: resample whole markets with replacement, not individual trades.

    Each path draws as many markets as were traded (with replacement), concatenates their trade
    vectors in a random order, and scores terminal PnL and worst-case drawdown on that equity curve.
    Because a market's trades move together exactly as they did in reality, the effective sample size
    is the number of independent markets, not the (far larger) number of correlated trades, so this is
    the honest survivability gate; the iid bootstrap_survivability overstates P(profit) when intra-market
    trades are correlated. Drawdown is reduced per market in O(1) from each market's (total, low, high,
    intrinsic-drawdown) summary, so 10k paths over hundreds of markets stay cheap. n_markets is the true
    independent-unit count the report should cite alongside n_trades.
    """
    slugs = [slug for slug in pnls_by_market.keys() if pnls_by_market[slug]]
    if not slugs:
        return {"n_markets": 0, "n_trades": 0, "frac_profitable": 0.0, "drawdown_p95": 0.0, "ci_low": 0.0, "ci_high": 0.0, "mean_terminal": 0.0}
    cumulatives = [np.cumsum(np.asarray(pnls_by_market[slug], dtype=np.float64)) for slug in slugs]
    totals = np.array([float(c[-1]) for c in cumulatives])
    lows = np.array([float(c.min()) for c in cumulatives])
    highs = np.array([float(c.max()) for c in cumulatives])
    intrinsic_dd = np.array([float((np.maximum.accumulate(c) - c).max()) for c in cumulatives])
    n_markets = len(slugs)
    n_trades = int(sum(c.shape[0] for c in cumulatives))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_markets, size=(n_paths, n_markets))
    drawn_totals = totals[draws]
    terminal = drawn_totals.sum(axis=1)
    # Worst-case drawdown over each path's equity curve, vectorized over all paths at once. The per-path
    # Python loop this replaces was ~98% of a sweep value's wall-clock and held the GIL (so a threaded
    # sweep could not parallelize it); the numpy reductions below drop the GIL. The equity before market i
    # is a bankroll-seeded left-to-right cumsum, whose float association is identical to the scalar
    # `prefix += totals[market]` accumulation, so the drawdown is bitwise-identical to the old loop.
    equity_before = np.cumsum(np.concatenate([np.full((n_paths, 1), bankroll), drawn_totals[:, :-1]], axis=1), axis=1)
    equity_high = equity_before + highs[draws]
    # Peak before market i is the inclusive running peak shifted one market right, floored at bankroll.
    running_peak = np.maximum.accumulate(equity_high, axis=1)
    peak_before = np.empty_like(equity_high)
    peak_before[:, 0] = bankroll
    peak_before[:, 1:] = np.maximum(bankroll, running_peak[:, :-1])
    dip = np.maximum(intrinsic_dd[draws], peak_before - (equity_before + lows[draws]))
    drawdown = dip.max(axis=1) / bankroll
    return {
        "n_markets": n_markets,
        "n_trades": n_trades,
        "frac_profitable": float(np.mean(terminal > 0.0)),
        "drawdown_p95": float(np.percentile(drawdown, 95.0)),
        "ci_low": float(np.percentile(terminal, 2.5)),
        "ci_high": float(np.percentile(terminal, 97.5)),
        "mean_terminal": float(np.mean(terminal)),
    }


class WorldModelStrategy(BaseStrategy):
    """Buys the under-priced token when the model's YES probability diverges from the book.

    prob_by_slug_ts maps (market_slug, tick_second) to the frozen model's spot-conditioned
    settlement YES probability. On each tick, for every active market with a fresh probability,
    the strategy compares it to the book's implied YES probability (the YES-book mid). When the
    divergence exceeds edge_threshold it buys YES (model higher than book) or NO (model lower), at
    a fixed size and within the engine's 500-share / no-short / cash limits. A no-op when the model
    agrees with the book is the correct behaviour: the tradable edge is only where they diverge.
    """

    def __init__(
        self,
        prob_by_slug_ts: dict,
        edge_threshold: float = 0.05,
        order_size: float = 50.0,
        max_tte_frac: float = 1.0,
        min_tte_frac: float = 0.0,
        use_cusum: bool = False,
        cusum_threshold: float = 0.003,
        sizing: str = "fixed",
        kelly_fraction: float = 0.25,
        kelly_cap: float = 0.05,
        bankroll: float = 10_000.0,
        use_predictability_gate: bool = False,
        predictability_threshold: float = 0.3,
        predictability_window: int = 120,
        calibration_temperature: float = 1.0,
        gate_er_by_slug_ts: dict | None = None,
        min_book_depth: float = 0.0,
        max_book_depth: float = 0.0,
        ev_by_slug_ts: dict | None = None,
    ) -> None:
        self.prob_by_slug_ts = prob_by_slug_ts
        # When ev_by_slug_ts is provided, the strategy trades from predicted book-relative EV directly
        # rather than from probability divergence. All other gates are unchanged so the comparison is fair.
        self.ev_by_slug_ts = ev_by_slug_ts
        # Optional book-depth band: trade only when two-sided depth is within [min, max]. The min floor
        # tests asset-identity vs depth; the max ceiling (0 = none) excludes the most-liquid/efficient
        # books where the within-BTC probe found the oracle-lag edge reverses, so the pair trades the
        # liquidity *band* the campaign identified rather than just "deep enough".
        self.min_book_depth = float(min_book_depth)
        self.max_book_depth = float(max_book_depth)
        # When provided, the gate reads each market's asset-correct precomputed ER per tick, so the
        # strategy generalizes beyond BTC; when None it falls back to the live BTC spot-window below.
        self.gate_er_by_slug_ts = gate_er_by_slug_ts
        # Post-hoc temperature scaling of the (over-confident) settlement probability; T > 1 softens it
        # toward 0.5. Fit on the train split and frozen, it is what makes Kelly sizing viable.
        self.calibration_temperature = float(calibration_temperature)
        self.edge_threshold = float(edge_threshold)
        self.order_size = float(order_size)
        self.max_tte_frac = float(max_tte_frac)
        # min_tte_frac excludes near-expiry trades (where the book has converged to the outcome and the
        # oracle-lag edge reverses); the edge concentrates early, when the book is least informed.
        self.min_tte_frac = float(min_tte_frac)
        self.use_cusum = bool(use_cusum)
        self.sizing = sizing
        self.kelly_fraction = float(kelly_fraction)
        self.kelly_cap = float(kelly_cap)
        self.bankroll = float(bankroll)
        self.cusum = CusumGate(cusum_threshold)
        self.prev_spot = None
        # The causal predictability gate is the convex selective-participation mechanism: trade only when
        # the underlying spot has been trending recently (rolling efficiency ratio above threshold), so the
        # strategy sits out random-walk windows where the model's edge is absent. It uses spot-so-far only.
        self.use_predictability_gate = bool(use_predictability_gate)
        self.predictability_threshold = float(predictability_threshold)
        self.predictability_window = int(predictability_window)
        self.spot_window = []
    def _size(self, model_prob: float, book_prob: float, side, price: float) -> float:
        if self.sizing == "kelly":
            return kelly_shares(model_prob, book_prob, side, self.bankroll, price, self.kelly_fraction, self.kelly_cap)
        return self.order_size
    def on_tick(self, state) -> list:
        spot = state.chainlink_btc
        event = 0
        if self.prev_spot is not None and self.prev_spot > 0.0 and spot > 0.0:
            event = self.cusum.update((spot - self.prev_spot) / self.prev_spot)
        self.prev_spot = spot
        if self.use_predictability_gate and self.gate_er_by_slug_ts is None and spot > 0.0:
            self.spot_window.append(spot)
            if len(self.spot_window) > self.predictability_window:
                self.spot_window.pop(0)
        # The CUSUM event gate restricts trading to genuine spot moves; without it the strategy acts
        # on every tick with an edge, which the convexity frame treats as forecasting noise.
        if self.use_cusum and event == 0:
            return []
        # The asset-level live gate (BTC spot-window) applies only when no per-market ER was precomputed;
        # with a precompute the gate is applied per market below so each asset gates on its own trend.
        if self.use_predictability_gate and self.gate_er_by_slug_ts is None and len(self.spot_window) >= 2 and efficiency_ratio(np.asarray(self.spot_window, dtype=np.float64)) < self.predictability_threshold:
            return []
        orders = []
        for slug, view in state.markets.items():
            if self.use_predictability_gate and self.gate_er_by_slug_ts is not None and self.gate_er_by_slug_ts.get((slug, state.timestamp), 0.0) < self.predictability_threshold:
                continue
            if self.min_book_depth > 0.0 or self.max_book_depth > 0.0:
                book_depth = sum(level.size for level in view.yes_book.bids) + sum(level.size for level in view.yes_book.asks)
                if book_depth < self.min_book_depth:
                    continue
                if self.max_book_depth > 0.0 and book_depth > self.max_book_depth:
                    continue
            if view.time_remaining_frac > self.max_tte_frac or view.time_remaining_frac < self.min_tte_frac:
                continue
            position = state.positions.get(slug)
            held_yes = position.yes_shares if position is not None else 0.0
            held_no = position.no_shares if position is not None else 0.0
            if self.ev_by_slug_ts is not None:
                # Edge mode: trade directly from predicted book-relative EV.
                ev_pair = self.ev_by_slug_ts.get((slug, state.timestamp))
                if ev_pair is None:
                    continue
                ev_yes_hat, ev_no_hat = ev_pair
                if ev_yes_hat >= self.edge_threshold and ev_yes_hat > ev_no_hat and view.yes_book.best_ask > 0.0:
                    size = min(self.order_size, MAX_SHARES_PER_TOKEN - held_yes)
                    if size > 0.0:
                        orders.append(Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=size))
                elif ev_no_hat >= self.edge_threshold and ev_no_hat > ev_yes_hat and view.no_book.best_ask > 0.0:
                    size = min(self.order_size, MAX_SHARES_PER_TOKEN - held_no)
                    if size > 0.0:
                        orders.append(Order(market_slug=slug, token=Token.NO, side=Side.BUY, size=size))
            else:
                # Settlement mode: trade from probability divergence (existing path).
                model_prob = self.prob_by_slug_ts.get((slug, state.timestamp))
                if model_prob is None:
                    continue
                if self.calibration_temperature != 1.0:
                    clipped = min(max(model_prob, 1e-6), 1.0 - 1e-6)
                    logit = float(np.log(clipped / (1.0 - clipped)))
                    model_prob = float(1.0 / (1.0 + np.exp(-logit / self.calibration_temperature)))
                book_prob = view.yes_book.mid
                if book_prob <= 0.0 or book_prob >= 1.0:
                    continue
                edge = model_prob - book_prob
                if abs(edge) < self.edge_threshold:
                    continue
                if edge > 0.0 and view.yes_book.best_ask > 0.0:
                    # Clip the (possibly Kelly-sized) order to the remaining 500-share capacity rather than
                    # dropping it, so a conviction size larger than the cap still trades up to the limit.
                    size = min(self._size(model_prob, book_prob, Token.YES, view.yes_book.best_ask), MAX_SHARES_PER_TOKEN - held_yes)
                    if size > 0.0:
                        orders.append(Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=size))
                elif edge < 0.0 and view.no_book.best_ask > 0.0:
                    size = min(self._size(model_prob, book_prob, Token.NO, view.no_book.best_ask), MAX_SHARES_PER_TOKEN - held_no)
                    if size > 0.0:
                        orders.append(Order(market_slug=slug, token=Token.NO, side=Side.BUY, size=size))
        return orders


class NaiveLagStrategy(BaseStrategy):
    """Latency-arbitrage baseline (newgoal-2 1d): no world model, trades the CUSUM spot-move direction.

    On a CUSUM event it buys the token the spot just moved toward (up -> YES, down -> NO) at a fixed
    size, but only while the book has not yet repriced (the implied probability is still on the wrong
    side of 0.5 for the move). The world-model strategy must beat THIS to show its edge is the model
    rather than the mechanical oracle lag; if it does not, the positive PnL is latency arbitrage and
    the architecture result is a null, the economic analogue of the film_g gate.
    """

    def __init__(self, cusum_threshold: float = 0.003, order_size: float = 50.0, max_tte_frac: float = 1.0,
                 use_predictability_gate: bool = False, predictability_threshold: float = 0.3, predictability_window: int = 120,
                 gate_er_by_slug_ts: dict | None = None, gate_dir_by_slug_ts: dict | None = None,
                 min_book_depth: float = 0.0, max_book_depth: float = 0.0) -> None:
        self.order_size = float(order_size)
        self.max_tte_frac = float(max_tte_frac)
        self.cusum = CusumGate(cusum_threshold)
        self.prev_spot = None
        # The same book-depth band as the world model, so the comparison is fair: both trade only within
        # [min, max] depth, isolating whether the model's edge over trend-following survives in the band.
        self.min_book_depth = float(min_book_depth)
        self.max_book_depth = float(max_book_depth)
        # The predictability-gate mode makes the naive a *fair* baseline for the gated world-model
        # strategy: it uses the identical selective gate and buys the observable spot-trend direction
        # with no model, so the world model must beat it to show its calibrated probability adds value
        # over a dumb "buy the trend in trending markets" rule.
        self.use_predictability_gate = bool(use_predictability_gate)
        self.predictability_threshold = float(predictability_threshold)
        self.predictability_window = int(predictability_window)
        self.spot_window = []
        # When provided, the gate and trend direction are read per market from each asset's own spot, so
        # the baseline is asset-correct across BTC/ETH/SOL; when None it falls back to the live BTC path.
        self.gate_er_by_slug_ts = gate_er_by_slug_ts
        self.gate_dir_by_slug_ts = gate_dir_by_slug_ts
    def on_tick(self, state) -> list:
        spot = state.chainlink_btc
        event = 0
        if self.prev_spot is not None and self.prev_spot > 0.0 and spot > 0.0:
            event = self.cusum.update((spot - self.prev_spot) / self.prev_spot)
        self.prev_spot = spot
        live_direction = 0
        if self.use_predictability_gate and self.gate_er_by_slug_ts is None:
            if spot > 0.0:
                self.spot_window.append(spot)
                if len(self.spot_window) > self.predictability_window:
                    self.spot_window.pop(0)
            if len(self.spot_window) < 2 or efficiency_ratio(np.asarray(self.spot_window, dtype=np.float64)) < self.predictability_threshold:
                return []
            live_direction = 1 if self.spot_window[-1] >= self.spot_window[0] else -1
        elif not self.use_predictability_gate:
            if event == 0:
                return []
            live_direction = event
        orders = []
        for slug, view in state.markets.items():
            if self.use_predictability_gate and self.gate_er_by_slug_ts is not None:
                if self.gate_er_by_slug_ts.get((slug, state.timestamp), 0.0) < self.predictability_threshold:
                    continue
                direction = 1 if self.gate_dir_by_slug_ts.get((slug, state.timestamp), 1.0) >= 0.0 else -1
            else:
                direction = live_direction
            if self.min_book_depth > 0.0 or self.max_book_depth > 0.0:
                book_depth = sum(level.size for level in view.yes_book.bids) + sum(level.size for level in view.yes_book.asks)
                if book_depth < self.min_book_depth or (self.max_book_depth > 0.0 and book_depth > self.max_book_depth):
                    continue
            if view.time_remaining_frac > self.max_tte_frac:
                continue
            book_prob = view.yes_book.mid
            if book_prob <= 0.0 or book_prob >= 1.0:
                continue
            position = state.positions.get(slug)
            held_yes = position.yes_shares if position is not None else 0.0
            held_no = position.no_shares if position is not None else 0.0
            if direction > 0 and book_prob < 0.5 and view.yes_book.best_ask > 0.0 and held_yes + self.order_size <= MAX_SHARES_PER_TOKEN:
                orders.append(Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=self.order_size))
            elif direction < 0 and book_prob > 0.5 and view.no_book.best_ask > 0.0 and held_no + self.order_size <= MAX_SHARES_PER_TOKEN:
                orders.append(Order(market_slug=slug, token=Token.NO, side=Side.BUY, size=self.order_size))
        return orders


def world_model_yes_prob_series(wm, seq, device: torch.device, window_len: int = 64, chunk: int = 512,
                                sample_mode: str = "random_sample") -> dict:
    """Per-tick settlement YES probability for one market, keyed by tick second.

    For each tick with at least window_len preceding context, the settlement head reads the causal
    window ending at that tick and the sigmoid of its last-position logit is the spot-conditioned YES
    probability. Reuses the exact encode -> sequence -> settlement_head path the forecasting evaluator
    scores, so train and serve see one forward. Windows are batched in chunks to bound GPU memory.
    sample_mode='probs' takes the deterministic latent (no random sample), making the PnL reproducible.
    """
    flat = seq.to_flat()
    total_ticks = flat.shape[0]
    prob_by_ts = {}
    if total_ticks < window_len:
        return prob_by_ts
    starts = np.arange(0, total_ticks - window_len + 1)
    for chunk_start in range(0, len(starts), chunk):
        batch_starts = starts[chunk_start : chunk_start + chunk]
        windows = np.stack([flat[s : s + window_len] for s in batch_starts], axis=0)
        obs = torch.from_numpy(windows).float().to(device)
        action = torch.zeros((obs.shape[0], window_len), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
            embedding = wm.encoder(obs)
            post_logits = wm.dist_head.forward_post(embedding)
            sample = wm.straight_through_gradient(post_logits, sample_mode=sample_mode)
            flattened_sample = wm.flatten_sample(sample)
            if wm.model == "Transformer":
                from finmamba3.models.attention import get_subsequent_mask_with_batch_length
                mask = get_subsequent_mask_with_batch_length(window_len, flattened_sample.device)
                dist_feat = wm.sequence_model(flattened_sample, action, mask)
            else:
                dist_feat = wm.sequence_model(flattened_sample, action)
            dist_feat = wm.condition_dist_feat(dist_feat)
            settle_logits = wm.settlement_head(dist_feat).float()
        last_prob = torch.sigmoid(settle_logits[:, -1]).cpu().numpy()
        for i, start in enumerate(batch_starts):
            prob_by_ts[int(seq.ts_sec[start + window_len - 1])] = float(last_prob[i])
    return prob_by_ts


def world_model_edge_ev_series(wm, seq, device: torch.device, window_len: int = 64, chunk: int = 512,
                               sample_mode: str = "random_sample") -> dict:
    """Per-tick (ev_yes_hat, ev_no_hat) for one market, keyed by tick second.

    Mirrors world_model_yes_prob_series() exactly except the edge head replaces the settlement head.
    Returns an empty dict when the market is too short or the model has no edge head.
    """
    if not wm.use_edge_head or wm.edge_head is None:
        return {}
    flat = seq.to_flat()
    total_ticks = flat.shape[0]
    ev_by_ts = {}
    if total_ticks < window_len:
        return ev_by_ts
    starts = np.arange(0, total_ticks - window_len + 1)
    for chunk_start in range(0, len(starts), chunk):
        batch_starts = starts[chunk_start : chunk_start + chunk]
        windows = np.stack([flat[s : s + window_len] for s in batch_starts], axis=0)
        obs = torch.from_numpy(windows).float().to(device)
        action = torch.zeros((obs.shape[0], window_len), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=wm.use_amp):
            embedding = wm.encoder(obs)
            post_logits = wm.dist_head.forward_post(embedding)
            sample = wm.straight_through_gradient(post_logits, sample_mode=sample_mode)
            flattened_sample = wm.flatten_sample(sample)
            if wm.model == "Transformer":
                from finmamba3.models.attention import get_subsequent_mask_with_batch_length
                mask_t = get_subsequent_mask_with_batch_length(window_len, flattened_sample.device)
                dist_feat = wm.sequence_model(flattened_sample, action, mask_t)
            else:
                dist_feat = wm.sequence_model(flattened_sample, action)
            dist_feat = wm.condition_dist_feat(dist_feat)
            ev_hat = wm.edge_head(dist_feat).float()
        last_ev = ev_hat[:, -1].cpu().numpy()
        for i, start in enumerate(batch_starts):
            ts = int(seq.ts_sec[start + window_len - 1])
            ev_by_ts[ts] = (float(last_ev[i, 0]), float(last_ev[i, 1]))
    return ev_by_ts


def _causal_er_series(spot: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing efficiency ratio of a dense spot series at every index, shape (n,) in [0, 1].

    The net move over the trailing window divided by its total path length; computed from a prefix sum
    of absolute increments so each index is O(1). This is the asset-level predictability the gate reads,
    precomputed so the strategy can gate any asset's markets (the engine's MarketState carries only BTC).
    """
    spot = np.asarray(spot, dtype=np.float64)
    num = spot.shape[0]
    abs_increment = np.abs(np.diff(spot, prepend=spot[:1]))
    cumulative_path = np.concatenate([[0.0], np.cumsum(abs_increment)])
    er = np.zeros(num, dtype=np.float64)
    for t in range(num):
        lo = max(0, t - window + 1)
        path = cumulative_path[t + 1] - cumulative_path[lo]
        er[t] = abs(spot[t] - spot[lo]) / (path + 1e-8)
    return er


def _causal_net_sign_series(spot: np.ndarray, window: int) -> np.ndarray:
    """Sign of the trailing-window net move at every index (+1 up, -1 down), shape (n,), causal.

    The direction a no-model trend-follower would take; precomputed per asset so the naive baseline is
    asset-correct (a SOL market follows SOL's trend, not BTC's) for the cross-asset anti-artifact check.
    """
    spot = np.asarray(spot, dtype=np.float64)
    num = spot.shape[0]
    sign = np.ones(num, dtype=np.float64)
    for t in range(num):
        lo = max(0, t - window + 1)
        if spot[t] < spot[lo]:
            sign[t] = -1.0
    return sign


def _gate_signals_by_slug_ts(bt, prob_by_slug_ts: dict, window: int) -> tuple:
    """Causal gate ER and trend-direction at each (slug, tick), using the slug's own asset spot.

    Builds each asset's causal ER and net-sign series from its Chainlink spot once, then looks up the
    right asset's signals at each market tick. This makes both the gate (a SOL market gates on SOL's
    trend) and the naive direction asset-correct, so the strategy and its baseline generalize across
    assets without touching the BTC-only engine. Returns (gate_er_by_slug_ts, gate_dir_by_slug_ts).
    """
    num = len(bt.timeline)
    ts_array = np.fromiter((int(tick.ts_sec) for tick in bt.timeline), dtype=np.int64, count=num)
    btc = np.fromiter((float(tick.chainlink_btc) for tick in bt.timeline), dtype=np.float64, count=num)
    eth = np.fromiter((float(tick.chainlink_eth) for tick in bt.timeline), dtype=np.float64, count=num)
    sol = np.fromiter((float(tick.chainlink_sol) for tick in bt.timeline), dtype=np.float64, count=num)
    er_by_asset = {"BTC": _causal_er_series(btc, window), "ETH": _causal_er_series(eth, window), "SOL": _causal_er_series(sol, window)}
    sign_by_asset = {"BTC": _causal_net_sign_series(btc, window), "ETH": _causal_net_sign_series(eth, window), "SOL": _causal_net_sign_series(sol, window)}
    gate_er_by_slug_ts = {}
    gate_dir_by_slug_ts = {}
    for slug, ts in prob_by_slug_ts.keys():
        asset = _asset_from_slug(slug)
        index = min(max(int(np.searchsorted(ts_array, ts)), 0), num - 1)
        gate_er_by_slug_ts[(slug, ts)] = float(er_by_asset[asset][index])
        gate_dir_by_slug_ts[(slug, ts)] = float(sign_by_asset[asset][index])
    return gate_er_by_slug_ts, gate_dir_by_slug_ts


def _usable_tick_count_by_slug(bt) -> dict:
    # Count the ticks whose YES book has both sides, matching extract_features' usability test, so the
    # caller can drop markets that would raise "no usable ticks" or yield no scoreable window.
    count_by_slug = {}
    for tick in bt.timeline:
        for slug, stored in tick.order_books.items():
            if stored.yes_book.bids and stored.yes_book.asks:
                count_by_slug[slug] = count_by_slug.get(slug, 0) + 1
    return count_by_slug


def _eval_sequence(bt, slug, stats, include_binary: bool, include_spot: bool, include_cross: bool):
    # Build one held-out market's normalized sequence through the exact training pipeline, reusing the
    # train stats so the model is served the scale it learned on.
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    settlement = bt.settlements.get(slug)
    yes_outcome = _settlement_yes_outcome(settlement)
    spot_kwargs = _spot_feature_kwargs(slug, settlement, lifecycle_by_slug) if include_spot else {}
    raw = extract_features(
        bt.timeline, slug, yes_outcome=yes_outcome,
        include_binary_features=include_binary, include_spot_features=include_spot, **spot_kwargs,
    )
    if include_cross:
        raw = _append_cross_interval(bt, slug, raw, lifecycle_by_slug)
    return apply_normalization(raw, stats)


def _data_for_slugs(bt, slugs: list[str]) -> BacktestData:
    # The engine only tracks, trades and settles the lifecycles it is handed, so restricting them to
    # one regime's slugs scopes the run to that regime while the shared timeline carries every book.
    # The vendored engine consumes this repo's timeline dataclasses directly, so there is no per-tick
    # translation: the shared bt.timeline is reused as-is and only the lifecycles/settlements are scoped.
    wanted = {slug: True for slug in slugs}
    lifecycles = [
        MarketLifecycle(lc.market_slug, lc.interval, lc.start_ts, lc.end_ts)
        for lc in bt.lifecycles if lc.market_slug in wanted
    ]
    settlements = {}
    for slug in slugs:
        st = bt.settlements.get(slug)
        if st is not None:
            settlements[slug] = Settlement(
                st.market_slug, st.interval, st.outcome,
                st.start_ts, st.end_ts, st.chainlink_open, st.chainlink_close,
            )
    return BacktestData(
        timeline=bt.timeline, lifecycles=lifecycles, settlements=settlements,
        start_ts=bt.start_ts, end_ts=bt.end_ts,
    )


def _sharpe_from_snapshots(snapshots) -> float:
    # Per-snapshot Sharpe of the mark-to-market portfolio value; a relative number that lets the A/B
    # rank arms even though the absolute annualization is arbitrary for sub-hour markets.
    values = np.array([snap.total_value for snap in snapshots], dtype=np.float64)
    if values.size < 3:
        return 0.0
    returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
    if float(returns.std()) < 1e-12:
        return 0.0
    return float(np.sqrt(returns.size) * returns.mean() / returns.std())


def _arch_overrides(regime_film_enabled: bool, is_mimo: bool, n_layer: int | None) -> list[str]:
    overrides = [
        f"Models.WorldModel.RegimeFiLM.Enabled={'true' if regime_film_enabled else 'false'}",
        f"Models.WorldModel.Mamba3.is_mimo={'true' if is_mimo else 'false'}",
    ]
    if n_layer is not None:
        overrides.append(f"Models.WorldModel.Mamba3.n_layer={int(n_layer)}")
    return overrides


def _load_config(config_path: Path, overrides: list[str]) -> DotDict:
    with open(config_path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    return DotDict(parse_args_and_update_config(cfg_raw, argv=[f"--{kv.split('=')[0]}={kv.split('=')[1]}" for kv in overrides]))


def _cpp_params(strategy_kwargs: dict, ev_by_slug_ts: dict | None) -> dict:
    # Flatten the strategy kwargs into the flat params dict the native engine and its fail-fast guard
    # read; the guard rejects any key whose logic is not yet ported, so the cpp run cannot diverge.
    return {**strategy_kwargs, "ev_by_slug_ts": ev_by_slug_ts}


def _naive_cpp_params(order_size: float, max_tte_frac: float, cusum_threshold: float) -> dict:
    # The ungated naive baseline as the native engine reads it: fixed size on a CUSUM event, none of the
    # not-yet-ported gates engaged, so it passes the guard on the v1 native path.
    return {
        "edge_threshold": 0.0, "order_size": order_size, "max_tte_frac": max_tte_frac, "min_tte_frac": 0.0,
        "use_cusum": True, "cusum_threshold": cusum_threshold, "sizing": "fixed",
        "calibration_temperature": 1.0, "use_predictability_gate": False,
        "min_book_depth": 0.0, "max_book_depth": 0.0, "ev_by_slug_ts": None,
    }


def _run_engine_arm(
    engine: str, engine_data: BacktestData, strategy: BaseStrategy,
    marshalled, strategy_kind: int, params: dict, cash: float, signals=None,
):
    """Run one strategy arm through the Python reference engine or the native C++ engine.

    The cpp path replays the timeline marshal_timeline flattened once per regime; signals are the per-run
    cell-aligned gate arrays, which a sweep hoists out of its loop (constant gate window) and passes in,
    else run_marshalled aligns them. Both paths return the same result surface (pnl, trades, fills, snapshots).
    """
    if engine == "python":
        return BacktestEngine(engine_data, strategy, starting_cash=cash, snapshot_interval=60).run()
    return run_marshalled(marshalled, strategy_kind, params, cash, 60, signals=signals)


def _sweep_aligned(engine: str, cache: dict, marshalled, strategy_kind: int, params: dict, window: int):
    # Memoise the cpp sweep's cell-aligned signals per (window, arm): the gate ER / direction depend only
    # on the gate window, not the swept threshold, so a constant-window sweep aligns once and reuses across
    # values. The Python engine needs no alignment, so this is a no-op for it.
    if engine != "cpp":
        return None
    key = (window, strategy_kind)
    if key not in cache:
        cache[key] = align_run_signals(marshalled, strategy_kind, params)
    return cache[key]


def _engine_workers(engine: str, engine_threads: int) -> int:
    # The cpp engine releases the GIL inside its tick loop, so independent backtests run on real cores;
    # the python reference engine holds the GIL throughout, so it always stays serial. 0 = auto.
    if engine != "cpp":
        return 1
    if engine_threads > 0:
        return engine_threads
    return os.cpu_count() or 1


def _run_concurrent(tasks: list, max_workers: int) -> list:
    # Run independent zero-arg backtest tasks, preserving input order so the result equals the serial path
    # exactly. A single task or max_workers <= 1 stays on the calling thread (the python engine, and the
    # --engine-threads 1 parity escape hatch). The marshalled timeline and aligned signals each task reads
    # are immutable, and each backtest allocates its own engine state, so the runs do not interfere.
    if max_workers <= 1 or len(tasks) <= 1:
        return [task() for task in tasks]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = [pool.submit(task) for task in tasks]
        return [future.result() for future in futures]


def run_pnl_backtest(
    wm, bt, stats, slugs: list[str], device: torch.device,
    include_binary: bool, include_spot: bool, include_cross: bool,
    strategy_kwargs: dict, cash: float, prob_source: str = "settlement",
    slippage_per_share: float = 0.0, engine: str = "python", engine_threads: int = 0,
) -> dict:
    """Backtest one regime's slugs with the world-model strategy and the naive-lag baseline.

    Returns each arm's PnL/Sharpe/trades, the world model's per-trade bootstrap survivability, and
    the model-minus-naive PnL so the report can apply the anti-artifact gate (the model must beat
    the mechanical oracle-lag baseline). The engine data is built once and replayed for both arms.
    prob_source='edge' routes through the BookRelativeEdgeHead instead of the settlement head.
    slippage_per_share subtracts a per-share execution cost from each fill in the bootstrap only.
    """
    prob_by_slug_ts = {}
    ev_by_slug_ts = {}
    for slug in slugs:
        seq = _eval_sequence(bt, slug, stats, include_binary, include_spot, include_cross)
        if prob_source == "edge":
            for ts, ev_pair in world_model_edge_ev_series(wm, seq, device).items():
                ev_by_slug_ts[(slug, ts)] = ev_pair
                # prob_by_slug_ts is not used in edge mode but must be non-empty for strategy init
                prob_by_slug_ts[(slug, ts)] = 0.5
        else:
            for ts, prob in world_model_yes_prob_series(wm, seq, device).items():
                prob_by_slug_ts[(slug, ts)] = prob
    engine_data = _data_for_slugs(bt, slugs)
    include_depth = strategy_kwargs["min_book_depth"] > 0.0 or strategy_kwargs["max_book_depth"] > 0.0
    marshalled = marshal_timeline(engine_data, slugs, prob_by_slug_ts, include_depth=include_depth) if engine == "cpp" else None
    edge_signals = ev_by_slug_ts if prob_source == "edge" else None
    model_strategy = WorldModelStrategy(prob_by_slug_ts, bankroll=cash, ev_by_slug_ts=edge_signals, **strategy_kwargs)
    naive_strategy = NaiveLagStrategy(
        cusum_threshold=strategy_kwargs["cusum_threshold"],
        order_size=strategy_kwargs["order_size"],
        max_tte_frac=strategy_kwargs["max_tte_frac"],
    )
    model_params = _cpp_params(strategy_kwargs, edge_signals)
    naive_params = _naive_cpp_params(strategy_kwargs["order_size"], strategy_kwargs["max_tte_frac"], strategy_kwargs["cusum_threshold"])
    # Both arms replay the same immutable timeline independently, so run them concurrently on the cpp path.
    model_result, naive_result = _run_concurrent(
        [
            lambda: _run_engine_arm(engine, engine_data, model_strategy, marshalled, STRATEGY_WORLD_MODEL, model_params, cash),
            lambda: _run_engine_arm(engine, engine_data, naive_strategy, marshalled, STRATEGY_NAIVE_LAG, naive_params, cash),
        ],
        _engine_workers(engine, engine_threads),
    )
    bootstrap = bootstrap_survivability(per_trade_pnls(model_result, slippage_per_share), bankroll=cash)
    bootstrap_market = bootstrap_survivability_by_market(per_trade_pnls_by_market(model_result, slippage_per_share), bankroll=cash)
    return {
        "model": {
            "pnl": float(model_result.total_pnl),
            "sharpe": _sharpe_from_snapshots(model_result.portfolio_snapshots),
            "trades": int(model_result.total_trades),
            "bootstrap": bootstrap,
            "bootstrap_market": bootstrap_market,
        },
        "naive": {
            "pnl": float(naive_result.total_pnl),
            "trades": int(naive_result.total_trades),
        },
        "model_minus_naive": float(model_result.total_pnl - naive_result.total_pnl),
        "n_markets": len(slugs),
        "n_probs": len(prob_by_slug_ts),
    }


def settlement_calibration(prob_by_slug_ts: dict, bt, num_bins: int = 10) -> dict:
    """Reliability of the settlement YES probability over all precomputed (slug, ts): Brier + decile bins.

    The economic edge needs a calibrated p_model (newgoal-2 1c), so this checks whether the settlement
    head's probability matches the realized YES frequency. Brier is the mean squared error to the binary
    outcome; each bin compares the mean predicted probability to the realized YES rate, so a diagonal
    reliability curve means the head is calibrated and its divergence from the book is a real signal.
    """
    probs = []
    outcomes = []
    for (slug, _ts), prob in prob_by_slug_ts.items():
        outcome = _settlement_yes_outcome(bt.settlements.get(slug))
        if outcome is None:
            continue
        probs.append(prob)
        outcomes.append(outcome)
    if not probs:
        return {"brier": float("nan"), "n": 0, "bins": []}
    prob_array = np.asarray(probs, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    for i in range(num_bins):
        upper = prob_array <= edges[i + 1] if i == num_bins - 1 else prob_array < edges[i + 1]
        mask = (prob_array >= edges[i]) & upper
        if int(mask.sum()) > 0:
            bins.append({
                "bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_prob": float(prob_array[mask].mean()),
                "realized_yes": float(outcome_array[mask].mean()),
            })
    # Fit a temperature T minimizing the NLL of sigmoid(logit(p)/T) against the outcome, the standard
    # post-hoc calibration; T > 1 softens an over-confident head. A grid is enough at this resolution.
    clipped = np.clip(prob_array, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.5, 6.0, 56):
        calibrated = np.clip(1.0 / (1.0 + np.exp(-logit / temperature)), 1e-6, 1.0 - 1e-6)
        nll = float(-(outcome_array * np.log(calibrated) + (1.0 - outcome_array) * np.log(1.0 - calibrated)).mean())
        if nll < best_nll:
            best_nll, best_temperature = nll, float(temperature)
    return {
        "brier": float(np.mean((prob_array - outcome_array) ** 2)),
        "n": len(probs), "temperature": best_temperature, "bins": bins,
    }


def delay_prob_series(prob_by_slug_ts: dict, delay_secs: int) -> dict:
    """Shift each model probability later by delay_secs: the signal-delay latency stress test.

    The economic edge is a latency play (it harvests the book's lag behind spot), so it is maximally
    sensitive to how the model's signal is timed. Re-keying each (slug, ts) probability to (slug,
    ts + delay) makes the strategy at tick t trade on the model output from t - delay, leaving the
    causal predictability gate (computed on the original ticks) untouched and staling only the model
    signal. An edge that survives only because the signal is timestamp-aligned to the book vanishes at
    the first second of delay; a real edge degrades gracefully. delay_secs <= 0 is the identity.
    """
    if delay_secs <= 0:
        return prob_by_slug_ts
    return {(slug, ts + delay_secs): prob for (slug, ts), prob in prob_by_slug_ts.items()}


def run_threshold_sweep(
    wm, bt, stats, slugs: list[str], device: torch.device,
    include_binary: bool, include_spot: bool, include_cross: bool,
    base_kwargs: dict, values: list[float], cash: float, slippage_per_share: float = 0.0,
    sweep_param: str = "predictability", sample_mode: str = "random_sample", engine: str = "python",
    signal_delay_secs: int = 0, engine_threads: int = 0,
) -> dict:
    """Sweep one strategy hyperparameter on the full market set, precomputing the YES probs once.

    The deployable selective-participation strategy trades the whole market set (no regime split). With
    sweep_param='predictability' it fits the gate ER threshold; with 'edge' it fits the model-vs-book
    edge threshold at the gate threshold already in base_kwargs. Because the probs and the naive
    baseline are independent of these, they are computed once, so the sweep is cheap. Returns per-value
    total PnL, trade count, model-minus-naive, and bootstrap survivability for picking a train value.
    """
    prob_by_slug_ts = {}
    for slug in slugs:
        seq = _eval_sequence(bt, slug, stats, include_binary, include_spot, include_cross)
        for ts, prob in world_model_yes_prob_series(wm, seq, device, sample_mode=sample_mode).items():
            prob_by_slug_ts[(slug, ts)] = prob
    # The signal-delay stress test stales only the model probability; the gate is re-derived on the
    # delayed trade ticks below so it still reads the genuine causal ER at trade time. The native engine
    # marshals probs ahead of the loop, so delay is restricted to the python reference engine.
    assert signal_delay_secs <= 0 or engine == "python", "signal_delay_secs requires --engine python"
    traded_probs = delay_prob_series(prob_by_slug_ts, signal_delay_secs)
    engine_data = _data_for_slugs(bt, slugs)
    include_depth = base_kwargs["min_book_depth"] > 0.0 or base_kwargs["max_book_depth"] > 0.0
    marshalled = marshal_timeline(engine_data, slugs, traded_probs, include_depth=include_depth) if engine == "cpp" else None
    gate_signals_by_window: dict = {}
    aligned_by_window_arm: dict = {}
    def _row_task(value, strategy, model_params, model_signals, naive_strategy, naive_sweep_params, naive_signals):
        # Bind one sweep value's prepared inputs into a zero-arg closure so _run_concurrent can fan the
        # values across cores; the bootstrap is folded in so each task builds its whole row off-thread.
        def _task():
            model_result = _run_engine_arm(engine, engine_data, strategy, marshalled, STRATEGY_WORLD_MODEL, model_params, cash, model_signals)
            naive_result = _run_engine_arm(engine, engine_data, naive_strategy, marshalled, STRATEGY_NAIVE_LAG, naive_sweep_params, cash, naive_signals)
            trade_pnls = per_trade_pnls(model_result, slippage_per_share)
            boot = bootstrap_survivability(trade_pnls, bankroll=cash)
            boot_market = bootstrap_survivability_by_market(per_trade_pnls_by_market(model_result, slippage_per_share), bankroll=cash)
            return {
                "value": float(value),
                "sweep_param": sweep_param,
                "pnl": float(model_result.total_pnl),
                "pnl_after_slippage": float(sum(trade_pnls)),
                "trades": int(model_result.total_trades),
                "naive_pnl": float(naive_result.total_pnl),
                "naive_trades": int(naive_result.total_trades),
                "model_minus_naive": float(model_result.total_pnl - naive_result.total_pnl),
                "frac_profitable": boot["frac_profitable"],
                "drawdown_p95": boot["drawdown_p95"],
                "n_markets_traded": boot_market["n_markets"],
                "frac_profitable_market": boot_market["frac_profitable"],
                "drawdown_p95_market": boot_market["drawdown_p95"],
                "ci_low_market": boot_market["ci_low"],
            }
        return _task
    tasks = []
    for value in values:
        kwargs = dict(base_kwargs)
        kwargs["use_predictability_gate"] = True
        if sweep_param == "edge":
            kwargs["edge_threshold"] = value
        elif sweep_param == "window":
            kwargs["predictability_window"] = int(value)
        else:
            kwargs["predictability_threshold"] = value
        window = int(kwargs["predictability_window"])
        # The gate signals depend only on the window, so memoise them here (serially, before the parallel
        # fan-out below): a threshold or edge sweep computes them once and the cpp arms align them once too.
        if window not in gate_signals_by_window:
            gate_signals_by_window[window] = _gate_signals_by_slug_ts(bt, traded_probs, window)
        gate_er, gate_dir = gate_signals_by_window[window]
        strategy = WorldModelStrategy(traded_probs, bankroll=cash, gate_er_by_slug_ts=gate_er, **kwargs)
        model_params = {**_cpp_params(kwargs, None), "gate_er_by_slug_ts": gate_er}
        # Fair baseline: the identical asset-correct predictability gate, buying the observable spot-trend
        # direction with no model, so the model must beat trend-following over the same selective gate.
        naive_pred_threshold = value if sweep_param == "predictability" else base_kwargs["predictability_threshold"]
        naive_strategy = NaiveLagStrategy(
            cusum_threshold=base_kwargs["cusum_threshold"], order_size=base_kwargs["order_size"],
            max_tte_frac=base_kwargs["max_tte_frac"], use_predictability_gate=True,
            predictability_threshold=naive_pred_threshold, predictability_window=window,
            gate_er_by_slug_ts=gate_er, gate_dir_by_slug_ts=gate_dir,
            min_book_depth=base_kwargs["min_book_depth"], max_book_depth=base_kwargs["max_book_depth"],
        )
        naive_sweep_params = {
            **base_kwargs, "use_predictability_gate": True, "predictability_threshold": naive_pred_threshold,
            "predictability_window": window, "ev_by_slug_ts": None,
            "gate_er_by_slug_ts": gate_er, "gate_dir_by_slug_ts": gate_dir,
        }
        model_signals = _sweep_aligned(engine, aligned_by_window_arm, marshalled, STRATEGY_WORLD_MODEL, model_params, window)
        naive_signals = _sweep_aligned(engine, aligned_by_window_arm, marshalled, STRATEGY_NAIVE_LAG, naive_sweep_params, window)
        tasks.append(_row_task(value, strategy, model_params, model_signals, naive_strategy, naive_sweep_params, naive_signals))
    rows = _run_concurrent(tasks, _engine_workers(engine, engine_threads))
    return {
        "n_markets": len(slugs), "n_probs": len(prob_by_slug_ts),
        "calibration": settlement_calibration(prob_by_slug_ts, bt), "sweep": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PnL backtest of a frozen world model per spot-vol regime")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--norm-path", required=True, type=Path)
    p.add_argument("--regime-film", action="store_true", help="Rebuild the model with RegimeFiLM on (treatment arm).")
    p.add_argument("--is-mimo", action="store_true")
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--hours-val", type=float, default=6.0)
    p.add_argument("--assets", default="BTC", help="Comma-separated asset filter; the engine references are BTC-only.")
    p.add_argument("--intervals", default=None, help="Comma-separated interval filter (e.g. '15m'); newgoal-2 validates 15m before 5m.")
    p.add_argument("--regime-axis", choices=("predictability", "spot_vol"), default="predictability",
                   help="predictability = Efficiency-Ratio split (newgoal-2 primary axis); spot_vol = Phase 0.5 split.")
    p.add_argument("--volatility-quantile", type=float, default=0.5)
    p.add_argument("--edge-threshold", type=float, default=0.05)
    p.add_argument("--order-size", type=float, default=50.0)
    p.add_argument("--max-tte-frac", type=float, default=1.0)
    p.add_argument("--min-tte-frac", type=float, default=0.0,
                   help="Skip near-expiry trades below this time-remaining fraction (the edge is early-market; near-expiry reverses).")
    p.add_argument("--use-cusum", action="store_true", help="Gate trades on a CUSUM spot-move event (newgoal-2 1b).")
    p.add_argument("--cusum-threshold", type=float, default=0.003, help="CUSUM spot-return threshold; tune on train.")
    p.add_argument("--sizing", choices=("fixed", "kelly"), default="fixed", help="kelly = quarter-Kelly convex sizing (1c).")
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--kelly-cap", type=float, default=0.05)
    p.add_argument("--predictability-gate", action="store_true",
                   help="Trade only when the causal rolling spot efficiency ratio is above threshold (selective participation).")
    p.add_argument("--predictability-threshold", type=float, default=0.3, help="Causal-ER gate threshold; tune on train.")
    p.add_argument("--predictability-window", type=int, default=120, help="Ticks of recent spot for the causal ER gate.")
    p.add_argument("--sweep-thresholds", default=None,
                   help="Comma-separated predictability-gate thresholds to sweep on the full market set (train-tuning); "
                        "precomputes probs once and reports per-threshold deployable PnL + survivability.")
    p.add_argument("--slippage-per-share", type=float, default=0.0,
                   help="Per-share execution cost subtracted from each fill in the sweep's bootstrap (deployability stress test).")
    p.add_argument("--signal-delay-secs", type=int, default=0,
                   help="Stale the model signal by N seconds before trading (latency stress test); the gate stays at trade time. python engine only.")
    p.add_argument("--calibration-temperature", type=float, default=1.0,
                   help="Post-hoc temperature on the settlement prob (fit on train); >1 softens the over-confident head for Kelly.")
    p.add_argument("--sweep-param", choices=("predictability", "edge", "window"), default="predictability",
                   help="Which hyperparameter --sweep-thresholds varies: the gate ER (default), the model-vs-book edge, or the gate window.")
    p.add_argument("--min-book-depth", type=float, default=0.0,
                   help="Book-depth gate: trade only when two-sided depth clears this floor (tests asset-identity vs depth).")
    p.add_argument("--max-book-depth", type=float, default=0.0,
                   help="Book-depth ceiling (0 = none): skip the most-liquid/efficient books, trading the liquidity band.")
    p.add_argument("--prob-source", choices=("settlement", "edge"), default="settlement",
                   help="settlement: trade from settlement probability (existing path); edge: trade from predicted EV (BookRelativeEdgeHead).")
    p.add_argument("--deterministic-latent", action="store_true",
                   help="Use the deterministic latent ('probs') in the prob precompute so the sweep PnL is reproducible.")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--out", type=Path, default=Path("reports/pnl_backtest.json"))
    p.add_argument("--device", default=None)
    p.add_argument("--engine", choices=("python", "cpp"), default="python",
                   help="python = pure-Python reference engine (default); cpp = native engine (build with "
                        "`python -m finmamba3.backtester.engine_cpp.build`). Parity-checked; amortises across a sweep.")
    p.add_argument("--engine-threads", type=int, default=0,
                   help="Threads for independent cpp backtests (sweep values, both arms); 0 = auto (os.cpu_count()), "
                        "1 = serial. The cpp engine releases the GIL so these run on real cores; the python engine ignores it.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(args.config, _arch_overrides(args.regime_film, args.is_mimo, args.n_layer))
    include_binary = cfg.Models.WorldModel.Encoder.get("BinaryMarketFeatures", False)
    include_spot = cfg.Models.WorldModel.Encoder.get("SpotFeatures", False)
    include_cross = cfg.Models.WorldModel.Encoder.get("CrossIntervalContext", False)
    wm = load_world_model(cfg, args.checkpoint, device)
    if args.prob_source == "edge":
        assert wm.use_edge_head, "checkpoint has no edge head; train with EdgeHead.Enabled=True or use --prob-source settlement."
    else:
        assert wm.use_settlement_head, "checkpoint has no settlement head; the PnL strategy trades its YES probability."
    assets = [a.strip().upper() for a in args.assets.split(",")]
    intervals = [s.strip() for s in args.intervals.split(",")] if args.intervals else None
    bt = build_timeline(data_dir=args.data_val, hours=args.hours_val, assets=assets, intervals=intervals)
    stats = load_normalization(args.norm_path)
    # Only markets with at least one scoreable window can be traded, so the split is taken over those;
    # this also avoids extract_features raising on a market whose YES book never has both sides.
    count_by_slug = _usable_tick_count_by_slug(bt)
    tradeable = [lc for lc in bt.lifecycles if count_by_slug.get(lc.market_slug, 0) >= 64]
    # The reference regime is where the model has the most reason to profit (predictable / low-vol);
    # the shifted regime is the harder one. degradation = pnl(reference) - pnl(shifted), so a positive
    # degradation means edge lost under the shift, and the FiLM gap subtracts the two arms' degradations.
    if args.regime_axis == "predictability":
        score_by_slug = predictability_from_timeline(bt.timeline, tradeable, metric="efficiency_ratio")
        split = volatility_split(tradeable, realized_vol=score_by_slug, quantile=args.volatility_quantile)
        reference_label, reference_markets = "high_pred", split.test_markets
        shifted_label, shifted_markets = "low_pred", split.train_markets
    else:
        score_by_slug = spot_realized_vol_from_timeline(bt.timeline, tradeable)
        split = volatility_split(tradeable, realized_vol=score_by_slug, quantile=args.volatility_quantile)
        reference_label, reference_markets = "low_vol", split.train_markets
        shifted_label, shifted_markets = "high_vol", split.test_markets
    reference_slugs = [m.market_slug for m in reference_markets]
    shifted_slugs = [m.market_slug for m in shifted_markets]
    assert reference_slugs and shifted_slugs, (
        f"{args.regime_axis} split is degenerate: {reference_label}={len(reference_slugs)} "
        f"{shifted_label}={len(shifted_slugs)}; widen --hours-val."
    )
    logger.info(f"{args.regime_axis} split ({split.description}): {reference_label}={len(reference_slugs)} {shifted_label}={len(shifted_slugs)} markets")
    strategy_kwargs = {
        "edge_threshold": args.edge_threshold,
        "order_size": args.order_size,
        "max_tte_frac": args.max_tte_frac,
        "min_tte_frac": args.min_tte_frac,
        "use_cusum": bool(args.use_cusum),
        "cusum_threshold": args.cusum_threshold,
        "sizing": args.sizing,
        "kelly_fraction": args.kelly_fraction,
        "kelly_cap": args.kelly_cap,
        "use_predictability_gate": bool(args.predictability_gate),
        "predictability_threshold": args.predictability_threshold,
        "predictability_window": args.predictability_window,
        "calibration_temperature": args.calibration_temperature,
        "min_book_depth": args.min_book_depth,
        "max_book_depth": args.max_book_depth,
    }
    if args.sweep_thresholds:
        values = [float(t) for t in args.sweep_thresholds.split(",")]
        all_slugs = [lc.market_slug for lc in tradeable]
        sweep_report = run_threshold_sweep(
            wm, bt, stats, all_slugs, device,
            include_binary, include_spot, include_cross, strategy_kwargs, values, args.cash, args.slippage_per_share,
            args.sweep_param, "probs" if args.deterministic_latent else "random_sample", engine=args.engine,
            signal_delay_secs=args.signal_delay_secs, engine_threads=args.engine_threads,
        )
        sweep_report["signal_delay_secs"] = int(args.signal_delay_secs)
        sweep_report["checkpoint"] = str(args.checkpoint)
        sweep_report["data_val"] = str(args.data_val)
        sweep_report["slippage_per_share"] = float(args.slippage_per_share)
        for row in sweep_report["sweep"]:
            logger.info(
                f"[sweep] {row['sweep_param']}={row['value']:.3f} pnl={row['pnl']:+.1f} (after_slip={row['pnl_after_slippage']:+.1f}) "
                f"trades={row['trades']} naive={row['naive_pnl']:+.1f} ({row['naive_trades']} tr) "
                f"m-n={row['model_minus_naive']:+.1f} P(profit)={row['frac_profitable']:.3f} dd95={row['drawdown_p95']:.3f}"
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(sweep_report, indent=2))
        logger.info(f"[sweep] report written to {args.out}")
        return 0
    report = {
        "checkpoint": str(args.checkpoint), "regime_film": bool(args.regime_film),
        "regime_axis": args.regime_axis, "split": split.description,
        "reference_label": reference_label, "shifted_label": shifted_label,
        "slippage_per_share": float(args.slippage_per_share),
    }
    for label, slugs in ((reference_label, reference_slugs), (shifted_label, shifted_slugs)):
        report[label] = run_pnl_backtest(
            wm, bt, stats, slugs, device,
            include_binary, include_spot, include_cross, strategy_kwargs, args.cash,
            prob_source=args.prob_source,
            slippage_per_share=args.slippage_per_share, engine=args.engine, engine_threads=args.engine_threads,
        )
        cell = report[label]
        boot = cell["model"]["bootstrap"]
        logger.info(
            f"[pnl] {label}: model_pnl={cell['model']['pnl']:+.2f} naive_pnl={cell['naive']['pnl']:+.2f} "
            f"model-naive={cell['model_minus_naive']:+.2f} trades={cell['model']['trades']} "
            f"P(profit)={boot['frac_profitable']:.2f} dd95={boot['drawdown_p95']:.3f} markets={cell['n_markets']}"
        )
    report["degradation_model"] = report[reference_label]["model"]["pnl"] - report[shifted_label]["model"]["pnl"]
    report["degradation_naive"] = report[reference_label]["naive"]["pnl"] - report[shifted_label]["naive"]["pnl"]
    logger.info(f"[pnl] degradation_model ({reference_label}-{shifted_label}) = {report['degradation_model']:+.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info(f"[pnl] report written to {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

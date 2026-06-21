"""Concrete trading strategies for the PnL backtest.

The CUSUM spot-event gate, the world-model edge/divergence strategy, and the naive
latency-arbitrage baseline the world model must beat. All consume the precomputed
probability/EV and gate series; none runs the model itself.
"""
# region imports
from __future__ import annotations
import numpy as np
from finmamba3.backtester.engine import MAX_SHARES_PER_TOKEN
from finmamba3.backtester.strategy import BaseStrategy, Order, Side, Token
from finmamba3.eval.predictability import efficiency_ratio
from finmamba3.eval.survivability import kelly_shares
# endregion


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

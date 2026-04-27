"""Gymnasium trading environment over Polymarket LOB backtest data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lob.backtester.data_loader import BacktestData, TickData
from lob.backtester.strategy import Fill, OrderBookSnapshot, Settlement, Side, Token


ACTION_SPECS = (
    (Side.BUY, Token.YES),
    (Side.BUY, Token.NO),
    (Side.SELL, Token.YES),
    (Side.SELL, Token.NO),
)

OBS_FEATURE_NAMES = (
    "yes_mid",
    "yes_spread",
    "yes_top_imbalance",
    "yes_total_depth",
    "yes_ofi",
    "no_mid",
    "no_spread",
    "no_top_imbalance",
    "no_total_depth",
    "log_staleness",
    "time_to_expiry_frac",
    "time_to_expiry_hours",
    "binance_mid_return",
    "chainlink_drift",
    "yes_position_frac",
    "no_position_frac",
    "cash_frac",
    "exposure_frac",
)


@dataclass
class PositionState:
    yes_shares: float = 0.0
    no_shares: float = 0.0
    cost_basis: float = 0.0


def _asset_from_slug(slug: str) -> str:
    s = slug.lower()
    if s.startswith("eth") or s.startswith("ethereum"):
        return "ETH"
    if s.startswith("sol") or s.startswith("solana"):
        return "SOL"
    return "BTC"


def _book_mid(book: OrderBookSnapshot) -> float:
    if book.best_bid > 0.0 and book.best_ask > 0.0:
        return 0.5 * (book.best_bid + book.best_ask)
    return book.best_bid or book.best_ask or 0.0


def _top_imbalance(book: OrderBookSnapshot) -> float:
    bid = float(book.bids[0].size) if book.bids else 0.0
    ask = float(book.asks[0].size) if book.asks else 0.0
    denom = bid + ask
    return (bid - ask) / denom if denom > 0.0 else 0.0


def _total_depth(book: OrderBookSnapshot) -> float:
    return float(book.total_bid_size + book.total_ask_size)


class PolymarketLOBEnv(gym.Env):
    """Execution-aware Polymarket LOB environment.

    Actions submitted at observation ``t`` execute against the book at
    ``t + latency_ticks``. The default one-tick latency matches the DATAHACKS
    backtester contract.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: BacktestData,
        *,
        initial_cash: float = 10_000.0,
        max_markets: int = 8,
        size_buckets: tuple[float, ...] = (10.0, 25.0, 50.0),
        latency_ticks: int = 1,
        vol_scale: float = 0.01,
        max_position_shares: float = 1_000.0,
        reset_to_active: bool = True,
    ) -> None:
        super().__init__()
        if len(data.timeline) < 2:
            raise ValueError("PolymarketLOBEnv requires at least two timeline ticks")
        if latency_ticks < 1:
            raise ValueError("latency_ticks must be >= 1")
        self.data = data
        self.initial_cash = float(initial_cash)
        self.max_markets = int(max_markets)
        self.size_buckets = tuple(float(x) for x in size_buckets)
        self.latency_ticks = int(latency_ticks)
        self.vol_scale = float(vol_scale)
        self.max_position_shares = float(max_position_shares)
        self.reset_to_active = bool(reset_to_active)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_markets, len(OBS_FEATURE_NAMES)),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(1 + len(ACTION_SPECS) * len(self.size_buckets))

        self._lifecycle_by_slug = {lc.market_slug: lc for lc in data.lifecycles}
        self._i = 0
        self.cash = self.initial_cash
        self.positions: dict[str, PositionState] = {}
        self._settled: set[str] = set()
        self._high_water = self.initial_cash
        self._drawdown = 0.0
        self._prev_top: dict[tuple[str, Token], tuple[float, float]] = {}
        self._prev_binance_mid: dict[str, float] = {}
        self._done = False

    @property
    def tick(self) -> TickData:
        return self.data.timeline[self._i]

    def decode_action(self, action: int) -> tuple[Side | None, Token | None, float]:
        action = int(action)
        if action == 0:
            return None, None, 0.0
        action -= 1
        spec_idx = action // len(self.size_buckets)
        size_idx = action % len(self.size_buckets)
        if spec_idx < 0 or spec_idx >= len(ACTION_SPECS):
            raise ValueError(f"Invalid action {action + 1}")
        side, token = ACTION_SPECS[spec_idx]
        return side, token, self.size_buckets[size_idx]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        self._i = int(options.get("start_index", 0))
        self._i = min(max(self._i, 0), len(self.data.timeline) - 2)
        if self.reset_to_active and "start_index" not in options:
            self._i = self._first_active_index()
        self.cash = float(options.get("cash", self.initial_cash))
        self.positions = {}
        self._settled = set()
        self._high_water = self.cash
        self._drawdown = 0.0
        self._prev_top = {}
        self._prev_binance_mid = {}
        self._done = False
        obs = self._observation(self.tick)
        return obs, self._info(fill=None, invalid_action=False, settlements=[])

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called after episode termination; call reset()")

        value_before = self._portfolio_value(self.tick)
        submit_ts = self.tick.ts_sec
        self._i = min(self._i + self.latency_ticks, len(self.data.timeline) - 1)
        tick = self.tick

        settlements = self._settle_expired(tick.ts_sec)
        fill, invalid_action = self._execute_action(action, tick)

        value_after = self._portfolio_value(tick)
        turnover = (fill.cost / max(value_before, 1e-6)) if fill is not None else 0.0
        inventory_frac = self._inventory_fraction(tick, value_after)
        delta_log = math.log(max(value_after, 1e-6)) - math.log(max(value_before, 1e-6))

        self._high_water = max(self._high_water, value_after)
        drawdown = max(0.0, (self._high_water - value_after) / max(self._high_water, 1e-6))
        drawdown_increment = max(0.0, drawdown - self._drawdown)
        self._drawdown = drawdown

        reward = (
            math.tanh(delta_log / max(self.vol_scale, 1e-8))
            - 0.05 * turnover
            - 0.02 * inventory_frac ** 2
            - 0.10 * drawdown_increment
        )

        self._done = self._i >= len(self.data.timeline) - 1
        info = self._info(
            fill=fill,
            invalid_action=invalid_action,
            settlements=settlements,
            submit_ts=submit_ts,
        )
        info.update(
            {
                "portfolio_value": value_after,
                "turnover": turnover,
                "inventory_frac": inventory_frac,
                "drawdown": self._drawdown,
            }
        )
        return self._observation(tick), float(reward), self._done, False, info

    def _first_active_index(self) -> int:
        for idx, tick in enumerate(self.data.timeline[:-1]):
            if self._active_slugs(tick):
                return idx
        return 0

    def _active_slugs(self, tick: TickData) -> list[str]:
        out: list[str] = []
        for slug, stored in tick.order_books.items():
            lc = self._lifecycle_by_slug.get(slug)
            if lc is None or not (lc.start_ts <= tick.ts_sec < lc.end_ts):
                continue
            if stored.yes_book.bids and stored.yes_book.asks:
                out.append(slug)
        out.sort(key=lambda s: (self._lifecycle_by_slug[s].end_ts, s))
        return out

    def _target_slug(self, tick: TickData) -> str | None:
        active = self._active_slugs(tick)
        return active[0] if active else None

    def _execute_action(self, action: int, tick: TickData) -> tuple[Fill | None, bool]:
        side, token, size = self.decode_action(action)
        if side is None or token is None:
            return None, False
        slug = self._target_slug(tick)
        if slug is None:
            return None, True
        stored = tick.order_books.get(slug)
        if stored is None:
            return None, True
        book = stored.yes_book if token == Token.YES else stored.no_book
        levels = book.asks if side == Side.BUY else book.bids
        filled, cost = self._take_levels(levels, size)
        if filled <= 0.0:
            return None, True
        avg_price = cost / filled

        pos = self.positions.get(slug, PositionState())
        if side == Side.BUY:
            if cost > self.cash + 1e-9:
                return None, True
            self.cash -= cost
            if token == Token.YES:
                pos.yes_shares += filled
            else:
                pos.no_shares += filled
            pos.cost_basis += cost
            self.positions[slug] = pos
        else:
            held = pos.yes_shares if token == Token.YES else pos.no_shares
            if filled > held + 1e-9:
                return None, True
            self.cash += cost
            if token == Token.YES:
                pos.yes_shares -= filled
            else:
                pos.no_shares -= filled
            pos.cost_basis = max(0.0, pos.cost_basis - cost)
            if pos.yes_shares <= 1e-9 and pos.no_shares <= 1e-9:
                self.positions.pop(slug, None)
            else:
                self.positions[slug] = pos

        return (
            Fill(
                market_slug=slug,
                token=token,
                side=side,
                size=filled,
                avg_price=avg_price,
                cost=cost,
                timestamp=tick.ts_sec,
            ),
            False,
        )

    @staticmethod
    def _take_levels(levels, requested_size: float) -> tuple[float, float]:
        remaining = float(requested_size)
        filled = 0.0
        notional = 0.0
        for level in levels:
            qty = min(remaining, float(level.size))
            if qty <= 0.0:
                break
            filled += qty
            notional += qty * float(level.price)
            remaining -= qty
            if remaining <= 1e-9:
                break
        return filled, notional

    def _settle_expired(self, ts_sec: int) -> list[Settlement]:
        settled_now: list[Settlement] = []
        for slug in list(self.positions):
            if slug in self._settled:
                continue
            settlement = self.data.settlements.get(slug)
            if settlement is None or ts_sec < settlement.end_ts:
                continue
            pos = self.positions.pop(slug)
            if settlement.outcome == Token.YES:
                self.cash += pos.yes_shares
            else:
                self.cash += pos.no_shares
            self._settled.add(slug)
            settled_now.append(settlement)
        return settled_now

    def _portfolio_value(self, tick: TickData) -> float:
        value = self.cash
        for slug, pos in self.positions.items():
            settlement = self.data.settlements.get(slug)
            if settlement is not None and tick.ts_sec >= settlement.end_ts:
                value += pos.yes_shares if settlement.outcome == Token.YES else pos.no_shares
                continue
            stored = tick.order_books.get(slug)
            if stored is None:
                continue
            value += pos.yes_shares * max(stored.yes_book.best_bid, 0.0)
            value += pos.no_shares * max(stored.no_book.best_bid, 0.0)
        return float(value)

    def _inventory_fraction(self, tick: TickData, portfolio_value: float) -> float:
        exposure = 0.0
        for slug, pos in self.positions.items():
            stored = tick.order_books.get(slug)
            if stored is None:
                continue
            exposure += pos.yes_shares * _book_mid(stored.yes_book)
            exposure += pos.no_shares * _book_mid(stored.no_book)
        return float(exposure / max(portfolio_value, 1e-6))

    def _observation(self, tick: TickData) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        active = self._active_slugs(tick)[: self.max_markets]
        portfolio_value = self._portfolio_value(tick)
        for row, slug in enumerate(active):
            stored = tick.order_books[slug]
            lc = self._lifecycle_by_slug[slug]
            pos = self.positions.get(slug, PositionState())
            asset = _asset_from_slug(slug)
            binance_mid = getattr(tick, f"{asset.lower()}_mid")
            chainlink = getattr(tick, f"chainlink_{asset.lower()}")
            prev_binance = self._prev_binance_mid.get(asset, binance_mid)
            binance_ret = (
                math.log(binance_mid / prev_binance)
                if binance_mid > 0.0 and prev_binance > 0.0
                else 0.0
            )
            self._prev_binance_mid[asset] = binance_mid
            chainlink_drift = (
                (chainlink - binance_mid) / binance_mid
                if chainlink > 0.0 and binance_mid > 0.0
                else 0.0
            )
            book_ts = tick.book_timestamps.get(slug, tick.ts_sec)
            staleness = max(float(tick.ts_sec - book_ts), 0.0)
            time_remaining = max(float(lc.end_ts - tick.ts_sec), 0.0)
            duration = max(float(lc.end_ts - lc.start_ts), 1.0)
            yes_ofi = self._ofi(slug, Token.YES, stored.yes_book)
            no_book = stored.no_book
            yes_book = stored.yes_book

            exposure_frac = self._inventory_fraction(tick, portfolio_value)
            obs[row] = np.asarray(
                [
                    _book_mid(yes_book),
                    yes_book.spread,
                    _top_imbalance(yes_book),
                    min(math.log1p(_total_depth(yes_book)) / 10.0, 8.0),
                    yes_ofi,
                    _book_mid(no_book),
                    no_book.spread,
                    _top_imbalance(no_book),
                    min(math.log1p(_total_depth(no_book)) / 10.0, 8.0),
                    min(math.log1p(staleness) / 8.0, 8.0),
                    time_remaining / duration,
                    time_remaining / 3600.0,
                    float(np.clip(binance_ret * 100.0, -8.0, 8.0)),
                    float(np.clip(chainlink_drift * 100.0, -8.0, 8.0)),
                    pos.yes_shares / self.max_position_shares,
                    pos.no_shares / self.max_position_shares,
                    self.cash / max(self.initial_cash, 1e-6),
                    exposure_frac,
                ],
                dtype=np.float32,
            )
        return obs

    def _ofi(self, slug: str, token: Token, book: OrderBookSnapshot) -> float:
        bid = float(book.bids[0].size) if book.bids else 0.0
        ask = float(book.asks[0].size) if book.asks else 0.0
        key = (slug, token)
        prev_bid, prev_ask = self._prev_top.get(key, (bid, ask))
        self._prev_top[key] = (bid, ask)
        denom = max(bid + ask + prev_bid + prev_ask, 1.0)
        return float(np.clip(((bid - prev_bid) - (ask - prev_ask)) / denom, -8.0, 8.0))

    def _info(
        self,
        *,
        fill: Fill | None,
        invalid_action: bool,
        settlements: list[Settlement],
        submit_ts: int | None = None,
    ) -> dict[str, Any]:
        tick = self.tick
        return {
            "timestamp": tick.ts_sec,
            "submit_timestamp": submit_ts,
            "cash": self.cash,
            "target_market": self._target_slug(tick),
            "fill": fill,
            "invalid_action": invalid_action,
            "settlements": settlements,
        }

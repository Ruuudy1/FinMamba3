"""Survivability statistics for the PnL backtest.

Kelly trade sizing, realized per-trade PnL, and the fat-tail bootstrap (iid and the
market-block variant that resamples whole markets) plus the snapshot Sharpe. Pure NumPy;
no torch, so these stay trivially unit-testable and import-cheap.
"""
# region imports
from __future__ import annotations
import numpy as np
from finmamba3.backtester.strategy import Token
# endregion


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

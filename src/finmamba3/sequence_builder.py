"""Build normalized LOB feature sequences and populate replay buffers.

Shared by the Phase-A trainer and the eval/imagination CLIs so train and serve
go through the identical feature pipeline (no train/serve skew).
"""
# region imports
from __future__ import annotations
import logging
import math
from dataclasses import replace
from pathlib import Path
import numpy as np
from finmamba3.backtester import build_timeline
from finmamba3.backtester.data_loader import _asset_from_slug
from finmamba3.envs.fi2010_loader import load_fi2010_split
from finmamba3.envs.kaggle_lob_loader import load_kaggle_lob, max_rows_for_hours
from finmamba3.envs.lob_features import (
    LOBSequence,
    apply_normalization,
    extract_features,
    fit_normalization,
    load_normalization,
    make_aggregate_only,
    pick_longest_market,
    pick_top_markets,
    save_normalization,
)
from finmamba3.replay_buffer import ReplayBuffer
# endregion
logger = logging.getLogger(__name__)


def _settlement_yes_outcome(settlement) -> float | None:
    if settlement is None:
        return None
    outcome = settlement.outcome.value
    if outcome == "YES":
        return 1.0
    if outcome == "NO":
        return 0.0
    return None


CROSS_INTERVAL_WIDTH = 3
_SHORT_INTERVALS = ("5m", "15m")


def cross_interval_context_block(bt, hourly_slug: str, ts_sec_array: np.ndarray) -> np.ndarray:
    """Per-tick context summarizing the contemporaneous short-interval markets of an hourly market.

    For each hourly book-tick the block carries three channels (Phase 0.4): the mean YES mid of the
    same-asset 5m/15m markets active at that tick, the YES fraction among those that have already
    resolved this hour, and log1p of that resolved count. Those short markets resolve inside the
    hourly window and are already-settled, direct evidence about the hourly direction, so a
    concatenated summary is the simplest signal that exposes them (KISS, not a cross-attention
    module). The caller appends a zero block to non-hourly markets so the feature width stays uniform.
    """
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    hourly_lifecycle = lifecycle_by_slug[hourly_slug]
    asset = _asset_from_slug(hourly_slug)
    short_slugs = [
        lc.market_slug for lc in bt.lifecycles
        if lc.interval in _SHORT_INTERVALS and
        _asset_from_slug(lc.market_slug) == asset and
        lc.start_ts >= hourly_lifecycle.start_ts and
        lc.end_ts <= hourly_lifecycle.end_ts
    ]
    # Sort the resolved short markets by end time with their YES outcome so a searchsorted per tick
    # yields how many have settled and their cumulative YES count without rescanning the list.
    resolved = []
    for slug in short_slugs:
        outcome = _settlement_yes_outcome(bt.settlements.get(slug))
        if outcome is not None:
            resolved.append((lifecycle_by_slug[slug].end_ts, outcome))
    resolved.sort(key=lambda pair: pair[0])
    end_ts_sorted = np.fromiter((end for end, _ in resolved), dtype=np.int64, count=len(resolved))
    yes_cumsum = np.concatenate([[0.0], np.cumsum([out for _, out in resolved], dtype=np.float64)])
    # The active short mids live on the very timeline tick the hourly book lives on, so one pass over
    # the wanted timestamps reads them directly without a separate per-market mid series.
    row_by_ts = {int(ts): i for i, ts in enumerate(ts_sec_array)}
    active_mean_mid = np.full(len(ts_sec_array), 0.5, dtype=np.float64)
    for tick in bt.timeline:
        row = row_by_ts.get(int(tick.ts_sec))
        if row is None:
            continue
        mids = []
        for slug in short_slugs:
            sb = tick.order_books.get(slug)
            if sb is not None and sb.yes_book.bids and sb.yes_book.asks and sb.yes_book.mid > 0.0:
                mids.append(sb.yes_book.mid)
        if mids:
            active_mean_mid[row] = float(np.mean(mids))
    block = np.zeros((len(ts_sec_array), CROSS_INTERVAL_WIDTH), dtype=np.float32)
    for i, ts in enumerate(ts_sec_array):
        n_resolved = int(np.searchsorted(end_ts_sorted, int(ts), side="right"))
        block[i, 0] = active_mean_mid[i]
        block[i, 1] = float(yes_cumsum[n_resolved] / n_resolved) if n_resolved > 0 else 0.5
        block[i, 2] = math.log1p(float(n_resolved))
    return block


def _append_cross_interval(bt, slug: str, seq: LOBSequence, lifecycle_by_slug: dict) -> LOBSequence:
    # Hourly markets get the real contemporaneous-short summary; others get a zero block so the buffer
    # feature width is uniform across intervals, which the fixed-size encoder requires.
    if lifecycle_by_slug[slug].interval == "hourly":
        block = cross_interval_context_block(bt, slug, seq.ts_sec)
    else:
        block = np.zeros((seq.per_tick.shape[0], CROSS_INTERVAL_WIDTH), dtype=np.float32)
    return replace(seq, per_tick=np.concatenate([seq.per_tick, block], axis=1))


def _spot_feature_kwargs(slug: str, settlement, lifecycle_by_slug: dict) -> dict:
    # The spot block is anchored to the market's lifecycle bounds and its open Chainlink
    # reference. start_ts/end_ts come from the slug-parsed lifecycle (always present); the open
    # reference reuses the settlement's chainlink_open when computed, else extract_features scans.
    lifecycle = lifecycle_by_slug.get(slug)
    if lifecycle is None:
        raise RuntimeError(f"No lifecycle for market {slug!r}; cannot build spot features.")
    spot_open = settlement.chainlink_open if settlement is not None and settlement.chainlink_open > 0.0 else None
    return {
        "asset": _asset_from_slug(slug),
        "start_ts": int(lifecycle.start_ts),
        "end_ts": int(lifecycle.end_ts),
        "spot_open": spot_open,
    }


def populate_buffer(buffer: ReplayBuffer, seq: LOBSequence) -> None:
    flat = seq.to_flat()
    T = flat.shape[0]
    yes_outcome = seq.yes_outcome
    tte = seq.tte_frac
    spot_dist = seq.spot_signed_distance
    event_counts = seq.event_counts
    yes_ask = seq.yes_ask
    no_ask = seq.no_ask
    yes_mid = seq.yes_mid
    book_depth = seq.book_depth
    for t in range(T):
        outcome_t = float(yes_outcome[t]) if yes_outcome is not None else float('nan')
        buffer.append(
            obs=flat[t],
            action=0,
            reward=0.0,
            termination=0.0,
            outcome=outcome_t,
            event_counts=event_counts[t] if event_counts is not None else None,
            tte_frac=float(tte[t]) if tte is not None else float('nan'),
            spot_signed_distance=float(spot_dist[t]) if spot_dist is not None else float('nan'),
            yes_ask=float(yes_ask[t]) if yes_ask is not None else float('nan'),
            no_ask=float(no_ask[t]) if no_ask is not None else float('nan'),
            yes_mid=float(yes_mid[t]) if yes_mid is not None else float('nan'),
            book_depth=float(book_depth[t]) if book_depth is not None else float('nan'),
        )
    logger.info(f"replay buffer: loaded {T} ticks for market {seq.market_slug}")


def populate_buffer_multi(buffer: ReplayBuffer, seqs: list[LOBSequence]) -> None:
    """Append several markets' ticks in order, flagging each market's final tick.

    The segment-end flag lets the sampler reject any window that would straddle
    two markets, so per-market temporal continuity is preserved.
    """
    total = 0
    for seq in seqs:
        flat = seq.to_flat()
        T = flat.shape[0]
        yes_outcome = seq.yes_outcome
        tte = seq.tte_frac
        spot_dist = seq.spot_signed_distance
        event_counts = seq.event_counts
        yes_ask = seq.yes_ask
        no_ask = seq.no_ask
        yes_mid = seq.yes_mid
        book_depth = seq.book_depth
        for t in range(T):
            outcome_t = float(yes_outcome[t]) if yes_outcome is not None else float('nan')
            buffer.append(
                obs=flat[t],
                action=0,
                reward=0.0,
                termination=0.0,
                outcome=outcome_t,
                event_counts=event_counts[t] if event_counts is not None else None,
                tte_frac=float(tte[t]) if tte is not None else float('nan'),
                spot_signed_distance=float(spot_dist[t]) if spot_dist is not None else float('nan'),
                yes_ask=float(yes_ask[t]) if yes_ask is not None else float('nan'),
                no_ask=float(no_ask[t]) if no_ask is not None else float('nan'),
                yes_mid=float(yes_mid[t]) if yes_mid is not None else float('nan'),
                book_depth=float(book_depth[t]) if book_depth is not None else float('nan'),
                segment_end=(t == T - 1),
            )
        total += T
        logger.info(f"replay buffer: loaded {T} ticks for market {seq.market_slug}")
    logger.info(f"replay buffer: {total} ticks total across {len(seqs)} markets")


def build_sequences(
    data_dir: Path,
    market_slug: str | None,
    hours: float,
    norm_path: Path,
    fit_stats: bool,
    norm_clip: float,
    aggregate_only: bool,
    intervals: list[str] | None = None,
    include_binary_features: bool = False,
    include_spot_features: bool = False,
    include_cross_interval: bool = False,
) -> tuple[LOBSequence, str, object]:
    bt = build_timeline(data_dir=data_dir, hours=hours, intervals=intervals)
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    slug = market_slug or pick_longest_market(bt)
    settlement = bt.settlements.get(slug)
    yes_outcome = _settlement_yes_outcome(settlement)
    spot_kwargs = _spot_feature_kwargs(slug, settlement, lifecycle_by_slug) if include_spot_features else {}
    try:
        seq = extract_features(
            bt.timeline, slug, yes_outcome=yes_outcome,
            include_binary_features=include_binary_features,
            include_spot_features=include_spot_features, **spot_kwargs,
        )
    except RuntimeError:
        # Requested slug has no usable ticks in this split; fall back to the
        # longest market available in this split.
        slug = pick_longest_market(bt)
        settlement = bt.settlements.get(slug)
        yes_outcome = _settlement_yes_outcome(settlement)
        spot_kwargs = _spot_feature_kwargs(slug, settlement, lifecycle_by_slug) if include_spot_features else {}
        logger.warning(
            f"Market {market_slug!r} has no usable ticks in {data_dir}; "
            f"falling back to {slug!r}"
        )
        seq = extract_features(
            bt.timeline, slug, yes_outcome=yes_outcome,
            include_binary_features=include_binary_features,
            include_spot_features=include_spot_features, **spot_kwargs,
        )
    if include_cross_interval:
        seq = _append_cross_interval(bt, slug, seq, lifecycle_by_slug)
    if fit_stats:
        stats = fit_normalization(seq, clip_value=norm_clip)
        save_normalization(stats, norm_path)
        logger.info(f"normalization fit on {slug}, saved to {norm_path}")
    else:
        stats = load_normalization(norm_path)
    seq_norm = apply_normalization(seq, stats)
    if aggregate_only:
        seq_norm = make_aggregate_only(seq_norm)
    return seq_norm, slug, stats


def build_market_sequences(
    data_dir: Path,
    market_slug: str | None,
    hours: float,
    norm_path: Path,
    fit_stats: bool,
    norm_clip: float,
    aggregate_only: bool,
    max_markets: int,
    intervals: list[str] | None = None,
    include_binary_features: bool = False,
    include_spot_features: bool = False,
    include_cross_interval: bool = False,
    assets: list[str] | None = None,
) -> tuple[list[LOBSequence], list[str], object]:
    """Build per-market normalized sequences for the top-N longest markets.

    Each market's windowed features are computed on that market alone, so no
    feature straddles a market boundary. Normalization is fit once on the
    concatenation of all markets (boundaries do not affect per-feature mean and
    std), then applied per market. Returns the normalized sequences (longest
    first), their slugs, and the shared stats.
    """
    bt = build_timeline(data_dir=data_dir, hours=hours, intervals=intervals)
    lifecycle_by_slug = {lc.market_slug: lc for lc in bt.lifecycles}
    slugs = [market_slug] if market_slug else pick_top_markets(bt, max_markets, assets=assets)
    raw_seqs: list[LOBSequence] = []
    for slug in slugs:
        settlement = bt.settlements.get(slug)
        yes_outcome = _settlement_yes_outcome(settlement)
        spot_kwargs = _spot_feature_kwargs(slug, settlement, lifecycle_by_slug) if include_spot_features else {}
        raw_seqs.append(
            extract_features(
                bt.timeline, slug, yes_outcome=yes_outcome,
                include_binary_features=include_binary_features,
                include_spot_features=include_spot_features, **spot_kwargs,
            )
        )
    if include_cross_interval:
        raw_seqs = [_append_cross_interval(bt, slug, seq, lifecycle_by_slug) for slug, seq in zip(slugs, raw_seqs)]
    if fit_stats:
        # The temp sequence only feeds fit_normalization, which reads per_level and
        # per_tick; concatenating the other fields keeps it internally consistent.
        concat_seq = LOBSequence(
            market_slug="__concat__",
            per_level=np.concatenate([seq.per_level for seq in raw_seqs], axis=0),
            per_tick=np.concatenate([seq.per_tick for seq in raw_seqs], axis=0),
            midprice=np.concatenate([seq.midprice for seq in raw_seqs], axis=0),
            ts_sec=np.concatenate([seq.ts_sec for seq in raw_seqs], axis=0),
        )
        stats = fit_normalization(concat_seq, clip_value=norm_clip)
        save_normalization(stats, norm_path)
        logger.info(f"normalization fit on {len(raw_seqs)} markets, saved to {norm_path}")
    else:
        stats = load_normalization(norm_path)
    norm_seqs: list[LOBSequence] = []
    for seq in raw_seqs:
        seq_norm = apply_normalization(seq, stats)
        if aggregate_only:
            seq_norm = make_aggregate_only(seq_norm)
        norm_seqs.append(seq_norm)
    return norm_seqs, slugs, stats


def build_fi2010_sequences(
    data_dir: Path,
    split: str,
    horizon: int,
    norm_path: Path,
    fit_stats: bool,
    norm_clip: float,
    max_events: int | None = None,
) -> tuple[LOBSequence, str, object]:
    """FI-2010 mirror of build_sequences. Returns (sequence, slug, stats).

    Fits z-score normalization on the training split and reuses it for validation,
    matching the Polymarket flow. Works for both DecPre and ZScore source files.
    """
    bundle = load_fi2010_split(data_dir, split=split, horizon=horizon, max_events=max_events)
    seq = bundle.sequence
    if fit_stats:
        stats = fit_normalization(seq, clip_value=norm_clip)
        save_normalization(stats, norm_path)
        logger.info(f"FI-2010 normalization fit on {seq.market_slug}, saved to {norm_path}")
    else:
        stats = load_normalization(norm_path)
    seq_norm = apply_normalization(seq, stats)
    return seq_norm, seq.market_slug, stats


def _slice_lob_sequence(seq: LOBSequence, start: int, end: int) -> LOBSequence:
    # A row-aligned slice of every field is a valid shorter sequence; the Kaggle val split is the
    # tail carved out of the same stream the train split is the head of, so no future tick leaks back.
    return LOBSequence(
        market_slug=seq.market_slug,
        per_level=seq.per_level[start:end],
        per_tick=seq.per_tick[start:end],
        midprice=seq.midprice[start:end],
        ts_sec=seq.ts_sec[start:end],
        yes_outcome=None if seq.yes_outcome is None else seq.yes_outcome[start:end],
        event_counts=None if seq.event_counts is None else seq.event_counts[start:end],
    )


def build_kaggle_sequences(
    data_dir: Path,
    asset: str,
    resolution: str,
    split: str,
    hours_train: float,
    hours_val: float,
    norm_path: Path,
    fit_stats: bool,
    norm_clip: float,
    flat_threshold: float = 0.0,
) -> tuple[LOBSequence, str, object]:
    """Kaggle mirror of build_fi2010_sequences over one per-asset CSV. Returns (sequence, slug, stats).

    The Kaggle dataset ships a single continuous stream per asset and resolution, so the train and
    validation splits are carved chronologically from one file: train is the first hours_train, val
    the next hours_val, disjoint so no future tick leaks into training. Normalization is fit on the
    train slice and reused for validation, matching the FI-2010 and Polymarket flows.
    """
    rows_train = max_rows_for_hours(hours_train, resolution)
    rows_val = max_rows_for_hours(hours_val, resolution)
    if rows_train is None or rows_val is None:
        raise ValueError("Kaggle train/val hours must both be positive so the single stream can be split.")
    csv_path = Path(data_dir) / f"{asset}_{resolution}.csv"
    if split == "train":
        bundle = load_kaggle_lob(csv_path, asset, resolution, max_rows=rows_train, flat_threshold=flat_threshold)
        seq = bundle.sequence
    elif split == "validation":
        bundle = load_kaggle_lob(csv_path, asset, resolution, max_rows=rows_train + rows_val, flat_threshold=flat_threshold)
        total_rows = bundle.sequence.per_level.shape[0]
        if total_rows <= rows_train + 1:
            raise RuntimeError(
                f"Kaggle CSV {csv_path} has {total_rows} rows but the train slice alone wants {rows_train}; "
                f"lower HoursTrain or point at a longer file so the validation tail is non-empty."
            )
        seq = _slice_lob_sequence(bundle.sequence, rows_train, total_rows)
    else:
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}.")
    if fit_stats:
        stats = fit_normalization(seq, clip_value=norm_clip)
        save_normalization(stats, norm_path)
        logger.info(f"Kaggle normalization fit on {seq.market_slug} ({split}), saved to {norm_path}")
    else:
        stats = load_normalization(norm_path)
    seq_norm = apply_normalization(seq, stats)
    return seq_norm, seq.market_slug, stats

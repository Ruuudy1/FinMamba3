"""Build normalized LOB feature sequences and populate replay buffers.

Shared by the Phase-A trainer and the eval/imagination CLIs so train and serve
go through the identical feature pipeline (no train/serve skew).
"""
# region imports
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from finmamba3.backtester import build_timeline
from finmamba3.envs.fi2010_loader import load_fi2010_split
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


def populate_buffer(buffer: ReplayBuffer, seq: LOBSequence) -> None:
    flat = seq.to_flat()
    T = flat.shape[0]
    yes_outcome = seq.yes_outcome
    for t in range(T):
        outcome_t = float(yes_outcome[t]) if yes_outcome is not None else float('nan')
        buffer.append(
            obs=flat[t],
            action=0,
            reward=0.0,
            termination=0.0,
            outcome=outcome_t,
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
        for t in range(T):
            outcome_t = float(yes_outcome[t]) if yes_outcome is not None else float('nan')
            buffer.append(
                obs=flat[t],
                action=0,
                reward=0.0,
                termination=0.0,
                outcome=outcome_t,
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
) -> tuple[LOBSequence, str, object]:
    bt = build_timeline(data_dir=data_dir, hours=hours, intervals=intervals)
    slug = market_slug or pick_longest_market(bt)
    settlement = bt.settlements.get(slug)
    yes_outcome = _settlement_yes_outcome(settlement)
    try:
        seq = extract_features(bt.timeline, slug, yes_outcome=yes_outcome, include_binary_features=include_binary_features)
    except RuntimeError:
        # Requested slug has no usable ticks in this split; fall back to the
        # longest market available in this split.
        slug = pick_longest_market(bt)
        settlement = bt.settlements.get(slug)
        yes_outcome = _settlement_yes_outcome(settlement)
        logger.warning(
            f"Market {market_slug!r} has no usable ticks in {data_dir}; "
            f"falling back to {slug!r}"
        )
        seq = extract_features(bt.timeline, slug, yes_outcome=yes_outcome, include_binary_features=include_binary_features)
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
) -> tuple[list[LOBSequence], list[str], object]:
    """Build per-market normalized sequences for the top-N longest markets.

    Each market's windowed features are computed on that market alone, so no
    feature straddles a market boundary. Normalization is fit once on the
    concatenation of all markets (boundaries do not affect per-feature mean and
    std), then applied per market. Returns the normalized sequences (longest
    first), their slugs, and the shared stats.
    """
    bt = build_timeline(data_dir=data_dir, hours=hours, intervals=intervals)
    slugs = [market_slug] if market_slug else pick_top_markets(bt, max_markets)
    raw_seqs: list[LOBSequence] = []
    for slug in slugs:
        settlement = bt.settlements.get(slug)
        yes_outcome = _settlement_yes_outcome(settlement)
        raw_seqs.append(
            extract_features(
                bt.timeline, slug, yes_outcome=yes_outcome,
                include_binary_features=include_binary_features,
            )
        )
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

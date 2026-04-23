"""
Data Loader - Reads SQLite/CSV/Parquet into a unified timeline.

Reuses loading patterns from collectors/export_data.py but builds structures
optimized for the backtest engine: a sorted timeline of ticks, a settlement
map, and market lifecycle list.

Note: pandas is imported lazily inside functions to avoid crashes on systems
with broken numpy builds (e.g., MINGW-W64).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .strategy import (
    MarketLifecycle,
    MarketStatus,
    OrderBookLevel,
    OrderBookSnapshot,
    Settlement,
    StoredBook,
    Token,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Default paths relative to polymarket-bot/
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "live"

# INTERVALS from the collector
INTERVALS = {
    "5m": {"seconds": 300, "prefix": ["btc-updown-5m", "sol-updown-5m", "eth-updown-5m"]},
    "15m": {"seconds": 900, "prefix": ["btc-updown-15m", "sol-updown-15m", "eth-updown-15m"]},
    "hourly": {"seconds": 3600, "prefix": ["bitcoin-up-or-down", "solana-up-or-down", "ethereum-up-or-down"]},
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_HOURLY_PATTERN = re.compile(
    r"^(bitcoin|solana|ethereum)-up-or-down-([a-z]+)-(\d+)-(\d{4})-(\d+)(am|pm)-et$"
)


def _import_pandas():
    import pandas as pd
    return pd


# - Tick data structure -


@dataclass
class TickData:
    """All data available at a single 1-second tick."""
    ts_sec: int  # unix epoch seconds

    # Polymarket prices per market: slug -> price row dict
    market_prices: dict[str, dict] = field(default_factory=dict)

    # Order books per market: slug -> {yes_book: OBS, no_book: OBS}
    order_books: dict[str, dict] = field(default_factory=dict)

    # Last order book timestamp per market (for staleness check)
    book_timestamps: dict[str, int] = field(default_factory=dict)

    # Binance reference prices per asset.
    btc_mid: float = 0.0
    btc_spread: float = 0.0
    eth_mid: float = 0.0
    eth_spread: float = 0.0
    sol_mid: float = 0.0
    sol_spread: float = 0.0

    # Chainlink oracle prices per asset (source of truth for settlement).
    chainlink_btc: float = 0.0
    chainlink_eth: float = 0.0
    chainlink_sol: float = 0.0


# - Raw data loaders (adapted from export_data.py) -


def load_market_prices(
    db_path: Path,
    start_us: int | None = None,
    end_us: int | None = None,
) -> pd.DataFrame:
    """Load market_prices table from SQLite, optionally filtered by time range."""
    pd = _import_pandas()
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    query = "SELECT * FROM market_prices"
    params: list = []
    if start_us is not None or end_us is not None:
        clauses = []
        if start_us is not None:
            clauses.append("timestamp_us >= ?")
            params.append(start_us)
        if end_us is not None:
            clauses.append("timestamp_us <= ?")
            params.append(end_us)
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp_us"
    df = pd.read_sql_query(query, conn, params=params or None)
    conn.close()
    if not df.empty:
        df["ts_sec"] = df["timestamp_us"] // 1_000_000
    return df


def load_orderbooks(books_dir: Path) -> pd.DataFrame:
    """Load order book snapshots from CSV files (and legacy JSONL)."""
    pd = _import_pandas()
    frames = []

    for path in sorted(books_dir.glob("*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass

    # Legacy JSONL format
    for path in sorted(books_dir.glob("*.jsonl")):
        jsonl_rows = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                yes_book = rec.get("yes_book", {})
                no_book = rec.get("no_book", {})

                def _sort_bids(orders):
                    return sorted(
                        [[float(b["price"]), float(b["size"])] for b in orders],
                        key=lambda x: -x[0],
                    )

                def _sort_asks(orders):
                    return sorted(
                        [[float(a["price"]), float(a["size"])] for a in orders],
                        key=lambda x: x[0],
                    )

                yb = _sort_bids(yes_book.get("bids", []))
                ya = _sort_asks(yes_book.get("asks", []))
                nb = _sort_bids(no_book.get("bids", []))
                na = _sort_asks(no_book.get("asks", []))

                jsonl_rows.append({
                    "timestamp_us": rec["timestamp_us"],
                    "interval": rec.get("interval", "15m"),
                    "market_slug": rec["market_slug"],
                    "yes_bids_json": json.dumps(yb),
                    "yes_asks_json": json.dumps(ya),
                    "no_bids_json": json.dumps(nb),
                    "no_asks_json": json.dumps(na),
                    "yes_best_bid": yb[0][0] if yb else 0,
                    "yes_best_ask": ya[0][0] if ya else 0,
                    "no_best_bid": nb[0][0] if nb else 0,
                    "no_best_ask": na[0][0] if na else 0,
                    "yes_n_bids": len(yb),
                    "yes_n_asks": len(ya),
                    "no_n_bids": len(nb),
                    "no_n_asks": len(na),
                    "yes_total_bid_size": sum(b[1] for b in yb),
                    "yes_total_ask_size": sum(a[1] for a in ya),
                    "no_total_bid_size": sum(b[1] for b in nb),
                    "no_total_ask_size": sum(a[1] for a in na),
                })
        if jsonl_rows:
            frames.append(pd.DataFrame(jsonl_rows))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp_us").reset_index(drop=True)
    if not df.empty:
        df["ts_sec"] = df["timestamp_us"] // 1_000_000
    return df


def load_binance_lob(binance_dir: Path) -> pd.DataFrame:
    """Load Binance LOB Parquet files."""
    pd = _import_pandas()
    parquet_files = sorted(binance_dir.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed - skipping Binance LOB data")
        return pd.DataFrame()

    tables = []
    for f in parquet_files:
        try:
            tables.append(pq.read_table(f))
        except Exception:
            pass

    if not tables:
        return pd.DataFrame()

    combined = pa.concat_tables(tables)
    df = combined.to_pandas()
    df["ts_sec"] = df["timestamp_us"] // 1_000_000

    # Classify asset by price magnitude - the three assets (BTC, ETH, SOL)
    # are orders of magnitude apart so this is unambiguous.
    if "bid_price_1" in df.columns:
        df["asset"] = df["bid_price_1"].apply(
            lambda x: "BTC" if x > 10_000 else ("ETH" if x > 500 else "SOL")
        )
    return df


def load_chainlink_prices(
    db_path: Path,
    start_us: int | None = None,
    end_us: int | None = None,
) -> pd.DataFrame:
    """Load Chainlink BTC/USD prices from rtds_prices table."""
    pd = _import_pandas()
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='rtds_prices'"
    ).fetchall()
    if not tables:
        conn.close()
        return pd.DataFrame()

    query = "SELECT * FROM rtds_prices WHERE source='chainlink'"
    params: list = []
    if start_us is not None:
        query += " AND timestamp_us >= ?"
        params.append(start_us)
    if end_us is not None:
        query += " AND timestamp_us <= ?"
        params.append(end_us)
    query += " ORDER BY timestamp_us"
    df = pd.read_sql_query(query, conn, params=params or None)
    conn.close()

    if not df.empty:
        df["ts_sec"] = df["timestamp_us"] // 1_000_000
    return df


def load_market_outcomes(db_path: Path) -> dict[str, str]:
    """Load market_outcomes table if available. Returns slug -> outcome."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='market_outcomes'"
    ).fetchall()
    if not tables:
        conn.close()
        return {}
    cur = conn.execute(
        "SELECT market_slug, outcome FROM market_outcomes WHERE status='resolved'"
    )
    outcomes = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return outcomes


# - Market lifecycle parsing -


def parse_slug_lifecycle(slug: str) -> MarketLifecycle | None:
    """Parse market slug to determine interval, start, end timestamps."""
    # Try unix-timestamp-based slugs first (5m, 15m)
    for interval, cfg in INTERVALS.items():
        if interval == "hourly":
            continue
        prefixes = cfg["prefix"] if isinstance(cfg["prefix"], list) else [cfg["prefix"]]
        for prefix in prefixes:
            pattern = rf"^{re.escape(prefix)}-(\d+)$"
            m = re.match(pattern, slug)
            if m:
                start_ts = int(m.group(1))
                end_ts = start_ts + cfg["seconds"]
                return MarketLifecycle(
                    market_slug=slug,
                    interval=interval,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )

    # Try hourly date-based slug
    m = _HOURLY_PATTERN.match(slug)
    if m:
        _asset, month_name, day, year, hour, ampm = m.groups()
        month = _MONTHS.get(month_name)
        if month is None:
            return None
        hour_24 = int(hour)
        if ampm == "pm" and hour_24 != 12:
            hour_24 += 12
        elif ampm == "am" and hour_24 == 12:
            hour_24 = 0

        from datetime import datetime, timezone as tz
        import zoneinfo
        try:
            et = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            et = tz.utc
        try:
            dt_et = datetime(int(year), month, int(day), hour_24, 0, 0, tzinfo=et)
        except (ValueError, OverflowError):
            return None
        start_ts = int(dt_et.timestamp())
        end_ts = start_ts + 3600
        return MarketLifecycle(
            market_slug=slug,
            interval="hourly",
            start_ts=start_ts,
            end_ts=end_ts,
        )

    return None


# - Settlement computation -


def _asset_from_slug(slug: str) -> str:
    """Extract asset symbol from a market slug (e.g. 'btc-updown-5m-...' -> 'BTC')."""
    s = slug.lower()
    if s.startswith("btc-") or s.startswith("bitcoin-"):
        return "BTC"
    elif s.startswith("sol-") or s.startswith("solana-"):
        return "SOL"
    elif s.startswith("eth-") or s.startswith("ethereum-"):
        return "ETH"
    return "BTC"  # default fallback


def compute_settlements(
    lifecycles: list[MarketLifecycle],
    chainlink_df: pd.DataFrame,
    known_outcomes: dict[str, str] | None = None,
) -> dict[str, Settlement]:
    """
    Compute settlement outcomes from Chainlink oracle prices.

    Price at end >= price at start -> YES wins.
    Filters Chainlink prices by asset (BTC/SOL/ETH) based on market slug.
    Cross-references with market_outcomes table if available.
    """
    if known_outcomes is None:
        known_outcomes = {}

    settlements: dict[str, Settlement] = {}

    # Pre-filter Chainlink by asset symbol for efficiency
    asset_dfs: dict[str, Any] = {}
    if not chainlink_df.empty and "symbol" in chainlink_df.columns:
        for sym in chainlink_df["symbol"].unique():
            asset_key = sym.split("/")[0].upper() if "/" in str(sym) else str(sym).upper()
            asset_dfs[asset_key] = chainlink_df[chainlink_df["symbol"] == sym]
    elif not chainlink_df.empty:
        # Legacy: no symbol column, assume all BTC
        asset_dfs["BTC"] = chainlink_df

    for lc in lifecycles:
        slug = lc.market_slug

        # Try known outcomes first (from market_outcomes table)
        if slug in known_outcomes:
            outcome_str = known_outcomes[slug]
            outcome = Token.YES if outcome_str == "YES" else Token.NO
            settlements[slug] = Settlement(
                market_slug=slug,
                interval=lc.interval,
                outcome=outcome,
                start_ts=lc.start_ts,
                end_ts=lc.end_ts,
            )
            continue

        # Get the correct Chainlink price series for this asset
        asset = _asset_from_slug(slug)
        asset_cl = asset_dfs.get(asset, chainlink_df if not chainlink_df.empty else None)
        if asset_cl is None or asset_cl.empty:
            continue

        # Get Chainlink price closest to market start
        start_mask = (asset_cl["ts_sec"] >= lc.start_ts - 5) & (
            asset_cl["ts_sec"] <= lc.start_ts + 5
        )
        start_prices = asset_cl[start_mask]

        # Get Chainlink price closest to market end
        end_mask = (asset_cl["ts_sec"] >= lc.end_ts - 5) & (
            asset_cl["ts_sec"] <= lc.end_ts + 5
        )
        end_prices = asset_cl[end_mask]

        if start_prices.empty or end_prices.empty:
            # Fallback: use nearest available price
            start_idx = (asset_cl["ts_sec"] - lc.start_ts).abs().idxmin()
            end_idx = (asset_cl["ts_sec"] - lc.end_ts).abs().idxmin()
            open_price = float(asset_cl.loc[start_idx, "price"])
            close_price = float(asset_cl.loc[end_idx, "price"])
        else:
            open_price = float(
                start_prices.loc[
                    (start_prices["ts_sec"] - lc.start_ts).abs().idxmin(), "price"
                ]
            )
            close_price = float(
                end_prices.loc[
                    (end_prices["ts_sec"] - lc.end_ts).abs().idxmin(), "price"
                ]
            )

        outcome = Token.YES if close_price >= open_price else Token.NO
        settlements[slug] = Settlement(
            market_slug=slug,
            interval=lc.interval,
            outcome=outcome,
            start_ts=lc.start_ts,
            end_ts=lc.end_ts,
            chainlink_open=open_price,
            chainlink_close=close_price,
        )

    return settlements


# - Unified timeline builder -


@dataclass
class BacktestData:
    """Complete data bundle for a backtest run."""
    timeline: list[TickData]
    lifecycles: list[MarketLifecycle]
    settlements: dict[str, Settlement]
    start_ts: int
    end_ts: int


def _synthesize_book(
    bid: float, ask: float, base_size: float = 50.0, n_levels: int = 5,
) -> OrderBookSnapshot:
    """Build a synthetic order book from top-of-book bid/ask prices."""
    if bid <= 0 and ask <= 0:
        return OrderBookSnapshot()
    bids: list[OrderBookLevel] = []
    asks: list[OrderBookLevel] = []
    depth_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    for i in range(n_levels):
        mult = depth_multipliers[i] if i < len(depth_multipliers) else 3.0
        level_size = round(base_size * mult, 1)
        bid_price = round(bid - i * 0.01, 4)
        ask_price = round(ask + i * 0.01, 4)
        if bid_price >= 0.01:
            bids.append(OrderBookLevel(bid_price, level_size))
        if ask_price <= 0.99:
            asks.append(OrderBookLevel(ask_price, level_size))
    return OrderBookSnapshot(bids=tuple(bids), asks=tuple(asks))


def build_timeline(
    data_dir: Path | None = None,
    db_path: Path | None = None,
    books_dir: Path | None = None,
    binance_dir: Path | None = None,
    intervals: list[str] | None = None,
    hours: float | None = None,
    assets: list[str] | None = None,
) -> BacktestData:
    """
    Build a unified timeline from all data sources.

    Returns a BacktestData with one TickData per second, covering the full
    time range of available Polymarket price data.

    Args:
        hours: If set, only load the last N hours of data. 0 = full dataset.
        assets: If set, filter markets to only these assets (e.g. ["BTC"]).
                Massive speedup when the strategy only trades one asset.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    if db_path is None:
        db_path = data_dir / "polymarket.db"
    if books_dir is None:
        books_dir = data_dir / "polymarket_books"
    if binance_dir is None:
        binance_dir = data_dir / "binance_lob"
    if intervals is None:
        intervals = ["5m", "15m", "hourly"]

    # Compute time-range filter (microseconds for SQL queries)
    start_us: int | None = None
    end_us: int | None = None
    if hours is not None and hours > 0 and db_path.exists():
        # Find the max timestamp in the DB to compute the window
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT MAX(timestamp_us) FROM market_prices"
        ).fetchone()
        conn.close()
        if row and row[0]:
            end_us = int(row[0])
            start_us = end_us - int(hours * 3600 * 1_000_000)
            logger.info(
                f"Time filter: last {hours}h "
                f"({start_us // 1_000_000} - {end_us // 1_000_000})"
            )

    # Load raw data
    logger.info("Loading market prices...")
    prices_df = load_market_prices(db_path, start_us=start_us, end_us=end_us)
    logger.info("Loading order books...")
    books_df = load_orderbooks(books_dir)
    logger.info("Loading Binance LOB...")
    binance_df = load_binance_lob(binance_dir)
    logger.info("Loading Chainlink prices...")
    chainlink_df = load_chainlink_prices(db_path, start_us=start_us, end_us=end_us)
    known_outcomes = load_market_outcomes(db_path)

    if prices_df.empty:
        logger.warning("No market price data found")
        return BacktestData(
            timeline=[], lifecycles=[], settlements={}, start_ts=0, end_ts=0
        )

    # Filter to requested intervals
    prices_df = prices_df[prices_df["interval"].isin(intervals)]
    if prices_df.empty:
        logger.warning(f"No data for intervals {intervals}")
        return BacktestData(
            timeline=[], lifecycles=[], settlements={}, start_ts=0, end_ts=0
        )

    # Filter to requested assets - eliminates SOL/ETH work when strategy is BTC-only
    if assets:
        asset_set = {a.upper() for a in assets}
        prices_df = prices_df[prices_df["market_slug"].apply(
            lambda s: _asset_from_slug(s) in asset_set
        )]
        if prices_df.empty:
            logger.warning(f"No data for assets {assets}")
            return BacktestData(
                timeline=[], lifecycles=[], settlements={}, start_ts=0, end_ts=0
            )

    # Discover all market lifecycles from slugs
    all_slugs = prices_df["market_slug"].unique()
    lifecycles: list[MarketLifecycle] = []
    for slug in all_slugs:
        lc = parse_slug_lifecycle(slug)
        if lc:
            lifecycles.append(lc)
    lifecycles.sort(key=lambda x: x.start_ts)

    logger.info(f"Found {len(lifecycles)} markets across {intervals}")

    # Compute settlements
    settlements = compute_settlements(lifecycles, chainlink_df, known_outcomes)
    logger.info(f"Computed {len(settlements)} settlements")

    # Determine time range
    global_start = int(prices_df["ts_sec"].min())
    global_end = int(prices_df["ts_sec"].max())

    # Aggregate Binance to per-second, one dict per asset.
    # Mid and spread are derived on the fly from bid_price_1/ask_price_1
    # since the Parquet schema does not carry a mid_price column. The
    # "asset" column is set upstream in load_binance_lob.
    binance_by_sec: dict[str, dict[int, tuple[float, float]]] = {
        "BTC": {}, "ETH": {}, "SOL": {},
    }
    if not binance_df.empty and "bid_price_1" in binance_df.columns and "ask_price_1" in binance_df.columns:
        has_asset_col = "asset" in binance_df.columns
        for asset in ("BTC", "ETH", "SOL"):
            asset_lob = binance_df[binance_df["asset"] == asset] if has_asset_col else (
                binance_df if asset == "BTC" else binance_df.iloc[0:0]
            )
            if asset_lob.empty:
                continue
            asset_lob = asset_lob.assign(
                _mid=(asset_lob["bid_price_1"] + asset_lob["ask_price_1"]) / 2,
                _spread=(asset_lob["ask_price_1"] - asset_lob["bid_price_1"]),
            )
            agg = asset_lob.groupby("ts_sec").agg(
                mid=("_mid", "last"),
                spread=("_spread", "last"),
            )
            binance_by_sec[asset] = dict(
                zip(
                    agg.index.astype(int).tolist(),
                    zip(
                        agg["mid"].astype(float).tolist(),
                        agg["spread"].astype(float).tolist(),
                    ),
                )
            )

    # Aggregate Chainlink to per-second, one dict per asset.
    # Filter by symbol before groupby so `.last()` only sees rows for a
    # single asset within each bucket.
    chainlink_by_sec: dict[str, dict[int, float]] = {"BTC": {}, "ETH": {}, "SOL": {}}
    if not chainlink_df.empty and "symbol" in chainlink_df.columns:
        _symbol_to_asset = {"BTC/USD": "BTC", "ETH/USD": "ETH", "SOL/USD": "SOL"}
        for sym, asset in _symbol_to_asset.items():
            sub = chainlink_df[chainlink_df["symbol"] == sym]
            if sub.empty:
                continue
            agg = sub.groupby("ts_sec").agg(price=("price", "last"))
            chainlink_by_sec[asset] = dict(
                zip(
                    agg.index.astype(int).tolist(),
                    agg["price"].astype(float).tolist(),
                )
            )
    elif not chainlink_df.empty:
        # Fall back to single-symbol rows (rtds_prices with no symbol column).
        agg = chainlink_df.groupby("ts_sec").agg(price=("price", "last"))
        chainlink_by_sec["BTC"] = dict(
            zip(
                agg.index.astype(int).tolist(),
                agg["price"].astype(float).tolist(),
            )
        )

    # Group prices by ts_sec for fast lookup.
    # `to_dict('records')` is ~20x faster than iterrows() on large frames.
    prices_grouped: dict[int, dict] = {}
    for rec in prices_df.to_dict("records"):
        ts = int(rec["ts_sec"])
        slug = rec["market_slug"]
        bucket = prices_grouped.get(ts)
        if bucket is None:
            bucket = {}
            prices_grouped[ts] = bucket
        bucket[slug] = rec

    # Release the prices DataFrame - the grouped dict has everything we need.
    # Without this, pandas holds ~1 GB of row data alive through the JSON-parse
    # step below, causing GC thrashing on OneDrive-synced directories.
    # We rely on refcount-based cleanup; no explicit gc.collect() (scanning
    # the ~9M book-level objects takes over two minutes on big datasets).
    del prices_df
    del binance_df, chainlink_df
    import gc

    # Filter books_df to slugs actually in scope. After interval+asset
    # filtering on prices_df, `lifecycles` is the authoritative list of
    # markets the engine will see. Books for any other slug are wasted
    # parse work.
    if not books_df.empty and "market_slug" in books_df.columns:
        lifecycle_slugs = {lc.market_slug for lc in lifecycles}
        if lifecycle_slugs:
            books_df = books_df[books_df["market_slug"].isin(lifecycle_slugs)]

    # Build pre-parsed book snapshots indexed by (slug, ts) for O(1)
    # forward-fill. Use one groupby() pass rather than scanning the frame
    # per slug - the per-slug approach is O(N*S) on 8k+ slugs.
    #
    # No pickle cache: 1.75M dataclass instances pickle to ~6 GB of Python
    # object overhead, so re-parsing on cold start is faster. The del/gc
    # cleanup above keeps cold-parse near raw CPU cost (~90s on full train).
    import bisect
    books_by_slug: dict[str, Any] = {}
    book_ts_index: dict[str, list[int]] = {}
    book_snapshots: dict[str, dict[int, dict]] = {}

    if not books_df.empty:
        logger.info(f"Parsing {len(books_df):,} order-book snapshots...")
        # Pre-extract the columns we need to avoid pandas .get() overhead per row.
        book_cols = [
            "ts_sec", "market_slug",
            "yes_bids_json", "yes_asks_json",
            "no_bids_json", "no_asks_json",
        ]
        available = [c for c in book_cols if c in books_df.columns]
        slim = books_df[available].sort_values(["market_slug", "ts_sec"])
        parsed = 0
        progress_every = max(len(slim) // 5, 1)

        # Disable GC during the parse. We create millions of small objects
        # (OrderBookLevel tuples, OrderBookSnapshot dataclasses); with GC
        # enabled, each collection pass scans the whole prices_grouped dict
        # as well, which is a ~2-3x slowdown on large datasets.
        gc.disable()
        try:
            for slug, grp in slim.groupby("market_slug", sort=False):
                ts_list: list[int] = []
                snap_dict: dict[int, dict] = {}
                col_idx = {c: i for i, c in enumerate(grp.columns)}
                ts_i = col_idx["ts_sec"]
                ybi = col_idx.get("yes_bids_json")
                yai = col_idx.get("yes_asks_json")
                nbi = col_idx.get("no_bids_json")
                nai = col_idx.get("no_asks_json")
                for row in grp.itertuples(index=False, name=None):
                    bts = int(row[ts_i])
                    ts_list.append(bts)
                    snap_dict[bts] = StoredBook(
                        yes_book=OrderBookSnapshot.from_json(
                            str(row[ybi]) if ybi is not None else "[]",
                            str(row[yai]) if yai is not None else "[]",
                        ),
                        no_book=OrderBookSnapshot.from_json(
                            str(row[nbi]) if nbi is not None else "[]",
                            str(row[nai]) if nai is not None else "[]",
                        ),
                        book_ts=bts,
                    )
                    parsed += 1
                    if parsed % progress_every == 0:
                        logger.info(f"  parsed {parsed:,}/{len(slim):,} books")
                books_by_slug[slug] = None  # don't keep pandas group alive
                book_ts_index[slug] = ts_list
                book_snapshots[slug] = snap_dict
        finally:
            gc.enable()

        # Release the raw books DataFrame - parsed snapshots have all we need.
        # NO explicit gc.collect() - scanning the ~9M book-level objects
        # we just created would cost 2+ minutes and saves nothing material.
        del books_df, slim

        # Move everything created so far to the "permanent generation"
        # (gc.freeze). Subsequent collections during the tick loop only
        # scan newly-created objects, not the 1M+ immutable book snapshots
        # we just built. Removes the 1-2 minute stall we were seeing
        # between "parsed" and the first timeline-progress log.
        gc.freeze()

    # Track which slugs have JSONL books vs need synthetic
    slugs_with_books = set(books_by_slug.keys())
    slugs_need_synthetic = set(all_slugs) - slugs_with_books
    if slugs_need_synthetic:
        logger.info(
            f"Synthesizing order books for {len(slugs_need_synthetic)} markets "
            f"(no JSONL data; using SQLite bid/ask)"
        )

    # Build timeline tick by tick
    total_secs = global_end - global_start + 1
    logger.info(f"Building timeline: {total_secs} seconds, {len(lifecycles)} markets")
    timeline: list[TickData] = []

    # Sort lifecycle events so we only look at *currently active* markets per tick.
    # Without this, the inner slug loop is O(T * N) = billions of iterations on
    # the full train set (641K seconds * 8.4K markets).
    starts_sorted = sorted(
        ((lc.start_ts, lc.market_slug) for lc in lifecycles), key=lambda x: x[0]
    )
    ends_sorted = sorted(
        ((lc.end_ts, lc.market_slug) for lc in lifecycles), key=lambda x: x[0]
    )
    start_idx = 0
    end_idx = 0
    active_slugs: set[str] = set()

    # Track the last observed value for each asset so we can forward-fill.
    last_binance: dict[str, tuple[float, float]] = {
        "BTC": (0.0, 0.0), "ETH": (0.0, 0.0), "SOL": (0.0, 0.0),
    }
    last_chainlink_by_asset: dict[str, float] = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
    # Map of slug -> {yes_book, no_book, book_ts}.
    last_books: dict[str, dict] = {}

    progress_step = max(total_secs // 10, 1)

    for ts in range(global_start, global_end + 1):
        # Advance active set
        while start_idx < len(starts_sorted) and starts_sorted[start_idx][0] <= ts:
            active_slugs.add(starts_sorted[start_idx][1])
            start_idx += 1
        while end_idx < len(ends_sorted) and ends_sorted[end_idx][0] < ts:
            expired = ends_sorted[end_idx][1]
            active_slugs.discard(expired)
            last_books.pop(expired, None)
            end_idx += 1

        tick = TickData(ts_sec=ts)

        # Market prices (only available at recorded ticks)
        if ts in prices_grouped:
            tick.market_prices = prices_grouped[ts]

        # Order books: forward-fill from last known snapshot using bisect
        # Only iterate the ~50-200 currently active markets, not all 8.4K.
        for slug in active_slugs:
            if slug in book_ts_index:
                ts_list = book_ts_index[slug]
                idx = bisect.bisect_right(ts_list, ts) - 1
                if idx >= 0:
                    book_ts = ts_list[idx]
                    snap = book_snapshots[slug][book_ts]
                    last_books[slug] = snap

            if slug in last_books:
                # Reuse the parsed StoredBook tuple - no per-tick allocation.
                snap = last_books[slug]
                tick.order_books[slug] = snap
                tick.book_timestamps[slug] = snap.book_ts

            # Synthesize order books from bid/ask when no JSONL data
            elif slug in slugs_need_synthetic and slug in tick.market_prices:
                pdict = tick.market_prices[slug]
                yes_bid = float(pdict.get("yes_bid", 0))
                yes_ask = float(pdict.get("yes_ask", 0))
                no_bid = float(pdict.get("no_bid", 0))
                no_ask = float(pdict.get("no_ask", 0))
                if yes_bid > 0 or yes_ask > 0:
                    snap = StoredBook(
                        yes_book=_synthesize_book(yes_bid, yes_ask),
                        no_book=_synthesize_book(no_bid, no_ask),
                        book_ts=ts,
                    )
                    last_books[slug] = snap
                    tick.order_books[slug] = snap
                    tick.book_timestamps[slug] = ts

        # Forward-fill Binance mid and spread for each asset.
        for asset in ("BTC", "ETH", "SOL"):
            asset_dict = binance_by_sec[asset]
            if ts in asset_dict:
                last_binance[asset] = asset_dict[ts]
        tick.btc_mid, tick.btc_spread = last_binance["BTC"]
        tick.eth_mid, tick.eth_spread = last_binance["ETH"]
        tick.sol_mid, tick.sol_spread = last_binance["SOL"]

        # Forward-fill the Chainlink oracle price for each asset.
        for asset in ("BTC", "ETH", "SOL"):
            asset_dict = chainlink_by_sec[asset]
            if ts in asset_dict:
                last_chainlink_by_asset[asset] = asset_dict[ts]
        tick.chainlink_btc = last_chainlink_by_asset["BTC"]
        tick.chainlink_eth = last_chainlink_by_asset["ETH"]
        tick.chainlink_sol = last_chainlink_by_asset["SOL"]

        timeline.append(tick)

        if len(timeline) % progress_step == 0:
            pct = len(timeline) / total_secs * 100
            logger.info(
                f"  timeline build: {len(timeline):,}/{total_secs:,} "
                f"({pct:.0f}%, active={len(active_slugs)})"
            )

    logger.info(f"Timeline built: {len(timeline)} ticks, {len(lifecycles)} markets")
    return BacktestData(
        timeline=timeline,
        lifecycles=lifecycles,
        settlements=settlements,
        start_ts=global_start,
        end_ts=global_end,
    )

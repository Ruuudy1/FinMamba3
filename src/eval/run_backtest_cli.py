"""CLI wrapper around eval/backtest.run_backtest for a frozen world model.

Loads a pretrained world-model checkpoint, instantiates a
GreedyDirectionPolicy, builds a PolymarketLOBEnv from the val split, runs the
backtest, and writes BacktestMetrics as JSON. Optional --regime-split filters
the timeline by time or by realized-volatility quantile so the same harness
covers the non-stationarity A/B test.

Example
-------
    python -m eval.run_backtest_cli \\
        --world-checkpoint saved_models/lob/LOB/1iemugot/ckpt/world_model.pth \\
        --config src/config_files/configure_lob.yaml \\
        --data-val data/validation \\
        --max-steps 5000 \\
        --regime-split none \\
        --out reports/backtest_1iemugot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a backtest with a frozen world model")
    p.add_argument("--world-checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-val", required=True, type=Path)
    p.add_argument("--hours-val", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.005,
                   help="Direction threshold passed to GreedyDirectionPolicy")
    p.add_argument("--regime-split", default="none",
                   help="One of: none, time:<unix_ts>, volatility:<quantile>")
    p.add_argument("--out", type=Path, default=Path("reports/backtest.json"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def _device_from_arg(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _filter_backtest_data(bt, spec: str):
    """Apply a regime split to the BacktestData lifecycle list."""
    from eval.regime_split import time_split, volatility_split

    if spec == "none":
        return bt, "all"
    if spec.startswith("time:"):
        cutoff = float(spec.split(":", 1)[1])
        result = time_split(bt.lifecycles, cutoff)
        keep = {m.market_slug for m in result.test_markets}
        bt.lifecycles = [m for m in bt.lifecycles if m.market_slug in keep]
        return bt, result.description
    if spec.startswith("volatility:"):
        quantile = float(spec.split(":", 1)[1])
        # Use a degenerate empty realized_vol map; volatility_split will keep all
        # markets in the train half. The CLI surface is here for completeness;
        # populating realized_vol per slug is left for a future iteration.
        result = volatility_split(bt.lifecycles, realized_vol={}, quantile=quantile)
        keep = {m.market_slug for m in result.test_markets}
        bt.lifecycles = [m for m in bt.lifecycles if m.market_slug in keep]
        return bt, result.description
    raise ValueError(f"Unknown --regime-split spec: {spec}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)

    import yaml
    from config_utils import DotDict, parse_args_and_update_config
    from envs.polymarket_lob_env import PolymarketLOBEnv
    from eval.backtest import GreedyDirectionPolicy, run_backtest
    from lob.backtester import build_timeline
    from sub_models.world_models import WorldModel

    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)
    cfg_raw = parse_args_and_update_config(cfg_raw, argv=[])
    cfg = DotDict(cfg_raw)

    wm = WorldModel(action_dim=13, config=cfg, device=device).to(device)
    state = torch.load(args.world_checkpoint, map_location=device, weights_only=False)
    sd = state.get("world_model", state.get("state_dict", state))
    wm.load_state_dict(sd, strict=False)
    wm.eval()

    bt = build_timeline(data_dir=args.data_val, hours=args.hours_val)
    bt, regime_desc = _filter_backtest_data(bt, args.regime_split)
    logger.info(f"loaded backtest data: regime={regime_desc}, lifecycles={len(bt.lifecycles)}")

    env = PolymarketLOBEnv(bt)
    policy = GreedyDirectionPolicy(
        wm, mid_index=80, threshold=args.threshold, device=str(device)
    )
    metrics = run_backtest(env, policy, max_steps=args.max_steps)
    logger.info(
        f"backtest done: total_return={metrics.total_return:.4f} "
        f"sharpe={metrics.sharpe:.4f} maxDD={metrics.max_drawdown:.4f} "
        f"trades={metrics.num_trades} win_rate={metrics.win_rate:.4f}"
    )

    summary = asdict(metrics)
    # The portfolio curve is large; trim it to length-256 for the JSON output.
    summary["portfolio_curve_len"] = len(metrics.portfolio_curve)
    if metrics.portfolio_curve:
        idx = np.linspace(0, len(metrics.portfolio_curve) - 1, num=min(256, len(metrics.portfolio_curve))).astype(int)
        summary["portfolio_curve"] = [float(metrics.portfolio_curve[i]) for i in idx]
    summary["regime"] = regime_desc
    summary["max_steps"] = args.max_steps
    summary["threshold"] = args.threshold
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Held-out PnL backtest harness for a frozen world model.

Provides a deterministic policy that consumes the world-model latent and
emits a discrete action. Used to report PnL, Sharpe, and max drawdown on
held-out markets. Intended to be invoked after Phase A pretraining as a
standalone evaluator; the result is the headline real-world metric the
paper reports alongside reconstruction MSE and direction accuracy.

Example
-------
    from envs.polymarket_lob_env import PolymarketLOBEnv
    from eval.backtest import run_backtest, GreedyDirectionPolicy

    env = PolymarketLOBEnv(test_data)
    metrics = run_backtest(
        env,
        GreedyDirectionPolicy(world_model, mid_index=80, threshold=0.005),
        max_steps=10_000,
    )
    print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: np.ndarray) -> int: ...


@dataclass
class BacktestMetrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    portfolio_curve: list[float] = field(default_factory=list)


def run_backtest(env, policy: Policy, max_steps: int = 10_000) -> BacktestMetrics:
    """Run a policy through the environment and report PnL metrics.

    Parameters
    ----------
    env
        Gymnasium env with the PolymarketLOBEnv interface.
    policy
        Object implementing reset() and act(observation) -> int.
    max_steps
        Maximum number of steps to roll out before stopping.
    """
    obs, _ = env.reset()
    policy.reset()
    portfolio: list[float] = []
    returns: list[float] = []
    num_trades = 0
    wins = 0

    for _ in range(max_steps):
        action = int(policy.act(obs))
        obs, reward, done, _, info = env.step(action)
        portfolio.append(float(info.get("portfolio_value", 0.0)))
        returns.append(float(reward))
        if action != 0:
            num_trades += 1
            if reward > 0:
                wins += 1
        if done:
            break

    total_return = portfolio[-1] / portfolio[0] - 1.0 if portfolio and portfolio[0] > 0 else 0.0
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size > 0 and arr.std() > 1e-9:
        sharpe = float(np.sqrt(252.0 * 24.0 * 60.0 * 60.0) * arr.mean() / arr.std())
    else:
        sharpe = 0.0
    if portfolio:
        peak = np.maximum.accumulate(np.asarray(portfolio))
        drawdown = float(((peak - portfolio) / np.maximum(peak, 1e-9)).max())
    else:
        drawdown = 0.0
    win_rate = float(wins / max(1, num_trades))
    return BacktestMetrics(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=drawdown,
        num_trades=num_trades,
        win_rate=win_rate,
        portfolio_curve=portfolio,
    )


class GreedyDirectionPolicy:
    """Pick action based on the world-model direction-head argmax.

    Buy YES on predicted up, sell YES (equivalent to buy NO) on predicted
    down, no-op on flat. Uses the smallest size bucket so position sizing
    does not bias the metric.
    """

    def __init__(
        self,
        world_model,
        mid_index: int,
        threshold: float = 0.005,
        device: str = "cuda",
    ) -> None:
        self.world_model = world_model
        self.mid_index = int(mid_index)
        self.threshold = float(threshold)
        self.device = device
        self._latent = None

    def reset(self) -> None:
        self._latent = None

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        if not hasattr(self.world_model, "direction_head") or self.world_model.direction_head is None:
            return 0
        flat = observation.reshape(-1).astype(np.float32)
        if flat.size < self.mid_index + 1:
            return 0
        x = torch.from_numpy(flat).to(self.device).reshape(1, 1, -1)
        latent = self.world_model.encode_obs(x)
        last = latent[:, -1:]
        seq = self.world_model.sequence_model(
            last,
            torch.zeros((1, 1), dtype=torch.long, device=self.device),
        )
        logits = self.world_model.direction_head(seq)
        cls = int(logits.argmax(dim=-1).item())
        if cls == 2:
            return 1
        if cls == 0:
            return 4
        return 0

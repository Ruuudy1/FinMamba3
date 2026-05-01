"""Phase B smoke test: 1k PPO steps against a frozen world model.

Demonstrates that the latent learned in Phase A is RL-trainable. Not a full
Phase B run; just enough to catch regressions in the env wiring or the
agent's interaction with the imagination buffer. Intended to be invoked
once per checkpoint as a CI-style sanity check, not as a training routine.

Example
-------
    python -m eval.phase_b_smoke \\
        --checkpoint saved_models/lob/LOB/run_001/ckpt/world_model.pth \\
        --config src/config_files/configure_lob.yaml \\
        --max-steps 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B smoke test")
    p.add_argument("--checkpoint", required=True, help="Path to pretrained world_model.pth")
    p.add_argument("--config", required=True, help="Path to configure_lob.yaml")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--data-train", default="data/train")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from agents import ActorCriticAgent
    from envs.polymarket_lob_env import PolymarketLOBEnv
    from lob.backtester.data_loader import build_timeline
    from sub_models.world_models import WorldModel
    import yaml

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    # Convert nested dict to dotted-attribute config consistent with train_lob.
    from types import SimpleNamespace

    def _ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
        return d

    cfg = _ns(cfg_dict)
    device = torch.device(args.device)

    timeline = build_timeline(args.data_train)
    env = PolymarketLOBEnv(timeline)

    world_model = WorldModel(action_dim=env.action_space.n, config=cfg, device=device).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    world_model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad = False

    agent = ActorCriticAgent(
        feat_dim=world_model.hidden_state_dim + world_model.stoch_flattened_dim,
        action_dim=env.action_space.n,
        config=cfg,
        device=device,
    )

    obs, _ = env.reset()
    rewards = []
    for step in range(args.max_steps):
        flat = torch.from_numpy(obs.reshape(-1)).to(device).float().unsqueeze(0).unsqueeze(0)
        latent = world_model.encode_obs(flat)
        seq_in = torch.zeros((1, 1), dtype=torch.long, device=device)
        dist_feat = world_model.sequence_model(latent, seq_in)
        feat = torch.cat([latent[:, -1:], dist_feat[:, -1:]], dim=-1)
        action, _ = agent.sample(feat)
        obs, reward, done, _, _ = env.step(int(action.item()))
        rewards.append(float(reward))
        if done:
            obs, _ = env.reset()

    avg = sum(rewards) / max(1, len(rewards))
    print(f"phase_b_smoke: {len(rewards)} steps, mean reward {avg:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

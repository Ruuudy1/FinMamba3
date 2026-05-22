"""Phase B: warm-start an RL agent in the world model's imagination.

Builds an ActorCriticAgent on top of a Phase-A pretrained world model and trains
it on imagined rollouts (DreamerV3 style) sampled from the LOB replay buffer. It
runs entirely in the world model's latent space, so it never touches the env
observation schema (the env emits a different multi-market vector than the world
model consumes); imagination is the architecturally correct Phase B path here.

GPU-only (the Mamba backbone needs CUDA). Two prerequisites for *meaningful*
learning, both tracked in handoff.md:
  1. The world model must be built with action_dim equal to the agent action
     space, so imagined actions can be fed back into the dynamics. Pass
     --agent-action-dim to match. A Phase-A checkpoint trained with a different
     action_dim loads with strict=False (the action stem is reinitialized).
  2. A reward signal is required. With Reward.Enabled=false the imagined reward is
     zero and the agent has nothing to optimize; enable the reward head and train
     it on env reward before expecting a useful policy.

Example
-------
    python -m eval.phase_b_smoke \\
        --checkpoint saved_models/lob/LOB/<run>/ckpt/world_model_best.pth \\
        --config src/config_files/configure_lob.yaml \\
        --data-train data/train --steps 200
"""
# region imports
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import torch
import yaml
# endregion
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B imagination trainer")
    p.add_argument("--checkpoint", required=True, help="Path to a Phase-A world_model checkpoint")
    p.add_argument("--config", required=True, help="Path to configure_lob.yaml")
    p.add_argument("--data-train", default="data/train")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--hours-train", type=float, default=6.0)
    p.add_argument(
        "--agent-action-dim", type=int, default=None,
        help="Agent action space; defaults to the PolymarketLOBEnv action count.",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    src_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_dir))
    from config_utils import DotDict
    from replay_buffer import ReplayBuffer
    from agents import ActorCriticAgent
    from envs.polymarket_lob_env import PolymarketLOBEnv
    from lob.backtester.data_loader import build_timeline
    from sub_models.world_models import WorldModel
    from train_lob import _build_sequences, _populate_buffer
    with open(args.config) as f:
        config = DotDict(yaml.safe_load(f))
    device = torch.device(args.device)
    backtest_data = build_timeline(data_dir=Path(args.data_train), hours=args.hours_train)
    env = PolymarketLOBEnv(backtest_data)
    action_dim = args.agent_action_dim if args.agent_action_dim is not None else int(env.action_space.n)
    logger.info(f"agent action_dim={action_dim}")
    world_model = WorldModel(action_dim=action_dim, config=config, device=device).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = world_model.load_state_dict(state.get("world_model", state), strict=False)
    if missing or unexpected:
        logger.warning(
            f"checkpoint load: {len(missing)} missing, {len(unexpected)} unexpected params "
            "(expected if the Phase-A action_dim differs; the action stem is reinitialized)."
        )
    world_model.eval()
    for param in world_model.parameters():
        param.requires_grad = False
    agent = ActorCriticAgent(config, action_dim, device)
    # Imagination needs real LOB context to encode, so populate a replay buffer from the same data.
    norm_path = src_dir.parent / "saved_models" / "lob" / "normalization.json"
    include_binary = config.Models.WorldModel.Encoder.BinaryMarketFeatures
    train_seq, _, _ = _build_sequences(
        Path(args.data_train), None, args.hours_train, norm_path,
        fit_stats=False, norm_clip=config.BasicSettings.NormClip,
        aggregate_only=False, include_binary_features=include_binary,
    )
    replay_buffer = ReplayBuffer(config, device=device)
    _populate_buffer(replay_buffer, train_seq)
    imagine_batch_size = config.JointTrainAgent.ImagineBatchSize
    context_len = config.JointTrainAgent.ImagineContextLength
    imagine_len = config.JointTrainAgent.ImagineBatchLength
    for step in range(args.steps):
        sample_obs, sample_action, _, _ = replay_buffer.sample(imagine_batch_size, context_len, imagine=True)
        imagined, imagined_action, old_logits, context_out, imagined_reward, imagined_term = world_model.imagine_data(
            agent, sample_obs, sample_action, imagine_batch_size, imagine_len,
            log_video=False, logger=None, global_step=step,
        )
        agent.update(
            imagined, imagined_action, old_logits, context_out, None, None,
            imagined_reward, imagined_term.float(), None, step,
        )
    logger.info(f"phase B: {args.steps} imagination updates complete")
    return 0
if __name__ == "__main__":
    sys.exit(main())

"""Shared training steps for Drama world-model updates."""

from __future__ import annotations

import numpy as np
import torch

from replay_buffer import ReplayBuffer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sub_models.world_models import WorldModel


_LOSS_NAMES = (
    "reconstruction_loss",
    "reward_loss",
    "termination_loss",
    "dynamics_loss",
    "dynamics_real_kl_div",
    "representation_loss",
    "representation_real_kl_div",
    "direction_loss",
    "hawkes_loss",
    "settlement_loss",
    "total_loss",
)


def train_world_model_step(
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    batch_size,
    batch_length,
    logger,
    epoch,
    global_step,
    accum_steps: int = 1,
):
    # Accumulate on-device tensors per epoch to avoid per-step GPU-CPU syncs.
    # Convert to floats once at the end of the function for logging.
    epoch_means: dict[str, list[float]] = {name: [] for name in _LOSS_NAMES}
    for e in range(epoch):
        accum_stacks: list[list[torch.Tensor]] = [[] for _ in _LOSS_NAMES]
        for a in range(accum_steps):
            obs, action, reward, termination = replay_buffer.sample(
                batch_size, batch_length, imagine=False
            )
            losses = world_model.update(
                obs,
                action,
                reward,
                termination,
                global_step=global_step,
                epoch_step=e,
                logger=logger,
                accum_steps=accum_steps,
                is_last_accum=(a == accum_steps - 1),
            )
            for i, v in enumerate(losses):
                accum_stacks[i].append(v)
        # Single GPU-to-CPU transfer per epoch instead of 8 per accum step.
        stacked = torch.stack([torch.stack(stack).mean() for stack in accum_stacks])
        means = stacked.detach().cpu().numpy()
        for name, value in zip(_LOSS_NAMES, means):
            epoch_means[name].append(float(value))
    if logger is not None:
        for name, values in epoch_means.items():
            logger.log(
                f"WorldModel/{name}",
                float(np.mean(values)) if values else 0.0,
                global_step=global_step,
            )

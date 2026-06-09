"""Shared training steps for Drama world-model updates."""
# region imports
from __future__ import annotations
import numpy as np
import torch
from finmamba3.replay_buffer import ReplayBuffer
from typing import TYPE_CHECKING
# endregion
if TYPE_CHECKING:
    from finmamba3.models.world_model import WorldModel
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
    "regime_loss",
    "film_gamma_dev",
    "film_beta_mag",
    "regime_entropy",
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
    log_every: int = 1,
):
    # Only sync losses to CPU and log on logging steps, so a host transfer does not stall the
    # GPU every step. On non-logging steps the update runs with no GPU-to-CPU sync at all.
    should_log = logger is not None and global_step % log_every == 0
    epoch_means: dict[str, list[float]] = {name: [] for name in _LOSS_NAMES}
    for e in range(epoch):
        accum_stacks: list[list[torch.Tensor]] = [[] for _ in _LOSS_NAMES]
        for a in range(accum_steps):
            obs, action, reward, termination, outcome, tte_frac, spot_dist = replay_buffer.sample(
                batch_size, batch_length, imagine=False, with_supervision=True
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
                outcome=outcome,
                time_to_expiry_frac=tte_frac,
                spot_signed_distance=spot_dist,
            )
            if should_log:
                for i, v in enumerate(losses):
                    accum_stacks[i].append(v)
        if should_log:
            stacked = torch.stack([torch.stack(stack).mean() for stack in accum_stacks])
            means = stacked.detach().cpu().numpy()
            for name, value in zip(_LOSS_NAMES, means):
                epoch_means[name].append(float(value))
    if should_log:
        mean_by_loss = {name: float(np.mean(values)) for name, values in epoch_means.items()}
        for name, mean_value in mean_by_loss.items():
            logger.log(f"WorldModel/{name}", mean_value, global_step=global_step)
        # These per-step losses otherwise only reach wandb; echo the key terms to stdout so the
        # trajectory is visible in the captured training log (and its HF logs/ upload) offline.
        print(
            f"[loss] step={global_step} "
            f"total={mean_by_loss['total_loss']:.3f} "
            f"recon={mean_by_loss['reconstruction_loss']:.3f} "
            f"dyn_kl={mean_by_loss['dynamics_loss']:.3f} "
            f"rep={mean_by_loss['representation_loss']:.3f} "
            f"dir={mean_by_loss['direction_loss']:.3f} "
            f"settle={mean_by_loss['settlement_loss']:.3f} "
            f"film_g={mean_by_loss['film_gamma_dev']:.4f} "
            f"film_b={mean_by_loss['film_beta_mag']:.4f} "
            f"reg_H={mean_by_loss['regime_entropy']:.3f}",
            flush=True,
        )

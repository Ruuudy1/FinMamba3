"""Shared training steps for Drama world-model updates."""

from __future__ import annotations

import numpy as np

from replay_buffer import ReplayBuffer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sub_models.world_models import WorldModel


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
    epoch_reconstruction_loss_list = []
    epoch_reward_loss_list = []
    epoch_termination_loss_list = []
    epoch_dynamics_loss_list = []
    epoch_dynamics_real_kl_div_list = []
    epoch_representation_loss_list = []
    epoch_representation_real_kl_div_list = []
    epoch_total_loss_list = []
    for e in range(epoch):
        accum_losses = [[], [], [], [], [], [], [], []]
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
                accum_losses[i].append(v)
        (
            reconstruction_loss,
            reward_loss,
            termination_loss,
            dynamics_loss,
            dynamics_real_kl_div,
            representation_loss,
            representation_real_kl_div,
            total_loss,
        ) = [np.mean(vals) for vals in accum_losses]

        epoch_reconstruction_loss_list.append(reconstruction_loss)
        epoch_reward_loss_list.append(reward_loss)
        epoch_termination_loss_list.append(termination_loss)
        epoch_dynamics_loss_list.append(dynamics_loss)
        epoch_dynamics_real_kl_div_list.append(dynamics_real_kl_div)
        epoch_representation_loss_list.append(representation_loss)
        epoch_representation_real_kl_div_list.append(representation_real_kl_div)
        epoch_total_loss_list.append(total_loss)
    if logger is not None:
        logger.log(
            "WorldModel/reconstruction_loss",
            np.mean(epoch_reconstruction_loss_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/reward_loss",
            np.mean(epoch_reward_loss_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/termination_loss",
            np.mean(epoch_termination_loss_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/dynamics_loss",
            np.mean(epoch_dynamics_loss_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/dynamics_real_kl_div",
            np.mean(epoch_dynamics_real_kl_div_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/representation_loss",
            np.mean(epoch_representation_loss_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/representation_real_kl_div",
            np.mean(epoch_representation_real_kl_div_list),
            global_step=global_step,
        )
        logger.log(
            "WorldModel/total_loss",
            np.mean(epoch_total_loss_list),
            global_step=global_step,
        )

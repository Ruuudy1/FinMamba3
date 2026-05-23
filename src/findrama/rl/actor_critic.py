# region imports
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from pytorch_warmup import LinearWarmup
from findrama.models.laprop import LaProp
from findrama.models.losses import SymLogTwoHotLoss
from findrama.models.activations import ACTIVATION_BY_NAME
from findrama.training_utils import EMAScalar
from findrama.weight_init import layer_init
from findrama.rl.returns import calc_lambda_return, percentile
from findrama.rl.normalization import VecNormalize
# endregion
RMSNorm = nn.RMSNorm


class ActorCriticAgent(nn.Module):
    def __init__(self, conf, action_dim, device) -> None:
        super().__init__()
        feat_dim=conf.Models.WorldModel.CategoricalDim*conf.Models.WorldModel.ClassDim+conf.Models.WorldModel.HiddenStateDim
        num_layers=conf.Models.Agent.AC.NumLayers
        actor_hidden_dim=conf.Models.Agent.AC.Actor.HiddenUnits
        critic_hidden_dim=conf.Models.Agent.AC.Critic.HiddenUnits
        self.gamma = conf.Models.Agent.AC.Gamma
        self.lambd = conf.Models.Agent.AC.Lambda
        self.entropy_coef = conf.Models.Agent.AC.EntropyCoef
        self.use_amp = conf.BasicSettings.Use_amp
        self.max_grad_norm=conf.Models.Agent.AC.Max_grad_norm
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.action_dim = action_dim
        self.unimix_ratio = conf.Models.Agent.Unimix_ratio
        self.device = device
        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        act = ACTIVATION_BY_NAME[conf.Models.Agent.AC.Act]
        actor = [
            VecNormalize(feat_dim, device=device),
            layer_init(nn.Linear(feat_dim, actor_hidden_dim, bias=True)),
            RMSNorm(actor_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                layer_init(nn.Linear(actor_hidden_dim, actor_hidden_dim, bias=True)),
                RMSNorm(actor_hidden_dim),
                act()
            ])
        self.actor = nn.Sequential(
            *actor,
            layer_init(nn.Linear(actor_hidden_dim, action_dim), std=0.001)
        ).to(device)
        critic = [
            layer_init(nn.Linear(feat_dim, critic_hidden_dim, bias=True)),
            RMSNorm(critic_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                layer_init(nn.Linear(critic_hidden_dim, critic_hidden_dim, bias=True)),
                RMSNorm(critic_hidden_dim),
                act()
            ])
        self.critic = nn.Sequential(
            *critic,
            layer_init(nn.Linear(critic_hidden_dim, 255), std=0.001)
        ).to(device)
        self.slow_critic = copy.deepcopy(self.critic)
        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)
        if conf.Models.Agent.AC.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=conf.Models.Agent.AC.Laprop.LearningRate, eps=conf.Models.Agent.AC.Laprop.Epsilon)
        elif conf.Models.Agent.AC.Optimiser == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=conf.Models.Agent.AC.Adam.LearningRate,
                eps=conf.Models.Agent.AC.Adam.Epsilon
            )
        else:
            raise ValueError(f"Unknown optimiser: {conf.Models.Agent.AC.Optimiser}")
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)  # No-op schedule; required to drive the warmup scheduler.
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=conf.Models.Agent.AC.Warmup_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))
    def policy(self, x):
        logits = self.actor(x)
        logits = self.unimix(logits)
        return logits
    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value
    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value
    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value
    def unimix(self, logits):
        # Mix action logits with uniform noise for exploration.
        if self.unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.action_dim
            mixed_probs = self.unimix_ratio * uniform + (1-self.unimix_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits
    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.policy(latent)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action, logits
    def sample_as_env_action(self, latent, greedy=False):
        action, _ = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()
    def update(self, latent, action, old_logits, context_latent, context_reward, context_termination, reward, termination, logger, global_step):
        """Update policy and value model."""
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            # Decode value estimates and compute lambda returns.
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)
            # Update value function with slow-critic regularization.
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())
            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1, device=reward.device), S)  # Clip to 1 per the paper's max(1, S) formulation.
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss
        # Apply gradient descent step.
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # Unscale before grad clipping.
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.warmup_scheduler.dampen()
        self.update_slow_critic()
        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item(), global_step=global_step)
            logger.log('ActorCritic/value_loss', value_loss.item(), global_step=global_step)
            logger.log('ActorCritic/entropy_loss', -entropy_loss.item(), global_step=global_step)
            logger.log('ActorCritic/S', S.item(), global_step=global_step)
            logger.log('ActorCritic/norm_ratio', norm_ratio.item(), global_step=global_step)
            logger.log('ActorCritic/total_loss', loss.item(), global_step=global_step)

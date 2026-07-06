# region imports
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from pytorch_warmup import LinearWarmup
from finmamba3.models.laprop import LaProp
from finmamba3.models.losses import SymLogTwoHotLoss
from finmamba3.models.activations import ACTIVATION_BY_NAME
from finmamba3.training_utils import EMAScalar
from finmamba3.weight_init import layer_init
from finmamba3.rl.returns import calc_lambda_return, percentile
from finmamba3.rl.normalization import VecNormalize
# endregion
RMSNorm = nn.RMSNorm


class PPOAgent(nn.Module):

    def __init__(self, conf, action_dim, device):
        super().__init__()
        feat_dim=conf.Models.WorldModel.CategoricalDim*conf.Models.WorldModel.ClassDim+conf.Models.WorldModel.HiddenStateDim
        num_layers=conf.Models.Agent.PPO.NumLayers
        actor_hidden_dim=conf.Models.Agent.PPO.Actor.HiddenUnits
        critic_hidden_dim=conf.Models.Agent.PPO.Critic.HiddenUnits
        self.gamma = conf.Models.Agent.PPO.Gamma
        self.lambd = conf.Models.Agent.PPO.Lambda
        self.entropy_coef = conf.Models.Agent.PPO.EntropyCoef
        self.eps_clip=conf.Models.Agent.PPO.EpsilonClip
        self.K_epochs=conf.Models.Agent.PPO.K_epochs
        self.minibatch_size=conf.Models.Agent.PPO.Minibatch
        self.c1=conf.Models.Agent.PPO.CriticCoef
        self.c2=conf.Models.Agent.PPO.EntropyCoef
        self.kl_threshold=conf.Models.Agent.PPO.KL_threshold
        self.max_grad_norm=conf.Models.Agent.PPO.Max_grad_norm
        self.use_amp = conf.BasicSettings.Use_amp
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.action_dim = action_dim
        self.unimix_ratio = conf.Models.Agent.Unimix_ratio
        self.device = device
        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        act = ACTIVATION_BY_NAME[conf.Models.Agent.PPO.Act]
        actor = [
            VecNormalize(feat_dim, device=device),
            layer_init(nn.Linear(feat_dim, actor_hidden_dim, bias=True)),
            RMSNorm(actor_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            actor.extend([layer_init(nn.Linear(actor_hidden_dim, actor_hidden_dim, bias=True)), RMSNorm(actor_hidden_dim), act()])
        self.actor = nn.Sequential(*actor, layer_init(nn.Linear(actor_hidden_dim, action_dim), std=0.001)).to(device)
        critic = [layer_init(nn.Linear(feat_dim, critic_hidden_dim, bias=True)), RMSNorm(critic_hidden_dim), act()]
        for i in range(num_layers - 1):
            critic.extend([layer_init(nn.Linear(critic_hidden_dim, critic_hidden_dim, bias=True)), RMSNorm(critic_hidden_dim), act()])
        self.critic = nn.Sequential(*critic, layer_init(nn.Linear(critic_hidden_dim, 255), std=0.001)).to(device)
        self.slow_critic = copy.deepcopy(self.critic)
        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)
        if conf.Models.Agent.PPO.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=conf.Models.Agent.PPO.Laprop.LearningRate, eps=conf.Models.Agent.PPO.Laprop.Epsilon)
        elif conf.Models.Agent.PPO.Optimiser == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=conf.Models.Agent.PPO.Adam.LearningRate,
                eps=conf.Models.Agent.PPO.Adam.Epsilon
            )
        else:
            raise ValueError(f"Unknown optimiser: {conf.Models.Agent.PPO.Optimiser}")
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)  # No-op schedule; required to drive the warmup scheduler.
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=conf.Models.Agent.PPO.Warmup_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def get_logp_val_entr(self, latent, action, longer_value=True):
        if longer_value:
            logits = self.actor(latent[:, :-1])
        else:
            logits = self.actor(latent)
        value = self.critic(latent)
        dist = distributions.Categorical(logits=logits)
        logp_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return logp_prob, value, entropy

    def unimix(self, logits):
        # Mix action logits with uniform noise for exploration.
        if self.unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.action_dim
            mixed_probs = self.unimix_ratio * uniform + (1-self.unimix_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits

    def sample_as_env_action(self, latent, greedy=False):
        action, _ = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()

    def compute_loss(self, latent, action, logp_old, advs, rtgs, slow_return):
        logp, raw_values, entropy = self.get_logp_val_entr(latent, action, longer_value=False)
        ratio = torch.exp(logp - logp_old)
        # KL approximation per joschu.net/blog/kl-approx.html.
        kl_apx = ((ratio - 1) - (logp - logp_old)).mean()
        clip_advs = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advs
        # Negate the loss because Adam minimizes; policy gradient requires ascent.
        actor_loss = -(torch.min(ratio*advs.detach(), clip_advs.detach())).mean()
        slow_critic_loss = self.symlog_twohot_loss(raw_values, slow_return.detach())
        critic_loss = self.symlog_twohot_loss(raw_values, rtgs.detach())
        entropy_loss = entropy.mean()
        return actor_loss, critic_loss, slow_critic_loss, entropy_loss, kl_apx

    def calc_gae_and_reward_to_go(self, rewards, values, termination):
        # Invert termination to have 0 if the episode ended and 1 otherwise.
        inv_termination = (termination * -1) + 1
        batch_size, batch_length = rewards.shape[:2]
        deltas = torch.zeros((batch_size, batch_length), dtype=rewards.dtype, device=rewards.device)
        advantages = torch.zeros((batch_size, batch_length+1), dtype=rewards.dtype, device=rewards.device)
        # Calculate per-step TD deltas.
        for t in range(batch_length):
            next_value = values[:, t+1]
            deltas[:, t] = rewards[:, t] + self.gamma * inv_termination[:, t] * next_value - values[:, t]
        # Accumulate advantages using GAE.
        for t in reversed(range(batch_length)):
            next_advantage = advantages[:, t+1] if t < batch_length - 1 else 0
            advantages[:, t] = deltas[:, t] + self.gamma * self.lamb * inv_termination[:, t] * next_advantage
        # Add value estimates to advantages to recover the lambda returns.
        returns = advantages[:, :-1] + values[:, :-1]
        return advantages[:, :-1], returns

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def update(self, latent, action, old_logits, context_latent, context_reward, context_termination, reward, termination, logger, global_step):
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            feat_dim = latent.shape[-1]
            dist = distributions.Categorical(logits=old_logits)
            old_logp = dist.log_prob(action)
            flatten_latent = latent[:, :-1].reshape(-1, feat_dim)
            flatten_action = action.view(-1)
            flatten_old_logp = old_logp.view(-1).detach()
            batch_size = flatten_latent.shape[0]
            entropy_loss_list = []
            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            kl_approx_list = []
            for _ in range(self.K_epochs):
                # Recompute value after each PPO update per Andrychowicz et al. 2020, section 3.5.
                value = self.value(latent)
                slow_value = self.slow_value(latent)
                lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)
                slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
                lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
                upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
                S = upper_bound-lower_bound
                norm_ratio = torch.max(torch.ones(1, device=reward.device), S)  # Clip to 1 per the paper's max(1, S) formulation.
                norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
                flatten_advantages = norm_advantage.view(-1)
                flatten_returns = lambda_return.reshape(-1)
                flatten_slow_return = slow_lambda_return.reshape(-1)
                # Shuffle minibatch indices each PPO epoch.
                inds = np.arange(batch_size)
                np.random.shuffle(inds)
                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_inds = inds[start:end]
                    actor_loss, critic_loss, slow_critic_loss, entropy_loss, kl_apx = self.compute_loss(
                        flatten_latent[minibatch_inds],
                        flatten_action[minibatch_inds],
                        flatten_old_logp[minibatch_inds],
                        flatten_advantages[minibatch_inds],
                        flatten_returns[minibatch_inds],
                        flatten_slow_return[minibatch_inds]
                    )
                    total_loss = actor_loss + self.c1 * critic_loss + slow_critic_loss - self.c2 * entropy_loss
                    entropy_loss_list.append(-entropy_loss.item())
                    actor_loss_list.append(actor_loss.item())
                    critic_loss_list.append(critic_loss.item())
                    total_loss_list.append(total_loss.item())
                    kl_approx_list.append(kl_apx.item())
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)  # Unscale before grad clipping.
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.update_slow_critic()
            self.lr_scheduler.step()
            self.warmup_scheduler.dampen()
        if logger is not None:
            logger.log('ActorCritic/policy_loss', np.mean(actor_loss_list), global_step=global_step)
            logger.log('ActorCritic/value_loss', np.mean(critic_loss_list), global_step=global_step)
            logger.log('ActorCritic/entropy_loss', np.mean(entropy_loss_list), global_step=global_step)
            logger.log('ActorCritic/KL_approx', np.mean(kl_approx_list), global_step=global_step)
            logger.log('ActorCritic/S', S.item(), global_step=global_step)
            logger.log('ActorCritic/norm_ratio', norm_ratio.item(), global_step=global_step)
            logger.log('ActorCritic/total_loss', np.mean(total_loss_list), global_step=global_step)

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.actor(latent)
            logits = self.unimix(logits)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action, logits

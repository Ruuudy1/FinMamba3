import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange, reduce
from sub_models.laprop import LaProp
from pytorch_warmup import LinearWarmup

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.transformer_model import StochasticTransformerKVCache
from sub_models.fin_mamba import FinMambaSequenceModel
from sub_models.lob_auxiliary import (
    DirectionHead,
    EpisodicMemory,
    EpisodicMemoryFuser,
    HawkesIntensityHead,
    RegimeConditioner,
    RegimeHead,
    SettlementHead,
)
from sub_models.lob_encoder import (
    LOBDecoder,
    LOBEncoder,
    LOBReconstructionLoss,
    StudentTLOBDecoder,
    StudentTReconstructionLoss,
)
import agents
from tools import weight_init

RMSNorm = nn.RMSNorm


class DistHead(nn.Module):
    """Distribution head for posterior and prior categorical logits."""
    def __init__(self, image_feat_dim, hidden_state_dim, categorical_dim, class_dim, unimix_ratio=0.01, dtype=None, device=None) -> None:
        super().__init__()
        self.stoch_dim = categorical_dim
        self.post_head = nn.Linear(image_feat_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.prior_head = nn.Linear(hidden_state_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.unimix_ratio = unimix_ratio
        self.dtype=dtype
        self.device=device

    def unimix(self, logits, mixing_ratio=0.01):
        # Mix logits with uniform noise for uniform-prior regularization.
        if mixing_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            mixed_probs = mixing_ratio * torch.ones_like(probs, dtype=self.dtype, device=self.device) / self.stoch_dim + (1-mixing_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits



    

class RewardHead(nn.Module):
    def __init__(self, num_classes, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Build backbone layers dynamically.
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, num_classes, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward




class TerminationHead(nn.Module):
    def __init__(self, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Build backbone layers dynamically.
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # Remove the trailing singleton dimension.
        return termination

class CategoricalKLDivLossWithFreeBits(nn.Module):
    """KL with a per-step free-bits floor. DreamerV3 uses 1.0 nat (Hafner et al. 2023);
    the floor prevents degenerate posterior=prior solutions where the latent carries no
    information about the input."""
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, action_dim, config, device):
        super().__init__()
        self.hidden_state_dim = config.Models.WorldModel.HiddenStateDim
        self.final_feature_width = config.Models.WorldModel.Transformer.FinalFeatureWidth
        self.categorical_dim = config.Models.WorldModel.CategoricalDim
        self.class_dim = config.Models.WorldModel.ClassDim
        self.stoch_flattened_dim = self.categorical_dim*self.class_dim
        self.use_amp = config.BasicSettings.Use_amp
        self.use_cg = config.BasicSettings.Use_cg
        self.tensor_dtype = torch.bfloat16 if self.use_amp and not self.use_cg else config.Models.WorldModel.dtype
        self.save_every_steps = config.JointTrainAgent.SaveEverySteps
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.device = device  # Stored for device placement in buffer init.
        self.model = config.Models.WorldModel.Backbone
        self.encoder_type = getattr(config.Models.WorldModel.Encoder, 'Type', 'lob')
        if self.encoder_type != 'lob':
            raise ValueError("FinDrama only supports Models.WorldModel.Encoder.Type='lob'")
        regime_cfg = getattr(config.Models.WorldModel, 'Regime', None)
        self.use_regime = bool(regime_cfg is not None and getattr(regime_cfg, 'Enabled', False))
        self.max_grad_norm = config.Models.WorldModel.Max_grad_norm
        max_seq_length = max(config.JointTrainAgent.BatchLength,
                             config.JointTrainAgent.ImagineContextLength + config.JointTrainAgent.ImagineBatchLength,
                             config.JointTrainAgent.RealityContextLength)
        enc_cfg = config.Models.WorldModel.Encoder
        self.encoder = LOBEncoder(
            k_levels=enc_cfg.K,
            f_level=enc_cfg.FeatureDimLevel,
            f_tick=enc_cfg.FeatureDimTick,
            d_model=enc_cfg.DModel,
            num_layers=enc_cfg.NumLayers,
            num_heads=enc_cfg.NumHeads,
            dim_feedforward=getattr(enc_cfg, 'DimFeedforward', enc_cfg.DModel * 2),
            dropout=config.Models.WorldModel.Dropout,
            output_flatten_dim=enc_cfg.OutputFlattenDim,
            aggregate_only=getattr(enc_cfg, 'AggregateOnly', False),
            gradient_checkpointing=getattr(enc_cfg, 'GradientCheckpointing', False),
            dtype=config.Models.WorldModel.dtype, device=device,
        )
        if self.model == 'Transformer':
            self.sequence_model = StochasticTransformerKVCache(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=self.hidden_state_dim,
                num_layers=config.Models.WorldModel.Transformer.NumLayers,
                num_heads=config.Models.WorldModel.Transformer.NumHeads,
                max_length=max_seq_length,
                dropout=config.Models.WorldModel.Dropout
            )
        elif self.model == 'Mamba':
            self.sequence_model = FinMambaSequenceModel(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                d_model=self.hidden_state_dim,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                block_type='Mamba',
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                },
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
        elif self.model == 'Mamba2':
            self.sequence_model = FinMambaSequenceModel(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                d_model=self.hidden_state_dim,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                block_type='Mamba2',
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                },
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
        elif self.model == 'Mamba3':
            mamba3_cfg = config.Models.WorldModel.Mamba3
            if self.hidden_state_dim % mamba3_cfg.headdim != 0:
                raise ValueError(
                    "Models.WorldModel.HiddenStateDim must be divisible by "
                    "Models.WorldModel.Mamba3.headdim"
                )
            if not getattr(mamba3_cfg, 'Enabled', True):
                raise ValueError("Backbone is Mamba3 but Models.WorldModel.Mamba3.Enabled is false")
            self.sequence_model = FinMambaSequenceModel(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                d_model=self.hidden_state_dim,
                n_layer=getattr(mamba3_cfg, 'n_layer', config.Models.WorldModel.Mamba.n_layer),
                block_type='Mamba3',
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': mamba3_cfg.d_state,
                    'headdim': mamba3_cfg.headdim,
                    'is_mimo': mamba3_cfg.is_mimo,
                    'mimo_rank': mamba3_cfg.mimo_rank,
                    'chunk_size': mamba3_cfg.chunk_size,
                    'is_outproj_norm': mamba3_cfg.is_outproj_norm,
                    'rope_fraction': mamba3_cfg.rope_fraction,
                },
                use_action_input=bool(getattr(config.Models.WorldModel, 'UseActionInput', True)),
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
        else:
            raise ValueError(f"Unknown dynamics model: {self.model}")               
        

        self.dist_head = DistHead(
            image_feat_dim=self.encoder.output_flatten_dim,
            hidden_state_dim=self.hidden_state_dim,
            categorical_dim=self.categorical_dim,
            class_dim=self.class_dim,
            unimix_ratio=config.Models.WorldModel.Unimix_ratio,
            dtype=config.Models.WorldModel.dtype, device=device
        )      
        decoder_kind = getattr(config.Models.WorldModel.Decoder, 'Kind', 'mse')
        self.decoder_kind = decoder_kind
        if decoder_kind == 'mse':
            self.obs_decoder = LOBDecoder(
                stoch_dim=self.stoch_flattened_dim,
                hidden_dim=getattr(config.Models.WorldModel.Decoder, 'HiddenDim', 512),
                num_layers=getattr(config.Models.WorldModel.Decoder, 'NumLayers', 3),
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                dtype=config.Models.WorldModel.dtype, device=device,
            )
        elif decoder_kind == 'studentt':
            self.obs_decoder = StudentTLOBDecoder(
                stoch_dim=self.stoch_flattened_dim,
                hidden_dim=getattr(config.Models.WorldModel.Decoder, 'HiddenDim', 512),
                num_layers=getattr(config.Models.WorldModel.Decoder, 'NumLayers', 3),
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                nu_init=float(getattr(config.Models.WorldModel.Decoder, 'NuInit', 5.0)),
                learnable_nu=bool(getattr(config.Models.WorldModel.Decoder, 'LearnableNu', True)),
                dtype=config.Models.WorldModel.dtype, device=device,
            )
        else:
            raise ValueError(
                f"Models.WorldModel.Decoder.Kind must be 'mse' or 'studentt'; got {decoder_kind!r}"
            )

        self.use_reward_head = bool(getattr(config.Models.WorldModel.Reward, 'Enabled', True))
        self.use_termination_head = bool(getattr(config.Models.WorldModel.Termination, 'Enabled', True))
        if self.use_reward_head:
            self.reward_decoder = RewardHead(
                num_classes=255,
                inp_dim=self.hidden_state_dim,
                hidden_units=config.Models.WorldModel.Reward.HiddenUnits,
                act=config.Models.WorldModel.Act,
                layer_num=config.Models.WorldModel.Reward.LayerNum,
                dtype=config.Models.WorldModel.dtype, device=device
            )
            self.reward_decoder.apply(weight_init)
        else:
            self.reward_decoder = None
        if self.use_termination_head:
            self.termination_decoder = TerminationHead(
                inp_dim=self.hidden_state_dim,
                hidden_units=config.Models.WorldModel.Termination.HiddenUnits,
                act=config.Models.WorldModel.Act,
                layer_num=config.Models.WorldModel.Termination.LayerNum,
                dtype=config.Models.WorldModel.dtype, device=device
            )
            self.termination_decoder.apply(weight_init)
        else:
            self.termination_decoder = None
        direction_cfg = getattr(config.Models.WorldModel, 'Direction', None)
        self.use_direction_head = bool(direction_cfg is not None and getattr(direction_cfg, 'Enabled', False))
        if self.use_direction_head:
            self.direction_head = DirectionHead(
                hidden_dim=self.hidden_state_dim,
                num_classes=int(getattr(direction_cfg, 'NumClasses', 3)),
                dropout=float(getattr(direction_cfg, 'Dropout', 0.0)),
                dtype=config.Models.WorldModel.dtype, device=device,
            )
            self.direction_loss_weight = float(getattr(direction_cfg, 'LossWeight', 0.5))
            self.direction_threshold = float(getattr(direction_cfg, 'Threshold', 1.0e-2))
            # Index of normalized midprice in the flat feature vector.
            self.midprice_index = int(enc_cfg.K) * int(enc_cfg.FeatureDimLevel)
        else:
            self.direction_head = None
            self.direction_loss_weight = 0.0
            self.direction_threshold = 0.0
            self.midprice_index = -1
        if self.use_regime:
            self.regime_head = RegimeHead(
                hidden_dim=self.hidden_state_dim,
                num_regimes=getattr(regime_cfg, 'NumRegimes', 8),
                embed_dim=getattr(regime_cfg, 'EmbedDim', 32),
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.regime_conditioner = RegimeConditioner(
                hidden_dim=self.hidden_state_dim,
                regime_dim=getattr(regime_cfg, 'EmbedDim', 32),
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
        else:
            self.regime_head = None
            self.regime_conditioner = None

        em_cfg = getattr(config.Models.WorldModel, 'EpisodicMemory', None)
        self.use_episodic_memory = bool(em_cfg is not None and getattr(em_cfg, 'Enabled', False))
        if self.use_episodic_memory:
            self.episodic_memory = EpisodicMemory(
                key_dim=self.hidden_state_dim,
                value_dim=self.hidden_state_dim,
                capacity=int(getattr(em_cfg, 'Capacity', 50_000)),
                novelty_threshold=float(getattr(em_cfg, 'NoveltyThreshold', 0.0)),
            )
            self.memory_use_novelty = bool(getattr(em_cfg, 'UseNovelty', False))
            self.episodic_memory_fuser = EpisodicMemoryFuser(
                hidden_dim=self.hidden_state_dim,
                memory_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.memory_topk = int(getattr(em_cfg, 'TopK', 4))
            self.memory_min_fill = int(getattr(em_cfg, 'MinFillBeforeRetrieve', 1024))
            self.memory_retrieve_every = int(getattr(em_cfg, 'RetrieveEvery', 4))
            self._memory_steps_since_retrieve = self.memory_retrieve_every
        else:
            self.episodic_memory = None
            self.episodic_memory_fuser = None
            self.memory_topk = 0
            self.memory_min_fill = 0
            self.memory_retrieve_every = 0
            self._memory_steps_since_retrieve = 0
            self.memory_use_novelty = False

        hawkes_cfg = getattr(config.Models.WorldModel, 'Hawkes', None)
        self.use_hawkes_head = bool(hawkes_cfg is not None and getattr(hawkes_cfg, 'Enabled', False))
        if self.use_hawkes_head:
            self.hawkes_head = HawkesIntensityHead(
                hidden_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.hawkes_loss_weight = float(getattr(hawkes_cfg, 'LossWeight', 0.1))
        else:
            self.hawkes_head = None
            self.hawkes_loss_weight = 0.0

        settle_cfg = getattr(config.Models.WorldModel, 'Settlement', None)
        self.use_settlement_head = bool(
            settle_cfg is not None and getattr(settle_cfg, 'Enabled', False)
        )
        if self.use_settlement_head:
            self.settlement_head = SettlementHead(
                hidden_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.settlement_loss_weight = float(getattr(settle_cfg, 'LossWeight', 0.25))
        else:
            self.settlement_head = None
            self.settlement_loss_weight = 0.0

        # When DirectionThresholds is a list, the direction loss is computed at each threshold and averaged.
        # This replaces the single-threshold (1%) target with a curve over thresholds.
        self.direction_thresholds = getattr(config.Models.WorldModel, 'DirectionThresholds', None)
        if isinstance(self.direction_thresholds, (list, tuple)):
            self.direction_thresholds = [float(t) for t in self.direction_thresholds]
        else:
            self.direction_thresholds = None

        # Decoder size_weight up-weights size and depth feature MSE relative to price features.
        # The 2.0 default biases the decoder toward volume signal but can crowd out predictive capacity for mid and spread.
        size_weight = float(getattr(config.Models.WorldModel.Decoder, 'SizeWeight', 2.0))
        if self.decoder_kind == 'mse':
            self.reconstruction_loss_func = LOBReconstructionLoss(
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                size_weight=size_weight,
            )
        else:
            self.reconstruction_loss_func = StudentTReconstructionLoss(
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
            )
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        # Free bits floors the categorical KL so the dynamics term cannot drive the prior to copy the posterior trivially.
        # Larger values starve the dynamics signal; smaller values risk prior collapse.
        free_bits = float(getattr(config.Models.WorldModel, 'FreeBits', 1.0))
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=free_bits)
        # Representation loss weight controls how strongly the posterior is pulled toward the prior.
        # DreamerV3 uses 0.5; the original Drama default is 0.1.
        self.representation_loss_weight = float(getattr(config.Models.WorldModel, 'RepresentationLossWeight', 0.1))
        if config.Models.WorldModel.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=config.Models.WorldModel.Laprop.LearningRate, eps=config.Models.WorldModel.Laprop.Epsilon, weight_decay=config.Models.WorldModel.Weight_decay)
        elif config.Models.WorldModel.Optimiser == 'Adam':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.Models.WorldModel.Adam.LearningRate, weight_decay=config.Models.WorldModel.Weight_decay)
        else:
            raise ValueError(f"Unknown optimiser: {config.Models.WorldModel.Optimiser}")
        lr_schedule = getattr(config.Models.WorldModel, 'LRSchedule', 'constant')
        if lr_schedule == 'cosine':
            # Cosine + warmup is the standard recipe for Mamba pretraining.
            t_max = max(int(config.JointTrainAgent.SampleMaxSteps), 1)
            eta_min_ratio = float(getattr(config.Models.WorldModel, 'LRMinRatio', 0.1))
            base_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=base_lr * eta_min_ratio
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=config.Models.WorldModel.Warmup_steps)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and config.Models.WorldModel.dtype is not torch.bfloat16,
        )
        # Watch for selective_scan and Mamba3 instability under bf16 autocast for the first NaNGuardSteps updates.
        # After that we trust the run.
        self.nan_guard_steps = int(getattr(config.Models.WorldModel, 'NaNGuardSteps', 50))
        self._nan_skip_count = 0

    def condition_dist_feat(self, dist_feat):
        if not self.use_regime:
            return dist_feat
        _, regime_emb = self.regime_head(dist_feat)
        return self.regime_conditioner(dist_feat, regime_emb)

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample
    def calc_last_dist_feat(self, latent, action, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(latent, action, inference_params)
            last_dist_feat = dist_feat[:, -1:]
            conditioned_last_dist_feat = self.condition_dist_feat(last_dist_feat)
            prior_logits = self.dist_head.forward_prior(conditioned_last_dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, conditioned_last_dist_feat
    def calc_last_post_feat(self, latent, action, current_obs, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(current_obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)            
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(latent, action, inference_params)
            last_dist_feat = dist_feat[:, -1:]
            last_dist_feat = self.condition_dist_feat(last_dist_feat)
            shifted_feat = last_dist_feat
            x = torch.cat((shifted_feat, flattened_sample), -1)
            post_feat = self._obs_out_layers(x)
            post_stat = self._obs_stat_layer(post_feat)
            post_logits = post_stat.reshape(list(post_stat.shape[:-1]) + [self.categorical_dim, self.categorical_dim])
            post_sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            post_flattened_sample = self.flatten_sample(post_sample)            

        return post_flattened_sample, post_feat    
    # Called only when using the Transformer backbone (requires KV cache).
    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.sequence_model.forward_with_kv_cache(last_flattened_sample, action)
            conditioned_dist_feat = self.condition_dist_feat(dist_feat)
            prior_logits = self.dist_head.forward_prior(conditioned_dist_feat)

            # Decode prior sample into observation.
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.obs_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            if self.use_reward_head:
                reward_hat = self.reward_decoder(conditioned_dist_feat)
                reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            else:
                reward_hat = torch.zeros(conditioned_dist_feat.shape[:2], device=conditioned_dist_feat.device, dtype=conditioned_dist_feat.dtype)
            if self.use_termination_head:
                termination_hat = self.termination_decoder(conditioned_dist_feat)
                termination_hat = termination_hat > 0
            else:
                termination_hat = torch.zeros(conditioned_dist_feat.shape[:2], device=conditioned_dist_feat.device, dtype=torch.bool)

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, conditioned_dist_feat
    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample
    
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        """Pre-allocate imagination buffers to avoid per-step allocation overhead."""
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.hidden_state_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.sample_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.dist_feat_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)

    def _imagine_data_full_prefix(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                                  imagine_batch_size, imagine_batch_length, log_video, logger, global_step):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        context_latent = self.encode_obs(sample_obs)

        generated_samples = []
        generated_actions = []
        old_logits_list = []

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            context_dist_feat = self.sequence_model(context_latent, sample_action)
            conditioned_context_dist_feat = self.condition_dist_feat(context_dist_feat)
            context_prior_logits = self.dist_head.forward_prior(conditioned_context_dist_feat)
            context_prior_sample = self.stright_throught_gradient(context_prior_logits)
            context_flattened_sample = self.flatten_sample(context_prior_sample)

            current_sample = context_flattened_sample[:, -1:]
            current_dist_feat = conditioned_context_dist_feat[:, -1:]
            generated_samples.append(current_sample)
            self.sample_buffer[:, 0:1] = current_sample
            self.dist_feat_buffer[:, 0:1] = current_dist_feat

            for i in range(imagine_batch_length):
                action, logits = agent.sample(torch.cat([current_sample, current_dist_feat], dim=-1))
                action_for_model = action.to(dtype=sample_action.dtype)
                generated_actions.append(action_for_model)
                old_logits_list.append(logits)
                self.action_buffer[:, i:i+1] = action_for_model

                prefix_samples = torch.cat([context_latent] + generated_samples, dim=1)
                prefix_actions = torch.cat([sample_action] + generated_actions, dim=1)
                dist_feat = self.sequence_model(prefix_samples, prefix_actions)[:, -1:]
                current_dist_feat = self.condition_dist_feat(dist_feat)
                self.dist_feat_buffer[:, i+1:i+2] = current_dist_feat

                prior_logits = self.dist_head.forward_prior(current_dist_feat)
                prior_sample = self.stright_throught_gradient(prior_logits)
                current_sample = self.flatten_sample(prior_sample)
                generated_samples.append(current_sample)
                self.sample_buffer[:, i+1:i+2] = current_sample

            old_logits_tensor = torch.cat(old_logits_list, dim=1) if old_logits_list else None
            if self.use_reward_head:
                reward_hat_tensor = self.reward_decoder(self.dist_feat_buffer[:, :-1])
                self.reward_hat_buffer = self.symlog_twohot_loss_func.decode(reward_hat_tensor)
            else:
                self.reward_hat_buffer.zero_()
            if self.use_termination_head:
                termination_hat_tensor = self.termination_decoder(self.dist_feat_buffer[:, :-1])
                self.termination_hat_buffer = termination_hat_tensor > 0
            else:
                self.termination_hat_buffer = torch.zeros_like(self.termination_hat_buffer, dtype=torch.bool)

        context_out = torch.cat([context_flattened_sample, conditioned_context_dist_feat], dim=-1)
        imagined_out = torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1)
        return imagined_out, self.action_buffer, old_logits_tensor, context_out, self.reward_hat_buffer, self.termination_hat_buffer

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):
        if self.model != 'Transformer':
            return self._imagine_data_full_prefix(
                agent, sample_obs, sample_action,
                imagine_batch_size, imagine_batch_length, log_video, logger, global_step
            )

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        self.sequence_model.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)

        # Encode context observations.
        context_latent = self.encode_obs(sample_obs)

        for i in range(sample_obs.shape[1]):
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.sample_buffer[:, 0:1] = last_latent
        self.dist_feat_buffer[:, 0:1] = last_dist_feat

        # Autoregressively imagine future latents.
        for i in range(imagine_batch_length):
            action, _ = agent.sample(torch.cat([self.sample_buffer[:, i:i+1], self.dist_feat_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.sample_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.sample_buffer[:, i+1:i+2] = last_latent
            self.dist_feat_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
        return torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1), self.action_buffer, None, None, self.reward_hat_buffer, self.termination_hat_buffer

    def imagine_data2(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):
        return self._imagine_data_full_prefix(
            agent, sample_obs, sample_action,
            imagine_batch_size, imagine_batch_length, log_video, logger, global_step
        )


    def update(self, obs, action, reward, termination, global_step, epoch_step,
               logger=None, accum_steps: int = 1, is_last_accum: bool = True,
               event_counts: torch.Tensor | None = None,
               outcome: torch.Tensor | None = None,
               time_to_expiry_frac: torch.Tensor | None = None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # Encode observations into posterior samples.
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
            # Reconstruct observations from samples.
            # The decoder returns the flat feature vector under MSE, or (mean, log_scale) under Student-t.
            decoder_out = self.obs_decoder(flattened_sample)
            if self.decoder_kind == 'studentt':
                obs_hat_mean, obs_hat_log_scale = decoder_out
                obs_hat = obs_hat_mean
            else:
                obs_hat = decoder_out
            # Compute sequence-model hidden states.
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
                dist_feat = self.sequence_model(flattened_sample, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(flattened_sample, action)
            conditioned_dist_feat = self.condition_dist_feat(dist_feat)
            # Episodic-memory fusion retrieves top-K nearest past hidden states using the most recent step as the query.
            # Retrieved values are broadcast across the sequence and fused via a gated residual.
            # Retrieval is CPU-bound, so it is throttled by RetrieveEvery and skipped until the memory holds MinFillBeforeRetrieve entries.
            if self.use_episodic_memory and len(self.episodic_memory) >= self.memory_min_fill:
                self._memory_steps_since_retrieve += 1
                if self._memory_steps_since_retrieve >= self.memory_retrieve_every:
                    self._memory_steps_since_retrieve = 0
                    query = conditioned_dist_feat[:, -1].detach()
                    mem_batch = self.episodic_memory.retrieve(query, k=self.memory_topk)
                    if mem_batch is not None:
                        mem_value = mem_batch.values.to(dtype=conditioned_dist_feat.dtype)
                        mem_value = mem_value.unsqueeze(1).expand(-1, conditioned_dist_feat.shape[1], -1)
                        conditioned_dist_feat = self.episodic_memory_fuser(conditioned_dist_feat, mem_value)
            prior_logits = self.dist_head.forward_prior(conditioned_dist_feat)
            # Compute observation loss.
            # Student-t uses negative log-likelihood over (mean, log_scale), and MSE uses the existing weighted MSE.
            if self.decoder_kind == 'studentt':
                reconstruction_loss = self.reconstruction_loss_func(
                    self.obs_decoder,
                    obs_hat_mean[:batch_size],
                    obs_hat_log_scale[:batch_size],
                    obs[:batch_size],
                )
            else:
                reconstruction_loss = self.reconstruction_loss_func(obs_hat[:batch_size], obs[:batch_size])
            # Predict reward and termination from the prior hidden state when their heads are enabled.
            if self.use_reward_head:
                reward_hat = self.reward_decoder(conditioned_dist_feat)
                reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            else:
                reward_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            if self.use_termination_head:
                termination_hat = self.termination_decoder(conditioned_dist_feat)
                termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            else:
                termination_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            # Compute dynamics and representation KL losses.
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            # Auxiliary directional supervision predicts the next-tick midprice direction from the prior hidden state.
            # The cross-entropy term forces the latent to encode predictive information rather than only reconstructive information.
            if self.use_direction_head:
                mid_norm = obs[..., self.midprice_index]
                direction_logits = self.direction_head(conditioned_dist_feat[:, :-1])
                if self.direction_thresholds is not None:
                    # Multi-threshold sweep averages cross-entropy across the threshold curve.
                    # This stops the head from being pinned to a single bucket.
                    losses = []
                    for thr in self.direction_thresholds:
                        targets, _ = DirectionHead.make_targets(mid_norm, float(thr))
                        losses.append(
                            F.cross_entropy(
                                direction_logits.reshape(-1, direction_logits.shape[-1]),
                                targets.reshape(-1),
                            )
                        )
                    direction_loss = torch.stack(losses).mean()
                else:
                    direction_targets, _ = DirectionHead.make_targets(mid_norm, self.direction_threshold)
                    direction_loss = F.cross_entropy(
                        direction_logits.reshape(-1, direction_logits.shape[-1]),
                        direction_targets.reshape(-1),
                    )
            else:
                direction_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            # Hawkes intensity auxiliary predicts log-intensity for buy and sell arrivals.
            # The term only contributes when event_counts is provided with shape (B, L, 2) for buy and sell counts per tick or bar.
            if self.use_hawkes_head and event_counts is not None:
                hawkes_logits = self.hawkes_head(conditioned_dist_feat)
                hawkes_loss = HawkesIntensityHead.poisson_nll(
                    hawkes_logits, event_counts.to(dtype=hawkes_logits.dtype)
                )
            else:
                hawkes_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            # Settlement auxiliary predicts the binary contract outcome from the latent.
            # The term only contributes when outcome is provided as a broadcastable scalar per sequence with shape (B,) or (B, L).
            if self.use_settlement_head and outcome is not None:
                settle_logits = self.settlement_head(conditioned_dist_feat).reshape(-1)
                outcome_flat = outcome.reshape(-1).to(dtype=settle_logits.dtype)
                if outcome_flat.shape[0] != settle_logits.shape[0]:
                    if outcome.dim() == 1:
                        outcome_flat = outcome.unsqueeze(1).expand(-1, conditioned_dist_feat.shape[1]).reshape(-1).to(dtype=settle_logits.dtype)
                ttf_flat = None
                if time_to_expiry_frac is not None:
                    ttf_flat = time_to_expiry_frac.reshape(-1).to(dtype=settle_logits.dtype)
                settlement_loss = SettlementHead.bce(settle_logits, outcome_flat, ttf_flat)
            else:
                settlement_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss
                + dynamics_loss + self.representation_loss_weight * representation_loss
                + self.direction_loss_weight * direction_loss
                + self.hawkes_loss_weight * hawkes_loss
                + self.settlement_loss_weight * settlement_loss
            )

        # Catch bf16 selective_scan blowups during early training.
        # Skipping the backward and optimizer step here keeps the rest of the run salvageable instead of propagating NaN parameters everywhere.
        if global_step < self.nan_guard_steps and not torch.isfinite(total_loss):
            self._nan_skip_count += 1
            if self._nan_skip_count >= 5:
                raise RuntimeError(
                    f"WorldModel.update produced non-finite loss for "
                    f"{self._nan_skip_count} consecutive steps under bf16 autocast; "
                    "consider setting BasicSettings.Use_amp=false to debug."
                )
            self.optimizer.zero_grad(set_to_none=True)
            zero = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
            return (zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)
        self._nan_skip_count = 0

        # Apply gradient update.
        self.scaler.scale(total_loss / accum_steps).backward()
        if is_last_accum:
            self.scaler.unscale_(self.optimizer)  # Unscale before grad clipping.
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self.warmup_scheduler.dampen()

        # Stash the last-frame hidden state per batch element into episodic memory.
        # One entry per sequence keeps the CPU buffer manageable while still building a representative regime catalog.
        if self.use_episodic_memory:
            last_hidden = conditioned_dist_feat[:, -1].detach()
            novelty = None
            if getattr(self, 'memory_use_novelty', False):
                # Novelty score is the symmetric KL between posterior and prior at the last step.
                # High KL means the observation surprised the prior, so the entry is worth cataloguing.
                with torch.no_grad():
                    p = post_logits[:, -1].detach()
                    q = prior_logits[:, -1].detach()
                    p_log_softmax = F.log_softmax(p, dim=-1)
                    q_log_softmax = F.log_softmax(q, dim=-1)
                    kl_pq = (p_log_softmax.exp() * (p_log_softmax - q_log_softmax)).sum(dim=(-1, -2))
                    kl_qp = (q_log_softmax.exp() * (q_log_softmax - p_log_softmax)).sum(dim=(-1, -2))
                    novelty = (kl_pq + kl_qp).reshape(-1)
            self.episodic_memory.add(last_hidden, last_hidden, novelty=novelty)

        # Return detached tensors so the caller can stack and sync once per log step.
        # This avoids paying many GPU-CPU syncs per call to update.
        return (
            reconstruction_loss.detach(), reward_loss.detach(), termination_loss.detach(),
            dynamics_loss.detach(), dynamics_real_kl_div.detach(), representation_loss.detach(),
            representation_real_kl_div.detach(), direction_loss.detach(),
            hawkes_loss.detach(), settlement_loss.detach(), total_loss.detach(),
        )

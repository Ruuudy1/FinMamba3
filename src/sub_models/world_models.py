import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch.distributions import OneHotCategorical
from einops import rearrange, reduce
from sub_models.laprop import LaProp
from pytorch_warmup import LinearWarmup
# from nfnets import AGC

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.transformer_model import StochasticTransformerKVCache
from sub_models.lob_auxiliary import RegimeConditioner, RegimeHead
from sub_models.lob_encoder import LOBDecoder, LOBEncoder, LOBReconstructionLoss
from mamba_ssm import MambaWrapperModel, MambaConfig, InferenceParams, update_graph_cache
import agents
from tools import weight_init

    
class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, hidden_state_dim, categorical_dim, class_dim, unimix_ratio=0.01, dtype=None, device=None) -> None:
        super().__init__()
        self.stoch_dim = categorical_dim
        self.post_head = nn.Linear(image_feat_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.prior_head = nn.Linear(hidden_state_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.unimix_ratio = unimix_ratio
        self.dtype=dtype
        self.device=device

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
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

        # Create the backbone with dynamic number of layers
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

        # Create the backbone with dynamic number of layers
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
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination

class CategoricalKLDivLossWithFreeBits(nn.Module):
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
        self.device = device # Maybe it's not needed
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
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                    }
                )                                
            self.sequence_model = MambaWrapperModel(mamba_config)
        elif self.model == 'Mamba2':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state, 
                    'layer': 'Mamba2'}
                )
            self.sequence_model = MambaWrapperModel(mamba_config)                      
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
        self.image_decoder = LOBDecoder(
            stoch_dim=self.stoch_flattened_dim,
            hidden_dim=getattr(config.Models.WorldModel.Decoder, 'HiddenDim', 512),
            num_layers=getattr(config.Models.WorldModel.Decoder, 'NumLayers', 3),
            k_levels=enc_cfg.K,
            f_level=enc_cfg.FeatureDimLevel,
            f_tick=enc_cfg.FeatureDimTick,
            dtype=config.Models.WorldModel.dtype, device=device,
        )
        
        self.reward_decoder = RewardHead(
            num_classes=255,
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Reward.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Reward.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.reward_decoder.apply(weight_init)
        self.termination_decoder = TerminationHead(
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Termination.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Termination.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.termination_decoder.apply(weight_init)
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

        self.reconstruction_loss_func = LOBReconstructionLoss(
            k_levels=enc_cfg.K,
            f_level=enc_cfg.FeatureDimLevel,
            f_tick=enc_cfg.FeatureDimTick,
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        if config.Models.WorldModel.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=config.Models.WorldModel.Laprop.LearningRate, eps=config.Models.WorldModel.Laprop.Epsilon, weight_decay=config.Models.WorldModel.Weight_decay)
        elif config.Models.WorldModel.Optimiser == 'Adam':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.Models.WorldModel.Adam.LearningRate, weight_decay=config.Models.WorldModel.Weight_decay)
        else:
            raise ValueError(f"Unknown optimiser: {config.Models.WorldModel.Optimiser}")
        # self.optimizer = AGC(self.parameters(), self.optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=config.Models.WorldModel.Warmup_steps)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and config.Models.WorldModel.dtype is not torch.bfloat16,
        )

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
    # only called when using Transformer
    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.sequence_model.forward_with_kv_cache(last_flattened_sample, action)
            conditioned_dist_feat = self.condition_dist_feat(dist_feat)
            prior_logits = self.dist_head.forward_prior(conditioned_dist_feat)

            # decoding
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(conditioned_dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(conditioned_dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, conditioned_dist_feat
    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
            # sample = dist.sample()
        elif sample_mode == "mode":
            sample = dist.mode
            # sample = dist.mode()
        elif sample_mode == "probs":
            sample = dist.probs
        return sample
    
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
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
    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        self.sequence_model.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)

        # context
        context_latent = self.encode_obs(sample_obs)

        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.sample_buffer[:, 0:1] = last_latent
        self.dist_feat_buffer[:, 0:1] = last_dist_feat

        # imagine
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
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        batch_size, seqlen_og, embedding_dim = context_latent.shape
        max_length = imagine_batch_length + seqlen_og
        
        if self.use_cg:
            if not hasattr(self.sequence_model, "_decoding_cache"):
                self.sequence_model._decoding_cache = None
            self.sequence_model._decoding_cache = update_graph_cache(
                self.sequence_model,
                self.sequence_model._decoding_cache,
                imagine_batch_size,
                seqlen_og,
                max_length,
                embedding_dim,
            )
            inference_params = self.sequence_model._decoding_cache.inference_params
            inference_params.reset(max_length, imagine_batch_size)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=imagine_batch_size, key_value_dtype=torch.bfloat16 if self.use_amp else None)

        
        def get_hidden_state(samples, action, inference_params):
            decoding = inference_params.seqlen_offset > 0

            if not self.use_cg or not decoding:
                hidden_state = self.sequence_model(
                    samples, action,
                    inference_params=inference_params,
                    # num_last_tokens=1,
                # ).logits.squeeze(dim=1)
                )
            else:
                hidden_state = self.sequence_model._decoding_cache.run(
                    samples, action, inference_params.seqlen_offset
                )
            return hidden_state        

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            # if eos_token_id is not None and (current_token == eos_token_id).all():
            #     return True
            if inference_params.seqlen_offset >= max_length:
                return True
            return False
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp and not self.use_cg):
            context_dist_feat = get_hidden_state(context_latent, sample_action, inference_params)
            inference_params.seqlen_offset += context_dist_feat.shape[1]
            conditioned_context_dist_feat = self.condition_dist_feat(context_dist_feat)
            context_prior_logits = self.dist_head.forward_prior(conditioned_context_dist_feat)
            context_prior_sample = self.stright_throught_gradient(context_prior_logits)
            context_flattened_sample = self.flatten_sample(context_prior_sample)

            dist_feat_list, sample_list = [
                conditioned_context_dist_feat[:, -1:]
            ], [context_flattened_sample[:, -1:]]
            self.sample_buffer[:, 0:1] = context_flattened_sample[:, -1:]
            self.dist_feat_buffer[:, 0:1] = conditioned_context_dist_feat[:, -1:]
            action_list, old_logits_list = [], []
            i = 0
            while not should_stop(sample_list[-1], inference_params):
                action, logits = agent.sample(torch.cat([self.sample_buffer[:, i:i+1], self.dist_feat_buffer[:, i:i+1]], dim=-1))
                action_list.append(action)
                self.action_buffer[:, i:i+1] = action
                old_logits_list.append(logits)
                dist_feat = get_hidden_state(sample_list[-1], action_list[-1], inference_params)
                conditioned_dist_feat = self.condition_dist_feat(dist_feat)
                dist_feat_list.append(conditioned_dist_feat)
                self.dist_feat_buffer[:, i+1:i+2] = conditioned_dist_feat
                inference_params.seqlen_offset += sample_list[-1].shape[1]
                # if repetition_penalty == 1.0:
                #     sampled_tokens = sample_tokens(scores[-1], inference_params)
                # else:
                #     logits = modify_logit_for_repetition_penalty(
                #         scores[-1].clone(), sequences_cat, repetition_penalty
                #     )
                #     sampled_tokens = sample_tokens(logits, inference_params)
                #     sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
                prior_logits = self.dist_head.forward_prior(dist_feat_list[-1])
                prior_sample = self.stright_throught_gradient(prior_logits)
                prior_flattened_sample = self.flatten_sample(prior_sample)
                sample_list.append(prior_flattened_sample)
                self.sample_buffer[:, i+1:i+2] = prior_flattened_sample
                i += 1
                    
                        
            # sample_tensor = torch.cat(sample_list, dim=1)
            # dist_feat_tensor = torch.cat(dist_feat_list, dim=1)
            # action_tensor = torch.cat(action_list, dim=1)
            old_logits_tensor = torch.cat(old_logits_list, dim=1)

            reward_hat_tensor = self.reward_decoder(self.dist_feat_buffer[:,:-1])
            self.reward_hat_buffer = self.symlog_twohot_loss_func.decode(reward_hat_tensor)
            termination_hat_tensor = self.termination_decoder(self.dist_feat_buffer[:,:-1])
            self.termination_hat_buffer = termination_hat_tensor > 0


        return torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1), self.action_buffer, old_logits_tensor, torch.cat([context_flattened_sample, conditioned_context_dist_feat], dim=-1), self.reward_hat_buffer, self.termination_hat_buffer


    def update(self, obs, action, reward, termination, global_step, epoch_step,
               logger=None, accum_steps: int = 1, is_last_accum: bool = True):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # dynamics models
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
                dist_feat = self.sequence_model(flattened_sample, action, temporal_mask)
            else:
                dist_feat = self.sequence_model(flattened_sample, action)
            conditioned_dist_feat = self.condition_dist_feat(dist_feat)
            prior_logits = self.dist_head.forward_prior(conditioned_dist_feat)

            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(conditioned_dist_feat)
            termination_hat = self.termination_decoder(conditioned_dist_feat)

            # env loss
            reconstruction_loss = self.reconstruction_loss_func(obs_hat[:batch_size], obs[:batch_size])
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + reward_loss + termination_loss + dynamics_loss + 0.1*representation_loss

        # gradient descent
        self.scaler.scale(total_loss / accum_steps).backward()
        if is_last_accum:
            self.scaler.unscale_(self.optimizer)  # for clip grad
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self.warmup_scheduler.dampen()

        return  reconstruction_loss.item(), reward_loss.item(), termination_loss.item(), \
                dynamics_loss.item(), dynamics_real_kl_div.item(), representation_loss.item(), \
                representation_real_kl_div.item(), total_loss.item()

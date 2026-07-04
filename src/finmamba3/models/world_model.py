# region imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange
from finmamba3.models.laprop import LaProp
from pytorch_warmup import LinearWarmup
from finmamba3.models.losses import CategoricalKLDivLossWithFreeBits, SymLogTwoHotLoss
from finmamba3.models.world_model_heads import DistHead, RewardHead, TerminationHead
from finmamba3.models.attention import get_subsequent_mask_with_batch_length
from finmamba3.models.transformer import StochasticTransformerKVCache, StochasticTransformerModernKVCache
from finmamba3.models.mamba_backbone import FinMambaSequenceModel
from finmamba3.models.regime_modulation import (
    efficiency_ratio_bucket_labels,
    efficiency_ratio_conditioning_feature,
    realized_vol_bucket_labels,
    realized_vol_conditioning_feature,
    regime_assignment_entropy,
    regime_load_balance_loss,
    regime_supervision_loss,
)
from finmamba3.models.lob_heads import (
    BookRelativeEdgeHead,
    DirectionHead,
    EpisodicMemory,
    EpisodicMemoryFuser,
    HawkesIntensityHead,
    RegimeConditioner,
    RegimeHead,
    SettlementHead,
)
from finmamba3.models.lob_encoder import (
    LOBDecoder,
    LOBEncoder,
    LOBReconstructionLoss,
    StudentTLOBDecoder,
    StudentTReconstructionLoss,
)
from finmamba3.rl.actor_critic import ActorCriticAgent
from finmamba3.weight_init import weight_init
# endregion
RMSNorm = nn.RMSNorm


class WorldModel(nn.Module):

    def __init__(self, action_dim, config, device):
        """Build the world model's encoder, sequence backbone, decoder and auxiliary heads from config.

        Notes on the RegimeFiLM and Direction knobs, kept here in full since a one-line
        why-comment at each assignment cannot carry the whole rationale:

        SuperviseVol removes the uniform-router failure mode where load balancing alone drives
        the router to a degenerate per-step uniform (so FiLM becomes a constant bias) by forcing
        the router onto the volatility axis the generalization eval splits on.

        SuperviseAxis: 'vol' is the realized-volatility bucket (the prior campaign's axis);
        'predictability' is the efficiency-ratio bucket (newgoal-2's axis, which a
        selective-betting policy has a genuine reason to encode).

        FeedObsVol: without it, the router only sees the latent-speed proxy, which tracks obs
        volatility too weakly for even a supervised router to discriminate regimes (verified:
        argmax-vs-vol agreement stays at chance).

        DecoupleRouterFromFiLM: without it, an adaptive optimizer keeps the router uniform no
        matter how the supervision weight is scaled, because the recon gradient direction on the
        shared params wins.

        ApplyAfterScan: the input-side affine folds losslessly into the block's own Delta, B and
        C projections, so the gauge direction joint training returns to identity; this flag is
        the non-gauge positive control, since a post-scan gate sits on the residual stream and
        the host projections cannot absorb it the same way.

        Direction.ClassBalanced: FI-2010's next-tick mid is flat on ~76% of ticks, so an
        unweighted cross-entropy is minimized by predicting the flat majority class and the head
        learns nothing about direction; inverse-frequency class weighting cancels that majority
        dominance.
        """
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
        self.encoder_type = config.Models.WorldModel.Encoder.get('Type', 'lob')
        if self.encoder_type != 'lob':
            raise ValueError("FinMamba3 only supports Models.WorldModel.Encoder.Type='lob'")
        regime_cfg = config.Models.WorldModel.get('Regime', None)
        self.use_regime = bool(regime_cfg is not None and regime_cfg.get('Enabled', False))
        # RegimeFiLM is the in-scan modulation path:
        # A regime hypernetwork emits per-block FiLM applied to each Mamba block input, distinct from the post-hoc self.use_regime conditioner.
        # Config reads mirror the sibling Regime block for cross-config safety.
        regime_film_cfg = config.Models.WorldModel.get('RegimeFiLM', None)
        self.use_regime_film = bool(regime_film_cfg is not None and regime_film_cfg.get('Enabled', False))
        self.regime_film_num_regimes = int(regime_film_cfg.get('NumRegimes', 8)) if regime_film_cfg is not None else 8
        self.regime_film_embed_dim = int(regime_film_cfg.get('EmbedDim', 32)) if regime_film_cfg is not None else 32
        self.regime_film_entropy_coef = float(regime_film_cfg.get('EntropyCoef', 0.01)) if regime_film_cfg is not None else 0.01
        # InitScale, Dropout, LRMult and WeightDecay are the FiLM tuning knobs:
        # A positive InitScale escapes the slow zero-init warmup, Dropout and a larger WeightDecay guard the hypernetwork's added capacity against overfitting, and LRMult lets the modulator learn faster than the backbone.
        self.regime_film_init_scale = float(regime_film_cfg.get('InitScale', 0.0)) if regime_film_cfg is not None else 0.0
        self.regime_film_dropout = float(regime_film_cfg.get('Dropout', 0.0)) if regime_film_cfg is not None else 0.0
        self.regime_film_lr_mult = float(regime_film_cfg.get('LRMult', 1.0)) if regime_film_cfg is not None else 1.0
        self.regime_film_weight_decay = regime_film_cfg.get('WeightDecay', None) if regime_film_cfg is not None else None
        # ConditionOnVol steers the router onto the volatility axis the generalization eval splits on.
        # The modulator derives a causal volatility proxy over VolWindow steps from the stem summary itself, so the conditioning is identical across training, imagination and evaluation forwards with no caller plumbing; the world model only forwards the flags.
        self.regime_film_condition_on_vol = bool(regime_film_cfg is not None and regime_film_cfg.get('ConditionOnVol', False))
        self.regime_film_vol_window = int(regime_film_cfg.get('VolWindow', 16)) if regime_film_cfg is not None else 16
        # SuperviseVol adds a cross-entropy from the regime logits to the realized-volatility bucket of each step, derived inline from the obs midprice.
        # It is the decision tree's highest-leverage lever:
        # It removes the uniform-router failure mode (load balancing alone drives the router to a degenerate per-step uniform, so FiLM becomes a constant bias) by forcing the router onto the volatility axis the generalization eval splits on.
        # SupervisionWeight scales that term.
        self.regime_film_supervise_vol = bool(regime_film_cfg is not None and regime_film_cfg.get('SuperviseVol', False))
        self.regime_film_supervision_weight = float(regime_film_cfg.get('SupervisionWeight', 0.0)) if regime_film_cfg is not None else 0.0
        # SuperviseAxis chooses what the router is supervised to predict:
        # 'vol' (the realized-volatility bucket, the prior campaign's axis) or 'predictability' (the efficiency-ratio bucket, newgoal-2's axis a selective-betting policy has a genuine reason to encode).
        # SuperviseChannelIndex points the supervision and the FeedObsVol conditioning at an obs channel (default the midprice); the spot block exposes the honest underlying axis the held-out eval splits on.
        self.regime_film_supervise_axis = regime_film_cfg.get('SuperviseAxis', 'vol') if regime_film_cfg is not None else 'vol'
        self.regime_film_supervise_index = regime_film_cfg.get('SuperviseChannelIndex', None) if regime_film_cfg is not None else None
        # FeedObsVol routes the realized volatility measured from the observation midprice into the router as its conditioning feature, the faithful axis the supervision label and eval split use.
        # Without it the router only sees the latent-speed proxy, which tracks obs volatility too weakly for even a supervised router to discriminate regimes (verified: argmax-vs-vol agreement stays at chance).
        self.regime_film_feed_obs_vol = bool(regime_film_cfg is not None and regime_film_cfg.get('FeedObsVol', False))
        # DecoupleRouterFromFiLM detaches the router logits on the path into the FiLM hypernetwork, so the reconstruction gradient can no longer drag the router back to uniform; the router is then driven purely by the supervision CE.
        # Without it, an adaptive optimizer keeps the router uniform no matter how the supervision weight is scaled, because the recon gradient direction on the shared params wins.
        self.regime_film_decouple_router = bool(regime_film_cfg is not None and regime_film_cfg.get('DecoupleRouterFromFiLM', False))
        # ApplyAfterScan moves the FiLM gate from each block's input to its output, past the selective scan.
        # The input-side affine folds losslessly into the block's own Delta, B and C projections, the gauge direction joint training returns to identity, so this flag is the non-gauge positive control:
        # A post-scan gate sits on the residual stream and the host projections cannot absorb it the same way.
        self.regime_film_apply_after_scan = bool(regime_film_cfg is not None and regime_film_cfg.get('ApplyAfterScan', False))
        assert not self.regime_film_condition_on_vol or self.regime_film_vol_window >= 2, (
            "Models.WorldModel.RegimeFiLM.VolWindow must be at least 2 to estimate volatility."
        )
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
            dim_feedforward=enc_cfg.get('DimFeedforward', enc_cfg.DModel * 2),
            dropout=config.Models.WorldModel.Dropout,
            output_flatten_dim=enc_cfg.OutputFlattenDim,
            aggregate_only=enc_cfg.get('AggregateOnly', False),
            gradient_checkpointing=enc_cfg.get('GradientCheckpointing', False),
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
        elif self.model == 'TransformerModern':
            self.sequence_model = StochasticTransformerModernKVCache(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=self.hidden_state_dim,
                num_layers=config.Models.WorldModel.Transformer.NumLayers,
                num_heads=config.Models.WorldModel.Transformer.NumHeads,
                max_length=max_seq_length,
                dropout=config.Models.WorldModel.Dropout,
                use_action_input=bool(config.Models.WorldModel.get('UseActionInput', True)),
                device=device,
                dtype=config.Models.WorldModel.dtype,
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
                use_regime_film=self.use_regime_film,
                num_regimes=self.regime_film_num_regimes,
                regime_embed_dim=self.regime_film_embed_dim,
                regime_condition_on_vol=self.regime_film_condition_on_vol,
                regime_vol_window=self.regime_film_vol_window,
                regime_init_scale=self.regime_film_init_scale,
                regime_dropout=self.regime_film_dropout,
                regime_decouple_router=self.regime_film_decouple_router,
                regime_apply_after_scan=self.regime_film_apply_after_scan,
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
                use_regime_film=self.use_regime_film,
                num_regimes=self.regime_film_num_regimes,
                regime_embed_dim=self.regime_film_embed_dim,
                regime_condition_on_vol=self.regime_film_condition_on_vol,
                regime_vol_window=self.regime_film_vol_window,
                regime_init_scale=self.regime_film_init_scale,
                regime_dropout=self.regime_film_dropout,
                regime_decouple_router=self.regime_film_decouple_router,
                regime_apply_after_scan=self.regime_film_apply_after_scan,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
        elif self.model == 'Mamba3':
            mamba3_cfg = config.Models.WorldModel.Mamba3
            if self.hidden_state_dim % mamba3_cfg.headdim != 0:
                raise ValueError("Models.WorldModel.HiddenStateDim must be divisible by Models.WorldModel.Mamba3.headdim")
            if not mamba3_cfg.get('Enabled', True):
                raise ValueError("Backbone is Mamba3 but Models.WorldModel.Mamba3.Enabled is false")
            self.sequence_model = FinMambaSequenceModel(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                d_model=self.hidden_state_dim,
                n_layer=mamba3_cfg.get('n_layer', config.Models.WorldModel.Mamba.n_layer),
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
                use_action_input=bool(config.Models.WorldModel.get('UseActionInput', True)),
                use_regime_film=self.use_regime_film,
                num_regimes=self.regime_film_num_regimes,
                regime_embed_dim=self.regime_film_embed_dim,
                regime_condition_on_vol=self.regime_film_condition_on_vol,
                regime_vol_window=self.regime_film_vol_window,
                regime_init_scale=self.regime_film_init_scale,
                regime_dropout=self.regime_film_dropout,
                regime_decouple_router=self.regime_film_decouple_router,
                regime_apply_after_scan=self.regime_film_apply_after_scan,
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
        decoder_kind = config.Models.WorldModel.Decoder.get('Kind', 'mse')
        self.decoder_kind = decoder_kind
        if decoder_kind == 'mse':
            self.obs_decoder = LOBDecoder(
                stoch_dim=self.stoch_flattened_dim,
                hidden_dim=config.Models.WorldModel.Decoder.get('HiddenDim', 512),
                num_layers=config.Models.WorldModel.Decoder.get('NumLayers', 3),
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                dtype=config.Models.WorldModel.dtype, device=device,
            )
        elif decoder_kind == 'studentt':
            self.obs_decoder = StudentTLOBDecoder(
                stoch_dim=self.stoch_flattened_dim,
                hidden_dim=config.Models.WorldModel.Decoder.get('HiddenDim', 512),
                num_layers=config.Models.WorldModel.Decoder.get('NumLayers', 3),
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                nu_init=float(config.Models.WorldModel.Decoder.get('NuInit', 5.0)),
                learnable_nu=bool(config.Models.WorldModel.Decoder.get('LearnableNu', True)),
                dtype=config.Models.WorldModel.dtype, device=device,
            )
        else:
            raise ValueError(f"Models.WorldModel.Decoder.Kind must be 'mse' or 'studentt'; got {decoder_kind!r}")
        self.use_reward_head = bool(config.Models.WorldModel.Reward.get('Enabled', True))
        self.use_termination_head = bool(config.Models.WorldModel.Termination.get('Enabled', True))
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
        direction_cfg = config.Models.WorldModel.get('Direction', None)
        self.use_direction_head = bool(direction_cfg is not None and direction_cfg.get('Enabled', False))
        if self.use_direction_head:
            self.direction_head = DirectionHead(
                hidden_dim=self.hidden_state_dim,
                num_classes=int(direction_cfg.get('NumClasses', 3)),
                dropout=float(direction_cfg.get('Dropout', 0.0)),
                dtype=config.Models.WorldModel.dtype, device=device,
            )
            self.direction_loss_weight = float(direction_cfg.get('LossWeight', 0.5))
            self.direction_threshold = float(direction_cfg.get('Threshold', 1.0e-2))
            self.direction_num_classes = int(direction_cfg.get('NumClasses', 3))
            # FI-2010's next-tick mid is flat on ~76% of ticks, so an unweighted cross-entropy is minimized by predicting the flat majority class for every position and the head learns nothing about direction.
            # Inverse-frequency class weighting cancels that majority dominance so the up/down classes carry gradient.
            # Default off to preserve prior runs.
            self.direction_class_balanced = bool(direction_cfg.get('ClassBalanced', False))
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
                num_regimes=regime_cfg.get('NumRegimes', 8),
                embed_dim=regime_cfg.get('EmbedDim', 32),
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.regime_conditioner = RegimeConditioner(
                hidden_dim=self.hidden_state_dim,
                regime_dim=regime_cfg.get('EmbedDim', 32),
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
        else:
            self.regime_head = None
            self.regime_conditioner = None
        em_cfg = config.Models.WorldModel.get('EpisodicMemory', None)
        self.use_episodic_memory = bool(em_cfg is not None and em_cfg.get('Enabled', False))
        if self.use_episodic_memory:
            self.episodic_memory = EpisodicMemory(
                key_dim=self.hidden_state_dim,
                value_dim=self.hidden_state_dim,
                capacity=int(em_cfg.get('Capacity', 50_000)),
                novelty_threshold=float(em_cfg.get('NoveltyThreshold', 0.0)),
            )
            self.memory_use_novelty = bool(em_cfg.get('UseNovelty', False))
            self.episodic_memory_fuser = EpisodicMemoryFuser(
                hidden_dim=self.hidden_state_dim,
                memory_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.memory_topk = int(em_cfg.get('TopK', 4))
            self.memory_min_fill = int(em_cfg.get('MinFillBeforeRetrieve', 1024))
            self.memory_retrieve_every = int(em_cfg.get('RetrieveEvery', 4))
            self._memory_steps_since_retrieve = self.memory_retrieve_every
        else:
            self.episodic_memory = None
            self.episodic_memory_fuser = None
            self.memory_topk = 0
            self.memory_min_fill = 0
            self.memory_retrieve_every = 0
            self._memory_steps_since_retrieve = 0
            self.memory_use_novelty = False
        hawkes_cfg = config.Models.WorldModel.get('Hawkes', None)
        self.use_hawkes_head = bool(hawkes_cfg is not None and hawkes_cfg.get('Enabled', False))
        if self.use_hawkes_head:
            self.hawkes_head = HawkesIntensityHead(
                hidden_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.hawkes_loss_weight = float(hawkes_cfg.get('LossWeight', 0.1))
        else:
            self.hawkes_head = None
            self.hawkes_loss_weight = 0.0
        settle_cfg = config.Models.WorldModel.get('Settlement', None)
        self.use_settlement_head = bool(settle_cfg is not None and settle_cfg.get('Enabled', False))
        if self.use_settlement_head:
            self.settlement_head = SettlementHead(
                hidden_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.settlement_loss_weight = float(settle_cfg.get('LossWeight', 0.25))
            # Phase 0.3 honest target:
            # TTEWeighted weights the realized-outcome BCE by (1 - tte_frac) so early-tick label noise stops dominating; SpotSignAux adds a dense pre-expiry term that supervises the head toward the observable running spot sign, weighted by tte_frac.
            self.settlement_tte_weighted = bool(settle_cfg.get('TTEWeighted', False))
            self.use_settlement_spot_sign = bool(settle_cfg.get('SpotSignAux', False))
            self.settlement_spot_sign_weight = float(settle_cfg.get('SpotSignWeight', 0.0))
        else:
            self.settlement_head = None
            self.settlement_loss_weight = 0.0
            self.settlement_tte_weighted = False
            self.use_settlement_spot_sign = False
            self.settlement_spot_sign_weight = 0.0
        edge_cfg = config.Models.WorldModel.get('EdgeHead', None)
        self.use_edge_head = bool(edge_cfg is not None and edge_cfg.get('Enabled', False))
        if self.use_edge_head:
            self.edge_head = BookRelativeEdgeHead(
                hidden_dim=self.hidden_state_dim,
                dtype=config.Models.WorldModel.dtype,
                device=device,
            )
            self.edge_huber_weight = float(edge_cfg.get('HuberWeight', 1.0))
            self.use_edge_action_ce = bool(edge_cfg.get('ActionCE', False))
            self.edge_action_ce_weight = float(edge_cfg.get('ActionCEWeight', 0.5))
            self.edge_threshold = float(edge_cfg.get('Threshold', 0.03))
        else:
            self.edge_head = None
            self.edge_huber_weight = 0.0
            self.use_edge_action_ce = False
            self.edge_action_ce_weight = 0.0
            self.edge_threshold = 0.03
        # When DirectionThresholds is a list, the direction loss is computed at each threshold and averaged.
        # This replaces the single-threshold (1%) target with a curve over thresholds.
        self.direction_thresholds = config.Models.WorldModel.get('DirectionThresholds', None)
        if self.direction_thresholds is not None:
            self.direction_thresholds = [float(t) for t in self.direction_thresholds]
        else:
            self.direction_thresholds = None
        # Decoder size_weight up-weights size and depth feature MSE relative to price features.
        # The 2.0 default biases the decoder toward volume signal but can crowd out predictive capacity for mid and spread.
        size_weight = float(config.Models.WorldModel.Decoder.get('SizeWeight', 2.0))
        # Per-feature size indices default to the Polymarket schema.
        # FI-2010 and other datasets with different level/tick orderings override these in the loss configs.
        level_size_indices = config.Models.WorldModel.Decoder.get('LevelSizeIndices', (1, 2, 3))
        tick_size_indices = config.Models.WorldModel.Decoder.get('TickSizeIndices', (6, 7, 11, 12))
        if self.decoder_kind == 'mse':
            self.reconstruction_loss_func = LOBReconstructionLoss(
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                size_weight=size_weight,
                level_size_indices=level_size_indices,
                tick_size_indices=tick_size_indices,
            )
        else:
            self.reconstruction_loss_func = StudentTReconstructionLoss(
                k_levels=enc_cfg.K,
                f_level=enc_cfg.FeatureDimLevel,
                f_tick=enc_cfg.FeatureDimTick,
                level_size_indices=level_size_indices,
                tick_size_indices=tick_size_indices,
            )
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        # Free bits floors the categorical KL so the dynamics term cannot drive the prior to copy the posterior trivially.
        # Larger values starve the dynamics signal; smaller values risk prior collapse.
        free_bits = float(config.Models.WorldModel.get('FreeBits', 1.0))
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=free_bits)
        # Representation loss weight controls how strongly the posterior is pulled toward the prior.
        # DreamerV3 uses 0.5; the original Drama default is 0.1.
        self.representation_loss_weight = float(config.Models.WorldModel.get('RepresentationLossWeight', 0.1))
        optimiser_name = config.Models.WorldModel.Optimiser
        weight_decay = config.Models.WorldModel.Weight_decay
        if optimiser_name == 'Laprop':
            base_lr = config.Models.WorldModel.Laprop.LearningRate
        elif optimiser_name == 'Adam':
            base_lr = config.Models.WorldModel.Adam.LearningRate
        else:
            raise ValueError(f"Unknown optimiser: {optimiser_name}")
        # The regime modulator gets its own optimizer group only when a tuning knob asks for it, so the default run stays a single group and bit-identical to before.
        # A larger LRMult lets the zero-init FiLM warm up faster; a WeightDecay override is the overfitting guard on its added capacity.
        if self.use_regime_film and (self.regime_film_lr_mult != 1.0 or self.regime_film_weight_decay is not None):
            film_prefix = "sequence_model.regime_modulator."
            base_named_params = [p for name, p in self.named_parameters() if not name.startswith(film_prefix)]
            film_named_params = [p for name, p in self.named_parameters() if name.startswith(film_prefix)]
            film_weight_decay = weight_decay if self.regime_film_weight_decay is None else float(self.regime_film_weight_decay)
            optimizer_params = [
                {"params": base_named_params, "lr": base_lr, "weight_decay": weight_decay},
                {"params": film_named_params, "lr": base_lr * self.regime_film_lr_mult, "weight_decay": film_weight_decay},
            ]
        else:
            optimizer_params = self.parameters()
        if optimiser_name == 'Laprop':
            self.optimizer = LaProp(optimizer_params, lr=base_lr, eps=config.Models.WorldModel.Laprop.Epsilon, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(optimizer_params, lr=base_lr, weight_decay=weight_decay)
        lr_schedule = config.Models.WorldModel.get('LRSchedule', 'constant')
        if lr_schedule == 'cosine':
            # Cosine + warmup is the standard recipe for Mamba pretraining.
            t_max = max(int(config.JointTrainAgent.SampleMaxSteps), 1)
            eta_min_ratio = float(config.Models.WorldModel.get('LRMinRatio', 0.1))
            base_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=base_lr * eta_min_ratio
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=config.Models.WorldModel.Warmup_steps)
        # Bf16 shares fp32's exponent range so loss scaling is a no-op semantically, but each unscale/step/update still issues an inf-check that syncs CPU-to-GPU.
        _amp_dtype = str(config.Models.WorldModel.dtype).lower()
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and _amp_dtype not in ("bfloat16", "bf16", "torch.bfloat16"),
        )
        # Watch for selective_scan and Mamba3 instability under bf16 autocast for the first NaNGuardSteps updates.
        # After that we trust the run.
        self.nan_guard_steps = int(config.Models.WorldModel.get('NaNGuardSteps', 50))
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
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    # Called only when using the Transformer backbone (requires KV cache).
    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.sequence_model.forward_with_kv_cache(last_flattened_sample, action)
            conditioned_dist_feat = self.condition_dist_feat(dist_feat)
            prior_logits = self.dist_head.forward_prior(conditioned_dist_feat)
            # Decode prior sample into observation.
            prior_sample = self.straight_through_gradient(prior_logits, sample_mode="random_sample")
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

    def straight_through_gradient(self, logits, sample_mode="random_sample"):
        # Sanitize logits before building the categorical distribution.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")
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

    def _imagine_data_full_prefix(self, agent: ActorCriticAgent, sample_obs, sample_action,
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
            context_prior_sample = self.straight_through_gradient(context_prior_logits)
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
                prior_sample = self.straight_through_gradient(prior_logits)
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

    def imagine_data(self, agent: ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):
        if self.model not in ('Transformer', 'TransformerModern'):
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

    def _direction_class_weight(self, flat_targets, dtype):
        """Per-batch inverse-frequency class weights for the direction cross-entropy, or None.

        Returns None when class balancing is off so the cross-entropy is the plain unweighted mean
        exactly as before. When on, each class is weighted by the reciprocal of its count in the
        batch so the flat majority no longer dominates the gradient; cross-entropy already divides by
        the summed target weights, so only the relative weights matter and an absent class's weight is
        never indexed.
        """
        if not self.direction_class_balanced:
            return None
        counts = torch.bincount(flat_targets, minlength=self.direction_num_classes).to(dtype=dtype)
        return counts.clamp(min=1.0).reciprocal()

    def update(self, obs, action, reward, termination, global_step, epoch_step,
               logger=None, accum_steps: int = 1, is_last_accum: bool = True,
               event_counts: torch.Tensor | None = None,
               outcome: torch.Tensor | None = None,
               time_to_expiry_frac: torch.Tensor | None = None,
               spot_signed_distance: torch.Tensor | None = None,
               yes_ask: torch.Tensor | None = None,
               no_ask: torch.Tensor | None = None,
               yes_mid: torch.Tensor | None = None,
               book_depth: torch.Tensor | None = None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # Encode observations into posterior samples.
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.straight_through_gradient(post_logits, sample_mode="random_sample")
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
            regime_aux = None
            if self.model in ('Transformer', 'TransformerModern'):
                temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
                dist_feat = self.sequence_model(flattened_sample, action, temporal_mask)
            elif self.use_regime_film:
                # When FeedObsVol is set, hand the router the realized volatility measured from the obs midprice so it can actually discriminate the regime it is supervised toward; otherwise the modulator falls back to its latent-speed proxy.
                regime_vol = None
                if self.regime_film_feed_obs_vol:
                    assert self.midprice_index >= 0, (
                        "RegimeFiLM.FeedObsVol needs the obs midprice channel index; enable the direction head."
                    )
                    supervise_index = self.regime_film_supervise_index if self.regime_film_supervise_index is not None else self.midprice_index
                    if self.regime_film_supervise_axis == "predictability":
                        regime_vol = efficiency_ratio_conditioning_feature(obs[..., supervise_index], self.regime_film_vol_window)
                    else:
                        regime_vol = realized_vol_conditioning_feature(obs[..., supervise_index], self.regime_film_vol_window)
                dist_feat, regime_aux = self.sequence_model(
                    flattened_sample, action, return_regime=True, regime_vol=regime_vol
                )
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
                flat_logits = direction_logits.reshape(-1, direction_logits.shape[-1])
                if self.direction_thresholds is not None:
                    # Multi-threshold sweep averages cross-entropy across the threshold curve.
                    # This stops the head from being pinned to a single bucket.
                    losses = []
                    for thr in self.direction_thresholds:
                        targets, _ = DirectionHead.make_targets(mid_norm, float(thr))
                        flat_targets = targets.reshape(-1)
                        losses.append(
                            F.cross_entropy(
                                flat_logits, flat_targets,
                                weight=self._direction_class_weight(flat_targets, flat_logits.dtype),
                            )
                        )
                    direction_loss = torch.stack(losses).mean()
                else:
                    direction_targets, _ = DirectionHead.make_targets(mid_norm, self.direction_threshold)
                    flat_targets = direction_targets.reshape(-1)
                    direction_loss = F.cross_entropy(
                        flat_logits, flat_targets,
                        weight=self._direction_class_weight(flat_targets, flat_logits.dtype),
                    )
            else:
                direction_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            # Hawkes intensity auxiliary predicts log-intensity for buy and sell arrivals.
            # The term only contributes when event_counts is provided with shape (B, L, 2) for buy and sell counts per tick or bar.
            if self.use_hawkes_head and event_counts is not None:
                hawkes_logits = self.hawkes_head(conditioned_dist_feat)
                hawkes_loss = HawkesIntensityHead.poisson_nll(hawkes_logits, event_counts.to(dtype=hawkes_logits.dtype))
            else:
                hawkes_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            # Settlement auxiliary predicts the binary contract outcome from the latent.
            # The term only contributes when outcome is provided as a broadcastable scalar per sequence with shape (B,) or (B, L).
            if self.use_settlement_head and outcome is not None:
                settle_logits = self.settlement_head(conditioned_dist_feat).reshape(-1)
                outcome_flat = outcome.reshape(-1).to(dtype=settle_logits.dtype)
                if outcome_flat.shape[0] != settle_logits.shape[0] and outcome.dim() == 1:
                    outcome_flat = outcome.unsqueeze(1).expand(-1, conditioned_dist_feat.shape[1]).reshape(-1).to(dtype=settle_logits.dtype)
                # Phase 0.3: concentrate the realized-outcome supervision near expiry, where the binary outcome is determined, rather than pooling the uninformative early ticks with it.
                ttf_flat = None
                if self.settlement_tte_weighted and time_to_expiry_frac is not None:
                    ttf_flat = time_to_expiry_frac.reshape(-1).to(dtype=settle_logits.dtype)
                settlement_loss = SettlementHead.bce(settle_logits, outcome_flat, ttf_flat)
                # The observable running spot sign is the same rule the contract settles on, so an early auxiliary toward it gives a dense, learnable target and keeps the causal sign salient in the latent.
                # `tte_frac` weights it toward the start (complement of the outcome).
                if self.use_settlement_spot_sign and spot_signed_distance is not None:
                    aux_ttf = time_to_expiry_frac.reshape(-1) if time_to_expiry_frac is not None else None
                    spot_sign_loss = SettlementHead.spot_sign_bce(settle_logits, spot_signed_distance.reshape(-1), aux_ttf)
                    settlement_loss = settlement_loss + self.settlement_spot_sign_weight * spot_sign_loss
            else:
                settlement_loss = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            # Book-relative edge head:
            # Supervise ev_yes_hat and ev_no_hat directly from the book ask prices and the realized settlement outcome.
            # Rows with NaN asks or zero depth are masked out so existing configs without edge data run unchanged.
            edge_zero = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            if (self.use_edge_head and outcome is not None and
                    yes_ask is not None and no_ask is not None and book_depth is not None):
                ev_hat = self.edge_head(conditioned_dist_feat).float()
                outcome_f = outcome.to(ev_hat.dtype)
                yes_ask_f = yes_ask.to(ev_hat.dtype)
                no_ask_f = no_ask.to(ev_hat.dtype)
                book_depth_f = book_depth.to(ev_hat.dtype)
                mask = BookRelativeEdgeHead.finite_mask(outcome_f, yes_ask_f, no_ask_f, book_depth_f)
                ev_yes_t, ev_no_t = BookRelativeEdgeHead.ev_targets(outcome_f, yes_ask_f, no_ask_f)
                edge_huber_loss = BookRelativeEdgeHead.huber_ev_loss(ev_hat, ev_yes_t, ev_no_t, mask)
                if self.use_edge_action_ce:
                    edge_action_loss = BookRelativeEdgeHead.action_ce_loss(
                        ev_hat, ev_yes_t, ev_no_t, mask, self.edge_threshold
                    )
                else:
                    edge_action_loss = edge_zero
            else:
                edge_huber_loss = edge_zero
                edge_action_loss = edge_zero
            # Regime-FiLM load-balance regularizer keeps the inferred regime distribution from collapsing.
            # The entropy, gamma deviation and beta magnitude reported beside it are diagnostics only:
            # They reveal whether the router collapsed and whether FiLM ever left its identity init, never the loss.
            regime_zero = torch.zeros((), device=obs.device, dtype=reconstruction_loss.dtype)
            if self.use_regime_film and regime_aux is not None:
                regime_loss = (self.regime_film_entropy_coef * regime_load_balance_loss(regime_aux.regime_logits)).to(reconstruction_loss.dtype)
                # Regime supervision pins the router to the realized-volatility axis so FiLM stops being a constant bias.
                # The bucket labels come from the obs midprice over the same VolWindow the router's vol proxy uses, so training, imagination and eval see one consistent definition.
                if self.regime_film_supervise_vol:
                    assert self.midprice_index >= 0, (
                        "RegimeFiLM.SuperviseVol needs the obs midprice channel index, which is only set "
                        "when the direction head is enabled; enable Models.WorldModel.Direction."
                    )
                    supervise_index = self.regime_film_supervise_index if self.regime_film_supervise_index is not None else self.midprice_index
                    if self.regime_film_supervise_axis == "predictability":
                        regime_labels = efficiency_ratio_bucket_labels(
                            obs[:batch_size, :, supervise_index], self.regime_film_vol_window, self.regime_film_num_regimes
                        )
                    else:
                        regime_labels = realized_vol_bucket_labels(
                            obs[:batch_size, :, supervise_index], self.regime_film_vol_window, self.regime_film_num_regimes
                        )
                    supervision = regime_supervision_loss(regime_aux.regime_logits[:batch_size], regime_labels)
                    regime_loss = regime_loss + (self.regime_film_supervision_weight * supervision).to(reconstruction_loss.dtype)
                regime_entropy = regime_assignment_entropy(regime_aux.regime_logits).to(reconstruction_loss.dtype)
                film_gamma_dev = regime_aux.gamma_dev.to(reconstruction_loss.dtype)
                film_beta_mag = regime_aux.beta_mag.to(reconstruction_loss.dtype)
            else:
                regime_loss = regime_zero
                regime_entropy = regime_zero
                film_gamma_dev = regime_zero
                film_beta_mag = regime_zero
            total_loss = (
                reconstruction_loss + reward_loss + termination_loss
                + dynamics_loss + self.representation_loss_weight * representation_loss
                + self.direction_loss_weight * direction_loss
                + self.hawkes_loss_weight * hawkes_loss
                + self.settlement_loss_weight * settlement_loss
                + self.edge_huber_weight * edge_huber_loss
                + self.edge_action_ce_weight * edge_action_loss
                + regime_loss
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
            return (zero,) * 17
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
            if self.memory_use_novelty:
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
            hawkes_loss.detach(), settlement_loss.detach(), regime_loss.detach(),
            film_gamma_dev.detach(), film_beta_mag.detach(), regime_entropy.detach(),
            edge_huber_loss.detach(), edge_action_loss.detach(), total_loss.detach(),
        )

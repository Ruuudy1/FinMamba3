"""Regime inference and FiLM modulation of the Mamba selection mechanism."""
# region imports
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from finmamba3.models.lob_heads import RegimeHead
# endregion


@dataclass
class RegimeAux:
    """Per-forward regime diagnostics the backbone surfaces for offline logging.

    regime_logits feeds the load-balance regularizer downstream; gamma_dev and beta_mag are
    the mean absolute departure of the FiLM scale from one and of the FiLM shift from zero, so a
    near-zero pair is the signature of a modulator still pinned to its zero-init identity, meaning
    FiLM is inert and the treatment arm is silently a baseline.
    """
    regime_logits: torch.Tensor
    gamma_dev: torch.Tensor
    beta_mag: torch.Tensor


def regime_load_balance_loss(regime_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the batch-averaged regime distribution.

    Minimizing this term maximizes the entropy of the mean regime assignment,
    which spreads usage across regimes (Switch-Transformer load balancing) and
    stops the inference network collapsing onto a single regime. Per-step
    assignments stay free to be peaky; only the batch average is regularized.
    """
    probs = torch.softmax(regime_logits.float(), dim=-1)
    mean_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
    num_regimes = mean_probs.shape[0]
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(float(num_regimes), device=entropy.device))
    return max_entropy - entropy


def regime_assignment_entropy(regime_logits: torch.Tensor) -> torch.Tensor:
    """Entropy in nats of the batch-averaged regime distribution, for logging only.

    This is the diagnostic counterpart of regime_load_balance_loss without the max-entropy
    offset: a value collapsing toward zero means the router has concentrated on a few regimes, so
    the FiLM modulation degenerates into a near-constant bias that folds into the existing block
    biases and cannot adapt across regimes. It is reported, never added to the loss.
    """
    probs = torch.softmax(regime_logits.float(), dim=-1)
    mean_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
    return -(mean_probs * torch.log(mean_probs + 1e-8)).sum()


def regime_per_sample_entropy(regime_logits: torch.Tensor) -> torch.Tensor:
    """Mean over steps of each step's own regime-distribution entropy, in nats, for logging only.

    The batch-mean entropy (regime_assignment_entropy) stays at ln(R) whenever the router is either
    genuinely regime-balanced or degenerately uniform-per-step, so it cannot tell those apart. This
    per-sample entropy instead drops toward zero exactly when each step commits to one regime, which is
    the signature that regime supervision has made the router discriminative rather than a constant
    averaged embedding. Reported beside reg_H, never added to the loss.
    """
    log_probs = torch.log_softmax(regime_logits.float(), dim=-1)
    probs = log_probs.exp()
    per_step_entropy = -(probs * log_probs).sum(dim=-1)
    return per_step_entropy.mean()


def causal_realized_vol(midprice: torch.Tensor, window: int) -> torch.Tensor:
    """Causal trailing std of midprice increments over `window` steps, shape [B, L].

    This is the realized volatility the generalization eval splits on, measured directly from the
    observation midprice. Left zero-padding makes the earliest steps read low-vol by construction, so the
    estimate stays causal. Both the regime-supervision bucket labels and the router conditioning feature
    derive from this one quantity, so the router sees exactly the axis it is supervised to predict.
    """
    delta = torch.zeros_like(midprice)
    delta[:, 1:] = midprice[:, 1:] - midprice[:, :-1]
    padded = F.pad(delta, (window - 1, 0))
    rolling = padded.unfold(dimension=1, size=window, step=1)
    return rolling.std(dim=-1, unbiased=False)


def realized_vol_conditioning_feature(midprice: torch.Tensor, window: int) -> torch.Tensor:
    """The router conditioning feature: batch-standardized log realized vol, shape [B, L, 1].

    The raw realized vol of z-scored FI-2010 mids is ~1e-3, so a raw log1p feature is ~1000x smaller than
    the O(1) hidden-summary channels it is concatenated with at the router head and is numerically drowned
    out (verified: a head reading it cannot fit the vol bucket it perfectly determines). Standardizing the
    log vol over the batch brings it to unit scale so the single conditioning channel actually competes
    with the hidden channels. The transform is affine and rank-preserving, so it still tracks the exact
    volatility axis the supervision label buckets; standardizing per forward batch keeps training and eval
    consistent in method.
    """
    log_vol = torch.log1p(causal_realized_vol(midprice, window))
    mean = log_vol.mean()
    std = log_vol.std().clamp(min=1e-6)
    return ((log_vol - mean) / std).unsqueeze(-1)


def realized_vol_bucket_labels(midprice: torch.Tensor, window: int, num_buckets: int) -> torch.Tensor:
    """Per-step volatility-bucket labels for regime supervision, shape [B, L] long in [0, num_buckets).

    The router is otherwise pinned uniform by the load-balance term, so FiLM degenerates into a constant
    bias that cannot adapt across regimes. Supervising the regime logits to predict which volatility
    bucket each step falls in forces the router onto the same realized-volatility axis the generalization
    eval splits on, so the emitted FiLM actually varies across regimes. Buckets are the batch volatility
    quantiles, which keeps occupancy balanced (so the supervision agrees with load balancing) while still
    forcing a confident, vol-correlated per-step assignment.
    """
    vol = causal_realized_vol(midprice, window)
    quantile_points = torch.linspace(0.0, 1.0, num_buckets + 1, device=vol.device, dtype=torch.float32)[1:-1]
    thresholds = torch.quantile(vol.reshape(-1).float(), quantile_points)
    return torch.bucketize(vol, thresholds).long()


def causal_efficiency_ratio(series: torch.Tensor, window: int) -> torch.Tensor:
    """Causal trailing Kaufman efficiency ratio of a channel over `window` steps, shape [B, L] in [0, 1].

    ER is the net move over the trailing window divided by the total path length: near 1 on a clean
    trend, near 0 on a random wander. Both the net move and the path are summed from the same per-step
    increments, left zero-padded so the estimate stays causal. This is the predictability axis a FiLM
    router has a genuine reason to encode (newgoal-2): "this window is forecastable" is the latent a
    selective-betting policy needs.
    """
    delta = torch.zeros_like(series)
    delta[:, 1:] = series[:, 1:] - series[:, :-1]
    padded = F.pad(delta, (window - 1, 0))
    windows = padded.unfold(dimension=1, size=window, step=1)
    net_move = windows.sum(dim=-1).abs()
    path_length = windows.abs().sum(dim=-1)
    return net_move / (path_length + 1e-8)


def efficiency_ratio_bucket_labels(series: torch.Tensor, window: int, num_buckets: int) -> torch.Tensor:
    """Per-step predictability-bucket labels for regime supervision, shape [B, L] long in [0, num_buckets).

    Buckets the causal efficiency ratio at the batch quantiles so occupancy stays balanced (agreeing with
    load balancing) while forcing a confident, predictability-correlated per-step assignment. The highest
    bucket is the trending / forecastable regime where a selective policy should trade.
    """
    er = causal_efficiency_ratio(series, window)
    quantile_points = torch.linspace(0.0, 1.0, num_buckets + 1, device=er.device, dtype=torch.float32)[1:-1]
    thresholds = torch.quantile(er.reshape(-1).float(), quantile_points)
    return torch.bucketize(er, thresholds).long()


def efficiency_ratio_conditioning_feature(series: torch.Tensor, window: int) -> torch.Tensor:
    """Batch-standardized causal efficiency ratio as the router conditioning feature, shape [B, L, 1].

    Mirrors realized_vol_conditioning_feature but on the predictability axis: the router reads a
    faithful, linearly-bucketable forecastability signal so the supervision can make it discriminate.
    Standardizing centers the already O(1) ratio so the single channel competes with the hidden ones.
    """
    er = causal_efficiency_ratio(series, window)
    mean = er.mean()
    std = er.std().clamp(min=1e-6)
    return ((er - mean) / std).unsqueeze(-1)


def regime_supervision_loss(regime_logits: torch.Tensor, vol_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy from the per-step regime logits to the realized-volatility bucket labels.

    This is the highest-leverage regime-FiLM lever: it removes the uniform-router failure mode by making
    the router predict the volatility regime directly, so the modulation it emits is regime-specific
    instead of a constant bias averaged over a degenerate-uniform assignment.
    """
    num_regimes = regime_logits.shape[-1]
    return F.cross_entropy(regime_logits.reshape(-1, num_regimes).float(), vol_labels.reshape(-1))


class RegimeFiLMModulator(nn.Module):
    """Infers a latent regime and emits per-block FiLM scale and shift.

    The regime is inferred causally per timestep from the Mamba stem summary, optionally
    concatenated with an external conditioning feature (a causal volatility proxy), then
    a hypernetwork maps the soft regime embedding to per-block channel-wise scale
    (gamma) and shift (beta). Applying these to each Mamba block's input modulates
    the selective parameters Delta, B and C, which are input-dependent, without
    touching the CUDA kernel. With init_scale=0 the hypernetwork is zero-initialized so
    gamma=1 and beta=0 at the start, meaning an untrained modulator exactly reproduces the
    unmodulated backbone and the regime-off baseline is recovered; a small positive init_scale
    breaks that identity so the modulator escapes the flat zero-init region sooner, trading
    exact baseline parity for a faster warmup.

    With condition_on_vol the router input is the stem summary concatenated with a causal latent
    volatility proxy the modulator computes from that same summary, which steers the regime onto
    the volatility axis the generalization eval splits on instead of whatever happens to minimize
    training reconstruction. The proxy is derived in-module from the summary alone, so every forward
    path (training, imagination rollouts, evaluation) produces it identically with no caller plumbing.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layer: int,
        num_regimes: int = 8,
        embed_dim: int = 32,
        condition_on_vol: bool = False,
        vol_window: int = 16,
        init_scale: float = 0.0,
        dropout: float = 0.0,
        decouple_router: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.num_regimes = num_regimes
        self.condition_on_vol = condition_on_vol
        self.vol_window = vol_window
        # With an adaptive optimizer, up-weighting the supervision loss does not proportionally move the
        # shared router params, so the reconstruction gradient flowing back through the FiLM path keeps the
        # router uniform regardless of supervision weight. decouple_router detaches the router logits on the
        # path into the hypernetwork, so reconstruction can no longer push the router toward uniform: the
        # router is driven purely by the supervision CE while the hypernetwork still adapts the per-regime
        # modulation to minimize reconstruction. This is the clean test of whether a genuinely
        # regime-discriminative FiLM helps, isolated from the optimizer's preference for a constant bias.
        self.decouple_router = decouple_router
        cond_dim = 1 if condition_on_vol else 0
        self.regime_head = RegimeHead(hidden_dim + cond_dim, num_regimes, embed_dim, **factory)
        # Dropout on the soft regime embedding is the overfitting guard on the hypernetwork's added
        # capacity; it stays a no-op at the default rate of zero.
        self.embedding_dropout = nn.Dropout(dropout)
        self.hyper = nn.Linear(embed_dim, 2 * n_layer * hidden_dim, **factory)
        # A positive init_scale seeds the hypernetwork weights with a small Gaussian so FiLM departs
        # from identity immediately; the default of zero preserves the exact identity-at-init baseline.
        if init_scale > 0.0:
            nn.init.normal_(self.hyper.weight, mean=0.0, std=init_scale)
        else:
            nn.init.zeros_(self.hyper.weight)
        nn.init.zeros_(self.hyper.bias)
    def _causal_latent_volatility(self, hidden_summary: torch.Tensor) -> torch.Tensor:
        """Per-step latent speed smoothed into a causal volatility proxy of shape [B, L, 1].

        The stem summary moves fast when the order book is churning, so the L2 norm of its
        first difference is a volatility surrogate available in every forward path. A trailing
        rolling std over vol_window steps turns that instantaneous speed into a regime-scale
        signal; left zero-padding makes the earliest steps low-vol by construction.
        """
        delta = torch.zeros_like(hidden_summary)
        delta[:, 1:] = hidden_summary[:, 1:] - hidden_summary[:, :-1]
        speed = delta.float().norm(dim=-1)
        padded_speed = F.pad(speed, (self.vol_window - 1, 0))
        rolling = padded_speed.unfold(dimension=1, size=self.vol_window, step=1)
        return torch.log1p(rolling.std(dim=-1, unbiased=False)).unsqueeze(-1)
    def forward(
        self, hidden_summary: torch.Tensor, external_vol: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.condition_on_vol:
            # external_vol is the realized volatility measured directly from the observation midprice,
            # the same axis the generalization eval splits on and the regime-supervision label derives
            # from. When the caller supplies it the router reads a faithful, linearly-bucketable vol
            # signal so supervision can make the router discriminate; the latent-speed proxy is the
            # caller-free fallback (it tracks obs volatility only weakly). The proxy/feature is cast back
            # to the summary dtype so the concat stays valid under bf16 autocast.
            if external_vol is not None:
                vol_feature = external_vol.to(hidden_summary.dtype)
            else:
                vol_feature = self._causal_latent_volatility(hidden_summary).to(hidden_summary.dtype)
            router_input = torch.cat([hidden_summary, vol_feature], dim=-1)
        else:
            router_input = hidden_summary
        # regime_logits feeds the supervision CE and the load-balance diagnostic with full gradient. The
        # embedding handed to the hypernetwork softmaxes a detached copy when decoupled, so reconstruction
        # adapts the per-regime modulation without dragging the router back to uniform.
        regime_logits = self.regime_head.logits(router_input)
        embedding_logits = regime_logits.detach() if self.decouple_router else regime_logits
        regime_emb = torch.softmax(embedding_logits, dim=-1) @ self.regime_head.embedding.weight
        regime_emb = self.embedding_dropout(regime_emb)
        batch_size = hidden_summary.shape[0]
        seq_len = hidden_summary.shape[1]
        film = self.hyper(regime_emb)
        film = film.reshape(batch_size, seq_len, self.n_layer, 2, self.hidden_dim)
        # Tanh keeps gamma in (0, 2) and beta in (-1, 1) for bf16 stability; both
        # equal the identity (gamma=1, beta=0) while the zero-init hypernetwork warms up.
        gammas = 1.0 + torch.tanh(film[..., 0, :])
        betas = torch.tanh(film[..., 1, :])
        return gammas, betas, regime_logits

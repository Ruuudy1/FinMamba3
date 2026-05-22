"""Regime inference and FiLM modulation of the Mamba selection mechanism."""
# region imports
from __future__ import annotations
import torch
import torch.nn as nn
from sub_models.lob_auxiliary import RegimeHead
# endregion


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


class RegimeFiLMModulator(nn.Module):
    """Infers a latent regime and emits per-block FiLM scale and shift.

    The regime is inferred causally per timestep from the Mamba stem summary, then
    a hypernetwork maps the soft regime embedding to per-block channel-wise scale
    (gamma) and shift (beta). Applying these to each Mamba block's input modulates
    the selective parameters Delta, B and C, which are input-dependent, without
    touching the CUDA kernel. The hypernetwork is zero-initialized so gamma=1 and
    beta=0 at the start, meaning an untrained modulator exactly reproduces the
    unmodulated backbone and the regime-off baseline is recovered.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layer: int,
        num_regimes: int = 8,
        embed_dim: int = 32,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.num_regimes = num_regimes
        self.regime_head = RegimeHead(hidden_dim, num_regimes, embed_dim, **factory)
        self.hyper = nn.Linear(embed_dim, 2 * n_layer * hidden_dim, **factory)
        nn.init.zeros_(self.hyper.weight)
        nn.init.zeros_(self.hyper.bias)
    def forward(self, hidden_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regime_logits, regime_emb = self.regime_head(hidden_summary)
        batch_size = hidden_summary.shape[0]
        seq_len = hidden_summary.shape[1]
        film = self.hyper(regime_emb)
        film = film.reshape(batch_size, seq_len, self.n_layer, 2, self.hidden_dim)
        # Tanh keeps gamma in (0, 2) and beta in (-1, 1) for bf16 stability; both
        # equal the identity (gamma=1, beta=0) while the zero-init hypernetwork warms up.
        gammas = 1.0 + torch.tanh(film[..., 0, :])
        betas = torch.tanh(film[..., 1, :])
        return gammas, betas, regime_logits

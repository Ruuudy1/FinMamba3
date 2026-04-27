"""Transformer encoder and MLP decoder for Polymarket LOB features.

Pluggable replacement for the image CNN in world_models.py. The encoder
must expose `output_flatten_dim` so DistHead wires identically.

Input shape: (B, L, F) where F = K*F_level + F_tick (the flat feature
vector from the replay buffer).
Output shape: (B, L, output_flatten_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, reduce

from envs.lob_features import F_LEVEL, F_TICK, K_LEVELS

RMSNorm = nn.RMSNorm


class LOBEncoder(nn.Module):
    """Transformer over K depth-curve tokens plus a CLS token carrying the
    tick-level scalar features.
    """

    def __init__(
        self,
        k_levels: int = K_LEVELS,
        f_level: int = F_LEVEL,
        f_tick: int = F_TICK,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_flatten_dim: int = 1024,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.k_levels = k_levels
        self.f_level = f_level
        self.f_tick = f_tick
        self.d_model = d_model
        self.output_flatten_dim = output_flatten_dim

        factory = {"dtype": dtype, "device": device}

        self.level_proj = nn.Linear(f_level, d_model, **factory)
        self.level_pos = nn.Parameter(torch.zeros(k_levels, d_model, **factory))
        nn.init.normal_(self.level_pos, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model, **factory))
        nn.init.normal_(self.cls_token, std=0.02)

        self.tick_proj = nn.Sequential(
            nn.Linear(f_tick, d_model, **factory),
            nn.SiLU(),
            nn.Linear(d_model, d_model, **factory),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **factory,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = RMSNorm(d_model, **factory)
        self.out_proj = nn.Linear(d_model, output_flatten_dim, **factory)

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        level_flat_dim = self.k_levels * self.f_level
        levels_flat = x[..., :level_flat_dim]
        tick = x[..., level_flat_dim:]
        levels = rearrange(
            levels_flat, "b l (k f) -> b l k f", k=self.k_levels, f=self.f_level
        )
        return levels, tick

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        levels, tick = self._split_input(x)
        B, L = levels.shape[:2]
        tok = self.level_proj(levels) + self.level_pos
        tok = rearrange(tok, "b l k d -> (b l) k d")

        cls = self.cls_token.expand(B * L, -1, -1)
        tick_emb = rearrange(self.tick_proj(tick), "b l d -> (b l) 1 d")
        cls = cls + tick_emb
        seq = torch.cat([cls, tok], dim=1)

        seq = self.transformer(seq)
        cls_out = self.norm(seq[:, 0])
        out = self.out_proj(cls_out)
        return rearrange(out, "(b l) d -> b l d", b=B, l=L)


class LOBDecoder(nn.Module):
    """MLP predicting the flat LOB feature vector from a sampled stoch latent."""

    def __init__(
        self,
        stoch_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        k_levels: int = K_LEVELS,
        f_level: int = F_LEVEL,
        f_tick: int = F_TICK,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        out_dim = k_levels * f_level + f_tick
        layers: list[nn.Module] = []
        in_dim = stoch_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, **factory))
            layers.append(RMSNorm(hidden_dim, **factory))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim, **factory))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class LOBReconstructionLoss(nn.Module):
    """Weighted MSE over the flat feature vector, reduced "B L F -> B L".

    Size/flow features are up-weighted because volume carries more signal
    than absolute price magnitudes.
    """

    def __init__(
        self,
        k_levels: int = K_LEVELS,
        f_level: int = F_LEVEL,
        f_tick: int = F_TICK,
        level_weight: float = 1.0,
        size_weight: float = 2.0,
        tick_weight: float = 1.0,
    ) -> None:
        super().__init__()
        # F_LEVEL dims: rel_price, log_size, cum_depth, vol_share, gap,
        # lvl_idx, side, staleness. F_TICK dims defined in lob_features.py.
        level_w = torch.full((f_level,), level_weight, dtype=torch.float32)
        level_w[1] = size_weight
        level_w[2] = size_weight
        level_w[3] = size_weight
        tick_w = torch.full((f_tick,), tick_weight, dtype=torch.float32)
        tick_w[6] = size_weight
        tick_w[7] = size_weight
        tick_w[11] = size_weight
        tick_w[12] = size_weight
        weight = torch.cat([level_w.repeat(k_levels), tick_w], dim=0)
        weight = weight * (weight.numel() / weight.sum())
        self.register_buffer("feature_weight", weight, persistent=False)

    def forward(self, obs_hat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        w = self.feature_weight.to(dtype=obs.dtype)
        sq = (obs_hat - obs) ** 2 * w
        loss = reduce(sq, "B L F -> B L", "sum")
        return loss.mean()

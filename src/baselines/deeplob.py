"""DeepLOB baseline (Zhang et al. 2018) for LOB direction prediction.

Reference architecture:
- 3 stacks of (Conv2d 1x2 stride 1x2, LeakyReLU, Conv 4x1, LeakyReLU,
  Conv 4x1, LeakyReLU) operating across price-size pairs and depth levels.
- Inception module with three parallel branches (1x1, 3x1 conv stack,
  5x1 conv stack, max-pool 3x1).
- LSTM with 64 hidden units.
- Linear head to 3-class direction logits (down / flat / up).

The original paper consumes a 100x40 raw LOB tensor. Polymarket has K=10
levels and 8 features per level (the per-level half of the 94-dim FinDrama
feature). This adapter reshapes that into the (T, 1, 10, 8) tensor DeepLOB
expects so the baseline runs on the same data the FinDrama encoder sees.

Use this as a benchmarking comparator, not as a component of the world model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from envs.lob_features import F_LEVEL, K_LEVELS


class _ConvStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Time-direction convolutions use kernel (3, 1) with padding (1, 0)
        # so the time axis is preserved exactly. Original DeepLOB uses
        # padding='same' which is the same effect.
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _Inception(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch_1x1(x), self.branch_3(x), self.branch_5(x), self.branch_pool(x)],
            dim=1,
        )


class DeepLOB(nn.Module):
    """Minimal DeepLOB implementation adapted to the Polymarket K x F_LEVEL grid.

    Input shape: (B, L, K * F_LEVEL) where the per-level features are laid
    out flat as in the FinDrama feature vector. The forward call internally
    reshapes to (B, 1, L, K * F_LEVEL) so the convolutions slide over time
    and across the depth axis.
    """

    def __init__(
        self,
        k_levels: int = K_LEVELS,
        f_level: int = F_LEVEL,
        conv_channels: int = 16,
        inception_channels: int = 16,
        lstm_hidden: int = 64,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.k_levels = int(k_levels)
        self.f_level = int(f_level)
        self.input_width = self.k_levels * self.f_level

        self.stack1 = _ConvStack(1, conv_channels)
        self.stack2 = _ConvStack(conv_channels, conv_channels)
        self.stack3 = _ConvStack(conv_channels, conv_channels)
        self.inception = _Inception(conv_channels, inception_channels)
        inception_out = inception_channels * 4
        # Each conv stack halves the width (1x2 stride 1x2). After 3 stacks the
        # width becomes input_width / 8. Inception preserves width.
        post_conv_width = max(1, self.input_width // (2 ** 3))
        self.lstm = nn.LSTM(inception_out * post_conv_width, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        if F != self.input_width:
            raise ValueError(
                f"DeepLOB expects flat per-level features of width "
                f"{self.input_width}; got {F}"
            )
        x = x.view(B, 1, L, F)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.inception(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, L, -1)
        h, _ = self.lstm(x)
        return self.head(h)

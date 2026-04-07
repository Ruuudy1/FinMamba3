# Copyright (c) 2024, Tri Dao, Albert Gu.
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    A flexible MLP module that supports standard and gated MLP functionality.

    Args:
        in_features (int): Input feature dimension.
        hidden_features (int, optional): Hidden layer dimension. Defaults to 8/3 of in_features.
        out_features (int, optional): Output feature dimension. Defaults to in_features.
        dropout (float): Dropout probability.
        activation (callable): Activation function. Defaults to F.silu.
        bias (bool): Whether to include bias in the linear layers.
        multiple_of (int): Ensures hidden_features is a multiple of this value.
        use_gate_mlp (bool): Whether to use the gated MLP mechanism.
    """ 
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        dropout=0.1,
        activation='gelu',
        bias=True,
        multiple_of=128,
        use_gate_mlp=False, # If use_gate_mlp is True, split the hidden dimension into two parts for gating
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        fc1_hidden_features = 2 * hidden_features if use_gate_mlp else hidden_features # expand the hidden features if use gate
        self.fc1 = nn.Linear(in_features, fc1_hidden_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self.use_gate_mlp = use_gate_mlp

        # Assert if use_gate_mlp=True, activation should ideally be GELU
        if self.use_gate_mlp:
            assert self.activation == F.gelu, (
                "It is recommended to use GELU activation when use_gate_mlp=True, "
                "as suggested in the original paper."
            )
    def forward(self, x):
        y = self.fc1(x)
        if self.use_gate_mlp:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        else:
            y = self.activation(y)
        y = self.fc2(y)
        # y = self.dropout(y)
        return y
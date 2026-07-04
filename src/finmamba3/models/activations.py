# region imports
from torch import nn
# endregion
# Explicit registry so config-driven activation names resolve without getattr(nn, ...).
ACTIVATION_BY_NAME = {"SiLU": nn.SiLU, "ReLU": nn.ReLU, "GELU": nn.GELU, "Tanh": nn.Tanh, "ELU": nn.ELU}

# region imports
import numpy as np
from torch import nn
# endregion
# Reciprocal of the truncated-normal std correction at the +/-2 sigma bound.
_TRUNC_NORMAL_FACTOR = 0.87962566103423978


def layer_init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _zero_bias(module):
    if module.bias is not None:
        module.bias.data.fill_(0.0)


def _fan_in_out(module):
    if module.__class__ is nn.Linear:
        return module.in_features, module.out_features
    space = module.kernel_size[0] * module.kernel_size[1]
    return space * module.in_channels, space * module.out_channels


def _init_trunc_normal(module):
    fan_in, fan_out = _fan_in_out(module)
    scale = 1.0 / ((fan_in + fan_out) / 2.0)
    std = np.sqrt(scale) / _TRUNC_NORMAL_FACTOR
    nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    _zero_bias(module)


def _init_layernorm(module):
    module.weight.data.fill_(1.0)
    _zero_bias(module)


# Exact-type dispatch; nn.RMSNorm and container modules are absent on purpose (no-op init).
_INIT_BY_TYPE = {
    nn.Linear: _init_trunc_normal,
    nn.Conv2d: _init_trunc_normal,
    nn.ConvTranspose2d: _init_trunc_normal,
    nn.LayerNorm: _init_layernorm,
}


def weight_init(module):
    initializer = _INIT_BY_TYPE.get(module.__class__)
    if initializer is not None:
        initializer(module)

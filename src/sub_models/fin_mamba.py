from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)
_LOGGED_MAMBA_CLASSES: set[str] = set()


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_local_mamba_module(module) -> bool:
    module_file = getattr(module, "__file__", None)
    candidates = []
    if module_file is not None:
        candidates.append(module_file)
    candidates.extend(getattr(module, "__path__", []))
    for candidate in candidates:
        try:
            if Path(candidate).resolve().is_relative_to(_repo_src_dir() / "mamba_ssm"):
                return True
        except (OSError, ValueError):
            continue
    return False


def _load_upstream_mamba_class(module_name: str, class_name: str):
    """Import source-installed mamba_ssm, not the removed/legacy repo copy."""

    for name, module in list(sys.modules.items()):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            if _is_local_mamba_module(module):
                del sys.modules[name]

    src_dir = _repo_src_dir()
    original_path = list(sys.path)

    def _points_to_repo_src(path: str) -> bool:
        candidate = Path.cwd() if path == "" else Path(path)
        try:
            return candidate.resolve() == src_dir
        except OSError:
            return False

    sys.path = [path for path in sys.path if not _points_to_repo_src(path)]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "FinDrama Mamba backbones require the upstream source-installed "
            "`mamba-ssm` package. Install it after CUDA PyTorch with: "
            "`MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir "
            "--force-reinstall git+https://github.com/state-spaces/mamba.git "
            "--no-build-isolation`."
        ) from exc
    finally:
        sys.path = original_path

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            if Path(module_file).resolve().is_relative_to(src_dir / "mamba_ssm"):
                raise ImportError(
                    f"Refusing to use vendored Mamba module at {module_file}; "
                    "install upstream `mamba-ssm` instead."
                )
        except ValueError:
            pass
    # Log resolved class once per process so a fallback path is obvious in logs.
    cache_key = f"{module_name}:{class_name}"
    if cache_key not in _LOGGED_MAMBA_CLASSES:
        _LOGGED_MAMBA_CLASSES.add(cache_key)
        logger.info(
            "[mamba] resolved %s.%s from %s",
            module_name, class_name, module_file or "<unknown>",
        )
        if module_name.endswith("mamba3") and module_file is not None:
            try:
                tilelang_mod = importlib.import_module(
                    "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo"
                )
                logger.info(
                    "[mamba] TileLang Mamba3 MIMO kernel available at %s",
                    getattr(tilelang_mod, "__file__", "<unknown>"),
                )
            except ImportError:
                logger.warning(
                    "[mamba] TileLang Mamba3 MIMO kernel NOT available; "
                    "MIMO will use the slower Python reference path."
                )
    return getattr(module, class_name)


class FinMambaSequenceModel(nn.Module):
    """FinDrama sequence wrapper around upstream Mamba blocks.

    The public API intentionally stays compatible with WorldModel:
    `sequence_model(latent, action)` returns `[B, L, d_model]`.
    """

    def __init__(
        self,
        *,
        stoch_dim: int,
        action_dim: int,
        d_model: int,
        n_layer: int,
        block_type: str,
        dropout_p: float = 0.0,
        ssm_cfg: dict | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.block_type = block_type
        self.dropout_p = dropout_p
        ssm_cfg = dict(ssm_cfg or {})
        factory = {"device": device, "dtype": dtype}

        if block_type == "Mamba3":
            headdim = int(ssm_cfg.get("headdim", 64))
            if d_model % headdim != 0:
                raise ValueError(
                    "Models.WorldModel.HiddenStateDim must be divisible by "
                    f"Models.WorldModel.Mamba3.headdim; got {d_model} and {headdim}."
                )
            block_cls = _load_upstream_mamba_class("mamba_ssm.modules.mamba3", "Mamba3")
        elif block_type == "Mamba2":
            block_cls = _load_upstream_mamba_class("mamba_ssm.modules.mamba2", "Mamba2")
        elif block_type == "Mamba":
            block_cls = _load_upstream_mamba_class("mamba_ssm.modules.mamba_simple", "Mamba")
        else:
            raise ValueError(f"Unknown Mamba block_type: {block_type}")

        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, d_model, bias=True, **factory),
            nn.RMSNorm(d_model, **factory),
            nn.SiLU(),
        )
        self.norms = nn.ModuleList([nn.RMSNorm(d_model, **factory) for _ in range(n_layer)])
        layers = []
        for i in range(n_layer):
            layer_kwargs = {
                "d_model": d_model,
                "layer_idx": i,
                **ssm_cfg,
                **factory,
            }
            if block_type == "Mamba3":
                layer_kwargs["n_layer"] = n_layer
                layer_kwargs["dropout"] = dropout_p
            layers.append(block_cls(**layer_kwargs))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_p)
        self.norm_f = nn.RMSNorm(d_model, **factory)

    def forward(self, samples, action, inference_params=None, **mixer_kwargs):
        action = F.one_hot(action.long(), self.action_dim).to(dtype=samples.dtype)
        hidden_states = self.stem(torch.cat([samples, action], dim=-1))

        # Mamba3 single-token cache/step kernels are intentionally bypassed on
        # T4/A100 compatibility runs; full-prefix recomputation calls this path.
        if self.block_type == "Mamba3":
            inference_params = None

        for norm, layer in zip(self.norms, self.layers):
            residual = hidden_states
            layer_input = norm(hidden_states)
            if inference_params is None:
                layer_out = layer(layer_input, **mixer_kwargs)
            else:
                layer_out = layer(
                    layer_input,
                    inference_params=inference_params,
                    **mixer_kwargs,
                )
            hidden_states = residual + self.dropout(layer_out)
        return self.norm_f(hidden_states)

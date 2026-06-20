# region imports
from __future__ import annotations
import importlib
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from finmamba3.models.regime_modulation import RegimeAux, RegimeFiLMModulator
# endregion
logger = logging.getLogger(__name__)
_LOGGED_MAMBA_CLASSES: list[str] = []


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
        import triton as _triton
        if not hasattr(_triton, "set_allocator"):
            # Pytorch-triton omits triton.set_allocator; mamba3_siso_fwd.py calls it at module level.
            # The no-op stub is safe because pytorch-triton manages TMA descriptor memory internally.
            _triton.set_allocator = lambda fn: None
    except ImportError:
        pass
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "FinMamba3 Mamba backbones require the upstream source-installed "
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
        _LOGGED_MAMBA_CLASSES.append(cache_key)
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
                # The kernel_cache lookup emits a WARNING on every fwd/bwd call (~7 per step at n_layer=4); silencing keeps stdout readable and removes per-step logging overhead.
                logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)
            except ImportError:
                logger.warning(
                    "[mamba] TileLang Mamba3 MIMO kernel NOT available; "
                    "MIMO requires this kernel and will raise at model "
                    "construction without it."
                )
    return getattr(module, class_name)


class FinMambaSequenceModel(nn.Module):
    """FinMamba3 sequence wrapper around upstream Mamba blocks.

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
        use_action_input: bool = True,
        use_regime_film: bool = False,
        num_regimes: int = 8,
        regime_embed_dim: int = 32,
        regime_condition_on_vol: bool = False,
        regime_vol_window: int = 16,
        regime_init_scale: float = 0.0,
        regime_dropout: float = 0.0,
        regime_decouple_router: bool = False,
        regime_apply_after_scan: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.d_model = d_model
        self.block_type = block_type
        self.dropout_p = dropout_p
        # The use_action_input flag gates the one-hot action concat in forward.
        # Phase A pretraining feeds zero actions, so the embedding gradient is degenerate and the path adds noise without signal.
        # Set this flag to False for Phase A runs.
        self.use_action_input = bool(use_action_input)
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
        stem_in = stoch_dim + action_dim if self.use_action_input else stoch_dim
        self.stem = nn.Sequential(
            nn.Linear(stem_in, d_model, bias=True, **factory),
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
        self.use_regime_film = bool(use_regime_film)
        # apply_after_scan moves the FiLM gate from each block's input to its output. An input-side affine
        # folds into the block's own Delta, B and C projections, so it is the gauge direction joint training
        # returns to identity; gating the output past the selective scan is the non-gauge control.
        self.apply_after_scan = bool(regime_apply_after_scan)
        if self.use_regime_film:
            self.regime_modulator = RegimeFiLMModulator(
                hidden_dim=d_model,
                n_layer=n_layer,
                num_regimes=num_regimes,
                embed_dim=regime_embed_dim,
                condition_on_vol=regime_condition_on_vol,
                vol_window=regime_vol_window,
                init_scale=regime_init_scale,
                dropout=regime_dropout,
                decouple_router=regime_decouple_router,
                dtype=dtype,
                device=device,
            )
        else:
            self.regime_modulator = None
    def forward(self, samples, action, inference_params=None, return_regime=False,
                regime_vol=None, **mixer_kwargs):
        if self.use_action_input:
            action = F.one_hot(action.long(), self.action_dim).to(dtype=samples.dtype)
            hidden_states = self.stem(torch.cat([samples, action], dim=-1))
        else:
            hidden_states = self.stem(samples)
        # Infer the latent regime from the stem summary, optionally steered by a volatility feature, and
        # emit per-block FiLM so the regime modulates each block's input and thus its input-dependent
        # Delta, B and C. regime_vol, when supplied, is the realized volatility measured from the
        # observation midprice (shape [B, L, 1]); it gives the router the same axis the supervision label
        # and the eval split use. When omitted the modulator falls back to its caller-free latent proxy.
        regime_logits = None
        gammas = None
        betas = None
        if self.use_regime_film:
            gammas, betas, regime_logits = self.regime_modulator(hidden_states, external_vol=regime_vol)
        # Mamba3 single-token cache and step kernels are intentionally bypassed on T4 and A100 compatibility runs.
        # Full-prefix recomputation calls this path instead.
        if self.block_type == "Mamba3":
            inference_params = None
        for layer_index, (norm, layer) in enumerate(zip(self.norms, self.layers)):
            residual = hidden_states
            layer_input = norm(hidden_states)
            if self.use_regime_film and not self.apply_after_scan:
                # Input-side affine: the gauge direction that folds into the block's input-dependent Delta, B and C.
                layer_input = gammas[:, :, layer_index, :] * layer_input + betas[:, :, layer_index, :]
            if inference_params is None:
                layer_out = layer(layer_input, **mixer_kwargs)
            else:
                layer_out = layer(
                    layer_input,
                    inference_params=inference_params,
                    **mixer_kwargs,
                )
            if self.use_regime_film and self.apply_after_scan:
                # Non-gauge control: gate the block output past the selective scan, on the residual stream,
                # where the next block's RMSNorm cannot fold the modulation back into the host projections.
                layer_out = gammas[:, :, layer_index, :] * layer_out + betas[:, :, layer_index, :]
            hidden_states = residual + self.dropout(layer_out)
        output = self.norm_f(hidden_states)
        if return_regime:
            # gamma_dev and beta_mag report how far FiLM has moved off its identity init; they are
            # zero when FiLM is disabled so the caller can log them unconditionally.
            gamma_dev = (gammas - 1.0).abs().mean() if gammas is not None else output.new_zeros(())
            beta_mag = betas.abs().mean() if betas is not None else output.new_zeros(())
            return output, RegimeAux(regime_logits=regime_logits, gamma_dev=gamma_dev, beta_mag=beta_mag)
        return output

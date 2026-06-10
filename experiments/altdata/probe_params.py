"""Print the total world-model parameter count for a config + CLI overrides (CPU build).

Used to pick matched-parameter depths for the G2 backbone ablation: build each backbone at candidate
depths and read the total, then choose the depth whose total lands nearest the Mamba-3 SISO reference.
Building the nn.Module allocates parameters without running the (CUDA-only) scan kernels, so this is
a cheap CPU probe.
"""
# region imports
import sys
import yaml
import torch
from finmamba3.config import DotDict, parse_args_and_update_config
from finmamba3.models.world_model import WorldModel
# endregion


def main() -> int:
    config_path = sys.argv[1]
    overrides = sys.argv[2:]
    with open(config_path, "r") as handle:
        cfg = DotDict(parse_args_and_update_config(yaml.safe_load(handle), argv=overrides))
    wm = WorldModel(action_dim=1, config=cfg, device=torch.device("cpu"))
    total = sum(parameter.numel() for parameter in wm.parameters())
    seq = sum(parameter.numel() for parameter in wm.sequence_model.parameters())
    print(f"backbone={cfg.Models.WorldModel.Backbone} total_params={total:,} sequence_model_params={seq:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

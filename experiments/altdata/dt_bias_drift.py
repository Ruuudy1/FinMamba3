"""dt_bias drift logging (plan-goal.md Sec. 7.3 item 3): turns Sec. 2.1's corrected algebra (a constant
input-side shift folds its Delta-component into the block's trainable dt_bias exactly) into a measurement.

Runs one FI-2010 predictability-supervised escalation, byte-identical to fi2010_escalation.sh, via
finmamba3.train.main() unmodified. A single monkeypatch wraps train_step.train_world_model_step (the
call is late-bound by name inside finmamba3.train, so patching the attribute on the train module takes
effect without touching any file under src/) to snapshot every Mamba3 block's dt_bias tensor after each
step; behavior is unchanged since the wrapper always delegates to and returns the original function's
result. Reports the L2 drift of each block's dt_bias from its step-0 value alongside film_g, so the
claimed absorption shows up as dt_bias moving while the run's other host parameters are unlogged here
(they are not the claim).
"""
# region imports
import json
import sys
import numpy as np
import torch
# endregion
LOG_EVERY = 25


def _dt_bias_tensors(world_model):
    return {name: param for name, param in world_model.named_parameters() if name.endswith("dt_bias")}


def main():
    import finmamba3.train as train_module
    from finmamba3.train_step import train_world_model_step as original_step

    drift_by_step: dict[int, dict[str, float]] = {}
    norm_by_step: dict[int, dict[str, float]] = {}
    init_by_name: dict[str, torch.Tensor] = {}

    def instrumented_step(*args, **kwargs):
        world_model = kwargs["world_model"]
        global_step = kwargs["global_step"]
        result = original_step(*args, **kwargs)
        if global_step == 0 or global_step % LOG_EVERY == 0:
            tensors = _dt_bias_tensors(world_model)
            if not init_by_name:
                for name, param in tensors.items():
                    init_by_name[name] = param.detach().clone()
            drift_by_step[global_step] = {
                name: float(torch.linalg.norm(param.detach() - init_by_name[name]).item())
                for name, param in tensors.items()
            }
            norm_by_step[global_step] = {name: float(torch.linalg.norm(param.detach()).item()) for name, param in tensors.items()}
        return result

    train_module.train_world_model_step = instrumented_step
    sys.argv = [
        "finmamba3.train",
        "--config", "configs/fi2010_studentt.yaml",
        "--data-train", "data/fi2010/train",
        "--data-val", "data/fi2010/validation",
        "--dataset", "fi2010",
        "--BasicSettings.Seed", "0",
        "--Models.WorldModel.RegimeFiLM.Enabled", "True",
        "--Models.WorldModel.RegimeFiLM.InitScale", "0.1",
        "--Models.WorldModel.RegimeFiLM.LRMult", "50",
        "--Models.WorldModel.RegimeFiLM.SuperviseVol", "True",
        "--Models.WorldModel.RegimeFiLM.SuperviseAxis", "predictability",
        "--Models.WorldModel.RegimeFiLM.SupervisionWeight", "30",
        "--Models.WorldModel.RegimeFiLM.FeedObsVol", "True",
        "--Models.WorldModel.RegimeFiLM.DecoupleRouterFromFiLM", "True",
        "--Models.WorldModel.RegimeFiLM.EntropyCoef", "0.0",
        "--norm-path", "saved_models/lob/fi2010_dtbias_norm_s0.json",
        "--JointTrainAgent.SampleMaxSteps", "3000",
    ]
    train_module.main()

    block_names = sorted({name for step_drift in drift_by_step.values() for name in step_drift})
    summary = {}
    for name in block_names:
        steps = sorted(drift_by_step)
        drift_series = [drift_by_step[s][name] for s in steps]
        norm_series = [norm_by_step[s][name] for s in steps]
        summary[name] = {
            "steps": steps, "drift": drift_series, "norm": norm_series,
            "final_drift": drift_series[-1], "final_norm": norm_series[-1],
            "relative_drift": drift_series[-1] / max(norm_series[0], 1e-8),
        }
        print(
            f"[dt_bias] {name}: init_norm={norm_series[0]:.5f} final_norm={norm_series[-1]:.5f} "
            f"final_drift={drift_series[-1]:.5f} (relative={summary[name]['relative_drift']:.2f}x init norm)"
        )
    json.dump(summary, open("reports/dt_bias_drift.json", "w"), indent=2)
    print("wrote reports/dt_bias_drift.json")


if __name__ == "__main__":
    main()

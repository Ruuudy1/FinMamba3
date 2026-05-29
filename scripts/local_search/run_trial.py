"""Single-trial driver for the local 4080 hyperparameter search loop.

Reads `runs/local_search/<trial_id>/config_overrides.json`, invokes
`python -m finmamba3.train` with the fixed 4080 kernel shape plus the
trial-specific dotted overrides, captures stdout, parses the wandb offline
files written by the run, and emits a structured `metrics_final.json` plus a
single appended line in `runs/local_search/trials.jsonl`.
"""
# region imports
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
# endregion


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
TRIAL_ROOT = REPO_ROOT / "runs" / "local_search"
WINDOWS_CUDA_HOME = Path(__file__).resolve().parent / "cuda_win"
WSL_FAST_ROOT = Path("/home/ruuudy/finmamba3_fast")
TILELANG_TMP = WSL_FAST_ROOT / "tilelang_tmp"
LOCAL_TARGET_STEP = int(os.environ.get("LOCAL_SEARCH_TARGET_STEP", "1000"))
SAMPLE_MAX_STEPS = LOCAL_TARGET_STEP + 1
SAVE_EVERY_STEPS = max(LOCAL_TARGET_STEP // 2, 1)
LOG_EVERY_STEPS = int(os.environ.get("LOCAL_SEARCH_LOG_EVERY_STEPS", "25"))
PROGRESS_STEP_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[")
IMPORTANT_LOG_MARKERS = (
    "Traceback",
    "Error",
    "Exception",
    "RuntimeError",
    "ValueError",
    "Killed",
    "validation top feature MSE",
    "new best val_loss=",
    "checkpoint",
    "wandb:",
)
KERNEL_OVERRIDES_BY_KEY = {
    "Models.WorldModel.Mamba3.d_state": "64",
    "Models.WorldModel.Mamba3.chunk_size": "8",
    "Models.WorldModel.Mamba3.is_mimo": "true",
}
RECIPE_OVERRIDES_BY_KEY = {
    "JointTrainAgent.BatchSize": "768",
    "JointTrainAgent.AccumSteps": "1",
    "JointTrainAgent.SampleMaxSteps": str(SAMPLE_MAX_STEPS),
    "JointTrainAgent.SaveEverySteps": str(SAVE_EVERY_STEPS),
    "JointTrainAgent.EarlyStoppingPatience": "3",
    "JointTrainAgent.EarlyStopMetric": "Val/yes_log_loss",
    "JointTrainAgent.SaveBestOnly": "true",
}


def resolve_overrides(trial_id):
    overrides_path = TRIAL_ROOT / trial_id / "config_overrides.json"
    if not overrides_path.exists():
        raise FileNotFoundError(f"No config_overrides.json at {overrides_path}")
    return json.loads(overrides_path.read_text())


def build_command(trial_overrides_by_key):
    config_path = REPO_ROOT / "configs" / "lob.yaml"
    command_parts = [
        sys.executable, "-m", "finmamba3.train",
        "--config", str(config_path),
        "--dataset", "polymarket",
        "--hours-train", "1.0",
        "--hours-val", "0.25",
    ]
    for dotted_key, override_value in KERNEL_OVERRIDES_BY_KEY.items():
        command_parts.append(f"--{dotted_key}")
        command_parts.append(override_value)
    for dotted_key, override_value in RECIPE_OVERRIDES_BY_KEY.items():
        command_parts.append(f"--{dotted_key}")
        command_parts.append(override_value)
    for dotted_key, override_value in trial_overrides_by_key.items():
        command_parts.append(f"--{dotted_key}")
        command_parts.append(str(override_value))
    return command_parts


def latest_summary_path(wandb_dir):
    candidates = offline_run_file_candidates(wandb_dir, "wandb-summary.json")
    if not candidates:
        return None
    return candidates[-1]


def latest_history_path(wandb_dir):
    candidates = offline_run_file_candidates(wandb_dir, "wandb-history.jsonl")
    if not candidates:
        return None
    return candidates[-1]


def offline_run_file_candidates(wandb_dir, filename):
    direct_candidates = sorted(wandb_dir.glob(f"offline-run-*/files/{filename}"))
    nested_candidates = sorted((wandb_dir / "wandb").glob(f"offline-run-*/files/{filename}"))
    return sorted(direct_candidates + nested_candidates)


def offline_run_wandb_candidates(wandb_dir):
    direct_candidates = sorted(wandb_dir.glob("offline-run-*/run-*.wandb"))
    nested_candidates = sorted((wandb_dir / "wandb").glob("offline-run-*/run-*.wandb"))
    return sorted(direct_candidates + nested_candidates)


def wandb_record_key(record_item):
    if record_item.nested_key:
        return "/".join(record_item.nested_key)
    return record_item.key


def parse_wandb_value(value_json):
    value = json.loads(value_json)
    if value is None:
        return None
    return value


def parse_wandb_binary_metric_history(wandb_dir, metric_key):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    value_by_step = {}
    for wandb_path in offline_run_wandb_candidates(wandb_dir):
        store = DataStore()
        store.open_for_scan(str(wandb_path))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            step = int(record.history.step.num)
            value = None
            for item in record.history.item:
                key = wandb_record_key(item)
                parsed_value = parse_wandb_value(item.value_json)
                if key == "_step" and parsed_value is not None:
                    step = int(parsed_value)
                if key == metric_key:
                    value = parsed_value
            if value is not None:
                value_by_step[step] = float(value)
        store.close()
    return value_by_step


def parse_wandb_binary_summary(wandb_dir, metric_key):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    value = None
    for wandb_path in offline_run_wandb_candidates(wandb_dir):
        store = DataStore()
        store.open_for_scan(str(wandb_path))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "summary":
                continue
            for item in record.summary.update:
                key = wandb_record_key(item)
                if key == metric_key:
                    value = parse_wandb_value(item.value_json)
        store.close()
    if value is None:
        return None
    return float(value)


def parse_metric_history(wandb_dir, history_path, metric_key):
    value_by_step = {}
    if history_path is None or not history_path.exists():
        return parse_wandb_binary_metric_history(wandb_dir, metric_key)
    for line in history_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if metric_key in row and row[metric_key] is not None:
                step = row.get("_step", 0)
                value_by_step[step] = float(row[metric_key])
    if not value_by_step:
        value_by_step = parse_wandb_binary_metric_history(wandb_dir, metric_key)
    return value_by_step


def best_step_and_value(value_by_step):
    if not value_by_step:
        return None, None
    best_step, best_value = min(
        value_by_step.items(),
        key=lambda step_value_pair: step_value_pair[1],
    )
    return best_step, best_value


def final_metric(wandb_dir, summary_path, metric_key):
    if summary_path is None or not summary_path.exists():
        return parse_wandb_binary_summary(wandb_dir, metric_key)
    summary = json.loads(summary_path.read_text())
    value = summary.get(metric_key)
    if value is None:
        return parse_wandb_binary_summary(wandb_dir, metric_key)
    return float(value)


def collect_metrics(trial_id, trial_overrides_by_key, wandb_dir,
                    wall_clock_sec, returncode, stdout_log_path):
    summary_path = latest_summary_path(wandb_dir)
    history_path = latest_history_path(wandb_dir)
    brier_by_step = parse_metric_history(wandb_dir, history_path, "Val/yes_brier")
    log_loss_by_step = parse_metric_history(wandb_dir, history_path, "Val/yes_log_loss")
    recon_by_step = parse_metric_history(wandb_dir, history_path, "Val/reconstruction_loss")
    best_brier_step, best_brier = best_step_and_value(brier_by_step)
    best_log_loss_step, best_log_loss = best_step_and_value(log_loss_by_step)
    best_recon_step, best_recon = best_step_and_value(recon_by_step)
    crash_tail = ""
    if returncode != 0 and stdout_log_path.exists():
        log_lines = stdout_log_path.read_text(encoding="utf-8").splitlines()
        crash_tail = "\n".join(log_lines[-40:])
    return {
        "trial_id": trial_id,
        "completed": returncode == 0,
        "returncode": returncode,
        "wall_clock_sec": round(wall_clock_sec, 1),
        "config_overrides": trial_overrides_by_key,
        "best_step_brier": best_brier_step,
        "best_val_yes_brier": best_brier,
        "best_step_log_loss": best_log_loss_step,
        "best_val_yes_log_loss": best_log_loss,
        "best_step_recon": best_recon_step,
        "best_val_reconstruction_loss": best_recon,
        "final_val_yes_brier": final_metric(wandb_dir, summary_path, "Val/yes_brier"),
        "final_val_yes_log_loss": final_metric(wandb_dir, summary_path, "Val/yes_log_loss"),
        "final_val_reconstruction_loss": final_metric(wandb_dir, summary_path, "Val/reconstruction_loss"),
        "final_val_mid_direction_accuracy": final_metric(wandb_dir, summary_path, "Val/mid_direction_accuracy"),
        "crash_tail": crash_tail,
    }


def progress_step(line):
    matches = PROGRESS_STEP_RE.findall(line)
    if not matches:
        return None
    return int(matches[-1])


def should_write_stdout_line(line, logged_progress_steps):
    stripped_line = line.strip()
    if not stripped_line:
        return False
    for marker in IMPORTANT_LOG_MARKERS:
        if marker in stripped_line:
            return True
    step = progress_step(stripped_line)
    if step is None:
        return True
    if step == 1 or step % LOG_EVERY_STEPS == 0:
        if step in logged_progress_steps:
            return False
        logged_progress_steps.add(step)
        return True
    return False


def stream_filtered_stdout(process, stdout_file):
    logged_progress_steps = set()
    for raw_line in process.stdout:
        for line in re.split(r"[\r\n]+", raw_line):
            if should_write_stdout_line(line, logged_progress_steps):
                stdout_file.write(line.rstrip() + "\n")
                stdout_file.flush()
    return process.wait()


def run_training_process(command_parts, subprocess_env, stdout_path):
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        process = subprocess.Popen(
            command_parts,
            cwd=str(REPO_ROOT),
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        return stream_filtered_stdout(process, stdout_file)


def write_metrics_file(trial_id, metrics_by_key):
    metrics_path = TRIAL_ROOT / trial_id / "metrics_final.json"
    metrics_path.write_text(json.dumps(metrics_by_key, indent=2))


def append_to_trials_jsonl(metrics_by_key):
    trials_path = TRIAL_ROOT / "trials.jsonl"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("a", encoding="utf-8") as trials_file:
        trials_file.write(json.dumps(metrics_by_key) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    trial_dir = TRIAL_ROOT / args.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_overrides_by_key = resolve_overrides(args.trial_id)
    command_parts = build_command(trial_overrides_by_key)
    if args.dry_run:
        print("DRY RUN")
        print(f"  command:  {' '.join(command_parts)}")
        print(f"  trial dir: {trial_dir}")
        return 0
    fast_trial_dir = WSL_FAST_ROOT / args.trial_id
    fast_trial_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = fast_trial_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    subprocess_env = dict(os.environ)
    existing_pythonpath = subprocess_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        subprocess_env["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), existing_pythonpath])
    else:
        subprocess_env["PYTHONPATH"] = str(SRC_ROOT)
    if (WINDOWS_CUDA_HOME / "bin" / "nvcc").exists():
        TILELANG_TMP.mkdir(parents=True, exist_ok=True)
        subprocess_env["CUDA_HOME"] = str(WINDOWS_CUDA_HOME)
        subprocess_env["CUDA_PATH"] = str(WINDOWS_CUDA_HOME)
        subprocess_env["TMPDIR"] = str(TILELANG_TMP)
        existing_path = subprocess_env.get("PATH", "")
        if existing_path:
            subprocess_env["PATH"] = os.pathsep.join([str(WINDOWS_CUDA_HOME / "bin"), existing_path])
        else:
            subprocess_env["PATH"] = str(WINDOWS_CUDA_HOME / "bin")
    subprocess_env["WANDB_DIR"] = str(wandb_dir)
    subprocess_env["WANDB_MODE"] = "offline"
    stdout_path = trial_dir / "stdout.log"
    start_time = time.time()
    returncode = run_training_process(command_parts, subprocess_env, stdout_path)
    wall_clock_sec = time.time() - start_time
    metrics_by_key = collect_metrics(
        args.trial_id, trial_overrides_by_key, wandb_dir,
        wall_clock_sec, returncode, stdout_path,
    )
    write_metrics_file(args.trial_id, metrics_by_key)
    append_to_trials_jsonl(metrics_by_key)
    status_word = "completed" if metrics_by_key["completed"] else "failed"
    print(f"trial {args.trial_id} {status_word}")
    print(f"  best Brier:     {metrics_by_key['best_val_yes_brier']}")
    print(f"  best LogLoss:   {metrics_by_key['best_val_yes_log_loss']}")
    print(f"  wall_clock_sec: {metrics_by_key['wall_clock_sec']}")
    return 0 if metrics_by_key["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())

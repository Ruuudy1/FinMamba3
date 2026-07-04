"""Build notebooks/colab_mimo_ablation.ipynb — a one-click A100 Colab notebook that trains the Mamba-3 MIMO
backbone on FI-2010 + Kaggle BTC, evaluates recon NLL + direction macro-F1, and prints the MIMO ablation row.
Reuses the proven dependency cell from colab_lob_pretrain.ipynb.
"""
# region imports
import json
# endregion
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})


md("""# FinMamba3 — Mamba-3 MIMO backbone ablation (A100, one-click)

Trains the **Mamba-3 MIMO** backbone on FI-2010 and Kaggle BTC, evaluates held-out reconstruction NLL +
next-tick direction macro-F1, and prints the **MIMO row** for the G2 backbone-ablation table — the one cell the
RTX 4080 cannot run (its warp-tiling-valid kernel configs exceed the ~100 KB SMEM cap; see RESULTS.md §5.1).

**Setup (one time):**
1. Runtime → Change runtime type → **A100 GPU**.
2. Runtime → **Run all**.

(The `sj-hryi/FinMamba3` dataset repo is public — **no HF token is required to download data**. Set an `HF_TOKEN`
Colab secret to enable checkpoint sync + **resume**: the run uploads to `sj-hryi/FinMamba3-checkpoints` every 200
steps, and if Colab disconnects, just **Run all** again — it pulls the latest checkpoint per dataset and warm-restarts
from where it left off, training only the remaining steps to 3000. So a disconnect costs ≤200 steps, not the whole run.)

MIMO on A100 (sm_80) uses `chunk_size=8` at the comparable `d_state=128` — an identical model (chunk_size is only a
kernel tiling parameter), baked into `configs/{fi2010,kaggle}_mimo.yaml`. It is slow (~9 s/it), so this defaults to a
3000-step budget matching the 4080 non-MIMO ablation for a fair comparison. On H100/H200 set `chunk_size: 16` in those
configs for ~2× throughput.""")
code('''import os, sys, subprocess, glob
from pathlib import Path

REPO_URL = "https://github.com/Ruuudy1/FinMamba3.git"
BRANCH = "a100"

# MIMO ablation datasets (each mirrors the 4080 non-MIMO ablation setup).
DATASETS = ["fi2010", "kaggle"]
KAGGLE_ASSET = "BTC"
KAGGLE_RESOLUTION = "1min"
MAX_STEPS = 3000          # matches the 4080 non-MIMO ablation step budget
SEED = 0
SMOKE_TEST = True         # run a 5-step plumbing check first (verifies the MIMO kernel builds/runs in ~1 min)
UPLOAD_EVERY = 200        # sync the checkpoint to HF every N steps (crash-safety); needs HF_TOKEN write access
RESUME = True             # on a fresh session, pull the latest HF checkpoint for each dataset and warm-restart
                          # from it, training only the remaining steps to MAX_STEPS. Survives Colab disconnects.
                          # Needs HF_TOKEN read access. Note: --resume-checkpoint resets the LR schedule, so a
                          # resumed run's LR re-warms over the remaining steps (a robustness tradeoff, not exact).

HF_REPO = "sj-hryi/FinMamba3"
CKPT_REPO = "sj-hryi/FinMamba3-checkpoints"   # where periodic + final checkpoints are uploaded
WORK_ROOT = Path.home() / "finmamba3_mimo"
PROJECT_DIR = str(WORK_ROOT / "FinMamba3")
CACHE_ROOT = WORK_ROOT / "cache"
DATA_ROOT = WORK_ROOT / "hf_data"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

# Per-dataset config + eval settings (the kaggle loader reads data/<ASSET>_<RES>.csv; the kaggle config
# carries Asset/Resolution/Hours; eval threshold mirrors the 4080 ablation: FI-2010 0.0, Kaggle 0.01).
# ckpt_path_in_repo: per-dataset HF folder for this run's checkpoint sync (keeps datasets from colliding).
# resume_from: HF folders to scan when resuming, newest-step wins. fi2010 also lists the legacy "checkpoints/lob"
# folder so the FIRST run of this notebook bootstraps off the cached pre-resume run that synced there.
PER_DATASET = {
    "fi2010": dict(config="configs/fi2010_mimo.yaml", data_train="data/fi2010/train",
                   data_val="data/fi2010/validation", threshold=0.0, ckpt_root="saved_models/lob/LOB",
                   ckpt_path_in_repo="mimo-ckpt/fi2010", resume_from=["mimo-ckpt/fi2010", "checkpoints/lob"]),
    "kaggle": dict(config="configs/kaggle_mimo.yaml", data_train="data", data_val="data",
                   threshold=0.01, ckpt_root="saved_models/kaggle/LOB",
                   ckpt_path_in_repo="mimo-ckpt/kaggle", resume_from=["mimo-ckpt/kaggle"]),
}


def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


def get_hf_token():
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


print("MIMO ablation:", DATASETS, "| steps", MAX_STEPS, "| seed", SEED)''')
code('''# The sj-hryi/FinMamba3 dataset repo is public (ungated), so the HF token is OPTIONAL -- it is used
# only if you happen to have one set (e.g. for higher rate limits). No token is required to download.
HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

pip_install("huggingface_hub")
from huggingface_hub import snapshot_download

DATA_ROOT.mkdir(parents=True, exist_ok=True)
patterns = []
if "fi2010" in DATASETS:
    patterns += ["data/fi2010/train/Train_Dst_NoAuction_DecPre_CF_7.txt",
                 "data/fi2010/validation/Val_Dst_NoAuction_DecPre_CF_7.txt"]
if "kaggle" in DATASETS:
    patterns += [f"data/kaggle/{KAGGLE_ASSET}_{KAGGLE_RESOLUTION}.csv"]
snapshot_download(repo_id=HF_REPO, repo_type="dataset", allow_patterns=patterns,
                  token=HF_TOKEN, local_dir=str(DATA_ROOT))
print("Downloaded from HF:", patterns)''')
code('''import shutil
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
subprocess.check_call(["git", "clone", "--branch", BRANCH, REPO_URL, PROJECT_DIR])
os.chdir(PROJECT_DIR)
print("Repo ready:", os.getcwd())''')
# Dependency cell, copied verbatim from colab_lob_pretrain.ipynb (the proven A100/H100 setup).
code('''os.chdir(PROJECT_DIR)

pip_cache = CACHE_ROOT / 'pip'
pip_cache.mkdir(parents=True, exist_ok=True)
os.environ['PIP_CACHE_DIR'] = str(pip_cache)

pip_install('--upgrade', 'pip')
pip_install('huggingface_hub[cli]')
pip_install('packaging', 'ninja', 'setuptools==69.5.1', 'numpy>=2,<3')

import typing_extensions as _typing_ext
from typing_extensions import TypeGuard as _TypeIs_shim
_typing_ext.TypeIs = _TypeIs_shim
del _typing_ext, _TypeIs_shim

pip_install('torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0',
            '--index-url', 'https://download.pytorch.org/whl/cu126')

import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
else:
    major, minor = 9, 0
os.environ['TORCH_CUDA_ARCH_LIST'] = f"{major}.{minor}"
print(f"TORCH_CUDA_ARCH_LIST: {major}.{minor}")

cxx11_abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
torch_minor = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
cuda_major = (torch.version.cuda or '12.4').split('.')[0]
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

CAUSAL_CONV1D_VERSION = '1.6.2.post1'
MAMBA_SSM_VERSION = '2.3.2.post1'
causal_conv1d_url = (
    f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v{CAUSAL_CONV1D_VERSION}/"
    f"causal_conv1d-{CAUSAL_CONV1D_VERSION}+cu{cuda_major}torch{torch_minor}cxx11abi{cxx11_abi}-{py_tag}-{py_tag}-linux_x86_64.whl"
)
mamba_ssm_url = (
    f"https://github.com/state-spaces/mamba/releases/download/v{MAMBA_SSM_VERSION}/"
    f"mamba_ssm-{MAMBA_SSM_VERSION}+cu{cuda_major}torch{torch_minor}cxx11abi{cxx11_abi}-{py_tag}-{py_tag}-linux_x86_64.whl"
)
pip_install('--force-reinstall', '--no-deps', causal_conv1d_url)
pip_install('--force-reinstall', '--no-deps', mamba_ssm_url)
pip_install('-e', '.')
pip_install('tilelang==0.1.8')
# tilelang 0.1.8 pulls apache-tvm-ffi 0.1.11 whose TVM bindings crash (NestedLoopChecker); pin to 0.1.9.
pip_install('apache-tvm-ffi==0.1.9', '--force-reinstall', '--no-deps')
pip_install('quack-kernels')
pip_install('transformers')
for _attempt in range(5):
    _result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--timeout', '300', 'triton>=3.5.0'])
    if _result.returncode == 0:
        break
if _result.returncode:
    raise subprocess.CalledProcessError(_result.returncode, _result.args)

import torch, numpy as np
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

import triton
if 'set_allocator' not in triton.__dict__:
    triton.set_allocator = lambda fn: None

import causal_conv1d_cuda
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
print('Mamba3 + MIMO TileLang import OK:', mamba3_mimo)''')
code('''# A100/H100 GUARD -- run this BEFORE training. The MIMO bwd_bwd TileLang kernel needs ~123 KB of dynamic
# shared memory per block. sm_80 (A100, 164 KB cap) and sm_90 (H100, 227 KB) fit it; sm_89 (L4 / RTX 4080,
# ~99 KB cap) CANNOT and dies with "Failed to set the allowed dynamic shared memory size to 123216".
# Fail fast here instead of after a ~3 min kernel compile.
import torch
cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}  sm_{cc[0]}{cc[1]}")
if cc not in [(8, 0), (9, 0)]:
    raise RuntimeError(
        f"{name} (sm_{cc[0]}{cc[1]}) cannot run the Mamba-3 MIMO kernel: its ~123 KB/block shared-memory need "
        f"exceeds the sm_89 cap (~99 KB). Switch the Colab runtime to an A100 (Runtime -> Change runtime type "
        f"-> A100 GPU; requires Colab Pro+), then Run all again. An H100 also works.")
print("GPU OK for Mamba-3 MIMO.")''')
code('''# Stage the HF-downloaded data into the repo's data/ dir (FI-2010 split files + Kaggle CSV).
project = Path(PROJECT_DIR)
data_dir = project / "data"
data_dir.mkdir(exist_ok=True)
if "fi2010" in DATASETS:
    for split, fn in [("train", "Train_Dst_NoAuction_DecPre_CF_7.txt"),
                      ("validation", "Val_Dst_NoAuction_DecPre_CF_7.txt")]:
        src = DATA_ROOT / "data" / "fi2010" / split / fn
        dst = data_dir / "fi2010" / split / fn
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)
    print("FI-2010 ready under", data_dir / "fi2010")
if "kaggle" in DATASETS:
    src = DATA_ROOT / "data" / "kaggle" / f"{KAGGLE_ASSET}_{KAGGLE_RESOLUTION}.csv"
    dst = data_dir / f"{KAGGLE_ASSET}_{KAGGLE_RESOLUTION}.csv"
    if not dst.exists():
        shutil.copy(src, dst)
    print("Kaggle ready:", dst.name)''')
code('''import torch
from huggingface_hub import snapshot_download


def latest_ckpt(root):
    paths = glob.glob(f"{root}/*/ckpt/world_model_final.pth")
    return max(paths, key=os.path.getmtime) if paths else None


def stream(cmd):
    # Stream output LIVE. Colab hides subprocess output under check_call (esp. tqdm), so we read it
    # line-by-line and re-print with flush -- you see per-step loss / progress in real time.
    print("Running:", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def find_resume(cfg):
    # Scan this dataset's HF checkpoint folders and return (local_path, step) for the most-trained
    # checkpoint, else (None, 0). Lets a fresh Colab session warm-restart from the last HF-synced
    # checkpoint instead of starting over after a disconnect. Needs RESUME and an HF read token.
    if not (RESUME and HF_TOKEN):
        return None, 0
    best_path, best_step = None, -1
    dst = str(WORK_ROOT / "resume_ckpts")
    for prefix in cfg["resume_from"]:
        try:
            local = snapshot_download(repo_id=CKPT_REPO, repo_type="model", token=HF_TOKEN,
                                      allow_patterns=[f"{prefix}/**/world_model*.pth"], local_dir=dst)
        except Exception as exc:
            print(f"  resume: nothing under {prefix} ({exc})", flush=True)
            continue
        for path in glob.glob(f"{local}/{prefix}/**/world_model*.pth", recursive=True):
            try:
                step = int(torch.load(path, map_location="cpu").get("step", 0))
            except Exception:
                continue
            if step > best_step:
                best_path, best_step = path, step
    if best_path:
        print(f"  resume: warm-restart from {best_path} (trained to step {best_step})", flush=True)
    return best_path, max(best_step, 0)


rows = {}
for ds in DATASETS:
    cfg = PER_DATASET[ds]
    norm = f"saved_models/lob/{ds}_mimo_norm.json"
    base = [sys.executable, "-u", "-m", "finmamba3.train", "--config", cfg["config"],
            "--data-train", cfg["data_train"], "--data-val", cfg["data_val"], "--dataset", ds,
            "--BasicSettings.Seed", SEED]
    resume_path, done_step = find_resume(cfg)
    remaining = MAX_STEPS - done_step
    # Plumbing check: build + run the MIMO TileLang kernel for 5 steps (~1 min once compiled). If THIS
    # produces no step output, the kernel build is the issue -- interrupt and report it (don't burn hours).
    if SMOKE_TEST:
        print(f"\\n--- SMOKE TEST: {ds} MIMO, 5 steps (verifies the kernel builds + runs) ---", flush=True)
        stream(base + ["--JointTrainAgent.SampleMaxSteps", 5, "--JointTrainAgent.SaveModels", "False",
                       "--norm-path", f"saved_models/lob/{ds}_smoke_norm.json"])
        print(f"SMOKE TEST PASSED for {ds}\\n", flush=True)
    if remaining > 0:
        tag = f" (resuming from step {done_step})" if resume_path else ""
        print(f"\\n{'='*64}\\nTRAIN Mamba-3 MIMO on {ds}: {remaining} of {MAX_STEPS} steps{tag}\\n{'='*64}", flush=True)
        train_cmd = base + ["--JointTrainAgent.SampleMaxSteps", remaining, "--norm-path", norm]
        if resume_path:   # warm-restart weights + optimizer; LR schedule re-warms over the remaining steps
            train_cmd += ["--resume-checkpoint", resume_path]
        if HF_TOKEN:      # crash-safety: sync to this dataset's own HF folder every UPLOAD_EVERY steps
            train_cmd += ["--ckpt-repo", CKPT_REPO, "--ckpt-upload-every", UPLOAD_EVERY,
                          "--ckpt-path-in-repo", cfg["ckpt_path_in_repo"]]
            print(f"checkpoints -> {CKPT_REPO}/{cfg['ckpt_path_in_repo']} every {UPLOAD_EVERY} steps", flush=True)
        else:
            print("no HF_TOKEN set: checkpoints stay local + no resume (set the HF_TOKEN Colab secret)", flush=True)
        stream(train_cmd)
        ckpt = latest_ckpt(cfg["ckpt_root"])
    else:
        print(f"\\n{ds}: already trained to step {done_step} >= {MAX_STEPS}; skipping to eval", flush=True)
        ckpt = resume_path
    print("checkpoint:", ckpt, flush=True)
    print(f"\\n{'='*64}\\nEVAL Mamba-3 MIMO on {ds}\\n{'='*64}", flush=True)
    out_md = f"reports/mimo_{ds}.md"
    stream([sys.executable, "-m", "finmamba3.eval.eval_backbone_metrics", "--config", cfg["config"],
            "--dataset", ds, "--checkpoint", ckpt, "--data-val", cfg["data_val"], "--norm-path", norm,
            "--is-mimo", "--threshold", cfg["threshold"], "--windows", 512, "--out", out_md])
    rows[ds] = Path(out_md).read_text()
    print(rows[ds], flush=True)''')
code('''print("\\n" + "=" * 64)
print("Mamba-3 MIMO ablation rows (recon NLL + direction macro-F1 + params)")
print("=" * 64)
for ds, table in rows.items():
    print(f"\\n--- {ds} ---\\n{table}")
print("\\nSlot the MIMO recon-NLL into the ablation tables (RESULTS.md §5.1 and finmamba3-paper.tex tab:backbone_ablation),")
print("then compare against the best non-MIMO backbone under matched params:")
print("  FI-2010 best non-MIMO recon NLL = -0.5403 (Mamba-1);  Kaggle best = -0.0213 (Mamba-2).")
print("Pre-registered: if MIMO does NOT match/exceed the best alternative, the MIMO-advantage claim fails — report either way.")''')
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebooks/colab_mimo_ablation.ipynb", "w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)
print(f"wrote notebooks/colab_mimo_ablation.ipynb ({len(cells)} cells)")

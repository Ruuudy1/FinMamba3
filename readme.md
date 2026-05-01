# FinDrama

FinDrama is the Polymarket limit-order-book version of Drama. The old Atari and
MemoryMaze paths have been removed; this repository now supports one workflow:
offline world-model pretraining on Polymarket LOB data, plus the first
Gymnasium trading environment for Phase B.

## Run On Colab

Use exactly one notebook:

`notebooks/colab_lob_pretrain.ipynb`

The notebook works on any CUDA GPU (H100, L4, A100, T4). H100 is recommended.

In Colab:

1. Set the runtime to a GPU instance (Runtime, Change runtime type).
2. Open `notebooks/colab_lob_pretrain.ipynb`.
3. Confirm `REPO_URL = "https://github.com/Ruuudy1/FinDrama.git"` and
   `BRANCH = "dev-mamba3-mimo"` in the first code cell.
4. Add your `HF_TOKEN` to Colab Secrets (key icon, left sidebar). The token
   needs write access to upload compiled wheels.
5. Run cells top to bottom.
6. Start with `SMOKE_TEST = True`. After one short update and validation print
   complete, set it to `False` for the full 20,000-step run.

The notebook installs PyTorch 2.6 CUDA 12.4, builds `causal-conv1d` and
`mamba-ssm` from source, downloads the data bundle automatically, and runs:

```bash
python -B src/train_lob.py --hours-train 6 --hours-val 1 --JointTrainAgent.SampleMaxSteps 20000
```

Checkpoints are written under:

```text
saved_models/lob/LOB/<run_id>/ckpt/world_model.pth
```

The final notebook cell uploads `saved_models/lob` to the HuggingFace dataset
repo at `ruuudy/FinDrama` under `checkpoints/lob/`.

## Wheel Cache

Compiled CUDA wheels (`causal-conv1d`, `mamba-ssm`) are cached on HuggingFace
so later runtimes skip the source build:

https://huggingface.co/datasets/ruuudy/FinDrama/tree/main

Wheels are keyed by Python version, PyTorch version, CUDA version, and GPU
architecture (for example `wheels-py312-torch260-cu124-sm90`). The first run
on a new GPU type builds and uploads; subsequent runs pull and install in
seconds.

Set `FORCE_REBUILD_WHEELS = True` in the first notebook cell to force a fresh
build after updating the dependency stack or if a cached wheel becomes stale.

## Data

The pretraining dataset is a public Google Drive folder:

https://drive.google.com/drive/u/0/folders/1fInfOLCJ9SAfRbghC67k1ppK_B5_Ucxz

The notebook downloads it automatically via `gdown`. No manual step is needed.
Leave `DATA_ZIP = ""` in the first cell unless you are providing a local zip.

The folder contains `train.tar.zip` and `validation.tar.zip`. The notebook
extracts both splits into `data/train` and `data/validation`.

## Local Smoke Test

Install a CUDA PyTorch build first, then install the project dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy>=2,<3" "causal-conv1d>=1.4.0" --no-build-isolation
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall git+https://github.com/state-spaces/mamba.git --no-build-isolation
pip install -r requirements.txt
```

Then run:

```bash
python -B src/train_lob.py --hours-train 1 --hours-val 0.25 --JointTrainAgent.SampleMaxSteps 20
```

Expected data layout:

```text
data/
  train/
    polymarket.db
    polymarket_books/
    binance_lob/
  validation/
    polymarket.db
    polymarket_books/
    binance_lob/
```

## Repository Layout

```text
notebooks/
  colab_lob_pretrain.ipynb
src/
  train_lob.py                    LOB pretraining entrypoint
  config_utils.py                 dotted CLI config overrides
  training_steps.py               world-model update loop
  replay_buffer.py
  agents.py                       actor-critic/PPO code for Phase B
  config_files/configure_lob.yaml
  envs/
    lob_features.py               robust LOB feature engineering
    polymarket_lob_env.py         Gymnasium trading environment
  lob/backtester/                 vendored DATAHACKS data structures + loader
  sub_models/
    lob_encoder.py                Transformer-over-depth-tokens encoder
    lob_auxiliary.py              regime/memory experiment scaffolding
    fin_mamba.py                  FinDrama sequence wrapper for upstream Mamba
    world_models.py               Mamba3 MIMO world model
tests/
  test_lob_features.py
  test_polymarket_lob_env.py
```

## Notes

- `train_lob.py` no longer imports `gym` or the removed Atari path.
- Normalized LOB features are clipped and checked before training.
- `Backbone: Mamba3` is the default. Full-sequence Phase A pretraining is the
  supported path; Phase B imagination uses full-prefix recomputation rather
  than Mamba3 step/inference-cache kernels.
- If a local run fails while importing `mamba_ssm.modules.mamba3`, the
  upstream source install is missing or was built against a different PyTorch
  CUDA wheel.

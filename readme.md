# FinDrama

FinDrama is the Polymarket limit-order-book version of Drama. The old Atari and
MemoryMaze paths have been removed; this repository now supports one workflow:
offline world-model pretraining on Polymarket LOB data, plus the first
Gymnasium trading environment for Phase B.

## Run On Colab

Use exactly one notebook:

`notebooks/colab_lob_pretrain.ipynb`

In Colab:

1. Runtime -> Change runtime type -> A100 GPU.
2. Open `notebooks/colab_lob_pretrain.ipynb`.
3. Confirm `REPO_URL = "https://github.com/Ruuudy1/FinDrama.git"` and
   `BRANCH = "master"` in the first code cell, and optionally set `DATA_ZIP`.
4. Run cells top to bottom.
5. Start with `SMOKE_TEST = True`; after one short update and validation print
   complete, set it back to `False` for the full `20_000` step run.

The notebook installs PyTorch 2.6 CUDA 12.4, builds `causal-conv1d`, source
installs upstream Mamba3, extracts the data bundle into `data/train` and
`data/validation`, then runs:

```bash
python -B src/train_lob.py --hours-train 6 --hours-val 1 --JointTrainAgent.SampleMaxSteps 20000
```

Checkpoints are written under:

```text
saved_models/lob/LOB/<run_id>/ckpt/world_model.pth
```

The final notebook cell copies `saved_models/lob` to Google Drive.

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
  supported T4/A100 path; Phase B imagination uses full-prefix recomputation
  rather than Mamba3 step/inference-cache kernels.
- If the local short run fails while importing `mamba_ssm.modules.mamba3`, the
  upstream source install is missing or was built against a different PyTorch
  CUDA wheel.

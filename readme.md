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
5. Start with `SMOKE_TEST = True`; after it reaches model construction, set it
   back to `False` for the full `20_000` step run.

The notebook installs PyTorch CUDA 12.1, builds `causal-conv1d` and
`mamba-ssm`, extracts the data bundle into `data/train` and `data/validation`,
then runs:

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
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d==1.2.0.post2 --no-build-isolation
pip install mamba-ssm==1.2.0.post1 --no-build-isolation
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
    world_models.py               Mamba/Mamba2 world model
tests/
  test_lob_features.py
  test_polymarket_lob_env.py
```

## Notes

- `train_lob.py` no longer imports `gym` or the removed Atari path.
- Normalized LOB features are clipped and checked before training.
- If the local short run fails with `No module named selective_scan_cuda`, the
  Mamba CUDA extension is not installed for the active Python environment. Use
  the Colab notebook or reinstall `causal-conv1d` and `mamba-ssm` after
  installing CUDA PyTorch.

# FinDrama: Drama Adapted for Polymarket Limit Order Books

Fork of [Drama](https://openreview.net/forum?id=7XIkRgYjK3) (a Mamba/Mamba-2 model-based RL agent) adapted to Polymarket limit-order-book (LOB) data. The image CNN is swapped for a Transformer over depth-curve tokens, and features are HFT-style aggregate statistics (order-flow imbalance, microprice, depth shares) rather than raw top-K prices.

Upstream citation:
```
@inproceedings{
    wang2025drama,
    title={Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient},
    author={Wenlong Wang and Ivana dusparic and Yucheng Shi and Ke Zhang and Vinny Cahill},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=7XIkRgYjK3}
}
```

## Repo layout

```
src/
  config_files/
    configure.yaml          Atari (unchanged upstream path)
    configure_lob.yaml      Polymarket LOB pretraining
  envs/
    my_atari.py             Atari env wrapper
    my_memory_maze.py       MemoryMaze env wrapper
    lob_features.py         Per-tick LOB feature engineering (K=10 depth tokens + 14 aggregate scalars)
  lob/backtester/           Vendored Polymarket data loader (build_timeline -> TickData)
  sub_models/
    world_models.py         WorldModel (CNN or LOB encoder via Encoder.Type flag)
    lob_encoder.py          Transformer encoder + MLP decoder for LOB features
    ...
  replay_buffer.py          ObsMode={image, features}
  train.py                  Atari training entry point
  train_lob.py              LOB world-model pretraining entry point
  eval.py                   Atari evaluation
data/                       (gitignored) extracted LOB data
notebooks/
  lob_pretrain_check.ipynb  Diagnostic plots for LOB pretraining
drama_colab_A100_Train.ipynb  Atari training on Colab
drama_colab_A100_Eval.ipynb   Atari evaluation on Colab
drama_colab_A100_LOB.ipynb    LOB pretraining on Colab
```

## LOB pretraining (primary path)

### Quick start on Google Colab

1. Open `drama_colab_A100_LOB.ipynb` in Colab.
2. Runtime -> Change runtime type -> A100 GPU (requires Colab Pro).
3. Keep the data zip (`drive-download-20260423T214219Z-3-001.zip`) anywhere under `/content/drive/MyDrive/`. Cell 2 auto-discovers it.
4. Push your branch to your fork and update `BRANCH` in cell 3.
5. Run cells 1-9 in order. Checkpoints + imagined rollouts land in `src/saved_models/lob/LOB/<run_id>/`; run cell 10 in a second tab to back them up to Drive every 10 minutes.
6. Open `notebooks/lob_pretrain_check.ipynb` to render diagnostic plots against a saved checkpoint.

### Data

Polymarket LOB bundle (train + validation tars) expected at `data/train/` and `data/validation/` with the layout
```
data/<split>/
  polymarket.db                 SQLite of market_prices / rtds_prices / market_outcomes
  polymarket_books/orderbooks.csv
  binance_lob/{btc,eth,sol}usdt.parquet
```
The Colab notebook handles extraction from the shared zip automatically.

### Local training

```
cd src
python train_lob.py \
    --data-train ../data/train \
    --data-val   ../data/validation \
    --hours-train 6 --hours-val 1
```

Feature dimensions are fixed by `src/envs/lob_features.py`:
- `K_LEVELS = 10` depth-curve tokens per tick
- `F_LEVEL = 8` per-token features (relative price offset, log size, cumulative depth, vol share, gap, level index, side flag, book staleness)
- `F_TICK = 14` per-tick scalars (mid, spread, log-spread, imbalance, microprice, weighted-mid displacement, log bid/ask volumes, delta mid/spread/imbalance, top OFI, trade intensity, rolling vol)

Total flat observation dim: `K*F_level + F_tick = 94`.

### GPU optimization

Key knobs in `configure_lob.yaml`:

| Setting | Default | Effect |
|---------|---------|--------|
| `BatchSize` | 64 | Mini-batch per accumulation step. Increase until OOM. |
| `AccumSteps` | 2 | Gradient accumulation steps; effective batch = `BatchSize × AccumSteps`. |
| `Use_amp` | True | bfloat16 mixed precision (always on; GradScaler disabled for bfloat16). |
| `Compile` | False | Set to `True` on Colab/Linux for 10–20% extra throughput (gated `os.name != 'nt'`). |

GPU utilization is logged every 30 s via `nvidia-smi` (`[GPU] util=…%  mem=…/… MiB`).

**Loading a checkpoint** (new dict format):
```python
ckpt = torch.load("saved_models/lob/LOB/<run_id>/ckpt/world_model.pth", weights_only=False)
world_model.load_state_dict(ckpt["world_model"])
world_model.optimizer.load_state_dict(ckpt["optimizer"])
# resume from ckpt["step"]
```

## Atari path (upstream)

Original Drama behaviour is preserved. With `Encoder.Type: cnn` (the default in `configure.yaml`) the existing CNN encoder, image decoder, and joint-train loop work unchanged.

### Colab evaluation of a pretrained Atari checkpoint

Pretrained Pong weights are available at [drama_checkpoints](https://drive.google.com/drive/folders/1jeoY7pMU8brPPiYDHzu3TOzhXSMdkAJy?usp=sharing). Upload `world_model.pth` and `agent.pth` to `MyDrive/drama_checkpoints/`, then open `drama_colab_A100_Eval.ipynb` on a GPU runtime and run cells 1-8.

### Colab training from scratch

`drama_colab_A100_Train.ipynb` on an A100 runtime. ~16 hours for 100k steps on `ALE/Pong-v5`.

### Local

```
conda create --name drama python=3.10
conda activate drama
pip install torch==2.2.1
pip install -r requirements.txt
cd src
python train.py
```

Configuration overrides via CLI: `python train.py --Models.WorldModel.Backbone Mamba`. The same `--Group.Key value` pattern works for `train_lob.py`.

## Docker

```
docker build -f Dockerfile -t drama .
docker run --gpus all -it --rm drama
```
Container starts inside `src/`. See the upstream Drama issue on Docker performance under WSL2.

## Code references

- [Mamba / Mamba-2](https://github.com/state-spaces/mamba)
- [STORM](https://github.com/weipu-zhang/STORM)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [DATAHACKS2026 backtester](https://github.com/Ruuudy1/DATAHACKS2026) (vendored data loader under `src/lob/backtester/`)

# Setup

## Environment
- Python with `torch` (CUDA build), `mamba_ssm`, and this package installed editable: `pip install -e .`
- GPU training was run on an RTX 4080 via WSL. The Polymarket MIMO headline config exceeds the 4080's shared-memory
  cap and segfaults, so all 4080 runs use the non-MIMO Mamba-3 path (`--Models.WorldModel.Mamba3.is_mimo False`).
  The MIMO headline model needs an A100+.

## Tests
```
python -m pytest tests/
```

## Datasets
- **FI-2010** — public benchmark, ingested via `src/finmamba3/envs/fi2010_loader.py`; direction comparison harness in
  `src/finmamba3/eval/benchmark_fi2010.py`; regime-gap evaluator in
  `src/finmamba3/eval/eval_regime_generalization_fi2010.py`.
- **Polymarket** — binary-outcome LOB data under `data/` (gitignored). Loaded via `src/finmamba3/backtester/`.

## PnL backtester (vendored execution engine)
`src/finmamba3/eval/pnl_backtest.py` judges a frozen world model by simulated trading PnL. The order-matching +
settlement engine it runs on is **vendored** here as `src/finmamba3/backtester/engine/` (a pure CPU engine with no
torch dependency) and consumes this repo's timeline dataclasses directly, so no external checkout or environment
variable is needed:

```
python -m finmamba3.eval.pnl_backtest --config configs/lob_spot.yaml --checkpoint <final.pth> ...
```

The engine exposes the `BaseStrategy` / `Order` interface that `WorldModelStrategy` implements; the dependency
direction is one-way (the engine never imports model or eval code).

## Reproducing the headline result
See `RESULTS.md` §3 for the exact train and backtest commands, the frozen thresholds, and the deterministic
(`--deterministic-latent`) headline.

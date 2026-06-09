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

## PnL backtester dependency (decoupled, not vendored)
`src/finmamba3/eval/pnl_backtest.py` judges a frozen world model by simulated trading PnL. It depends on the external
DATAHACKS2026 execution engine (a pure order-matching + settlement engine with no torch dependency), which is **not
vendored** here. Point the adapter at a local checkout with the `DATAHACKS2026_PATH` environment variable:

```
export DATAHACKS2026_PATH=/path/to/DATAHACKS2026
python -m finmamba3.eval.pnl_backtest --config configs/lob_spot.yaml --checkpoint <final.pth> ...
```

If `DATAHACKS2026_PATH` is unset, the adapter falls back to a repo-local `tmppolymarket-bot/` checkout. The engine
exposes the `BaseStrategy` / `Order` interface that `WorldModelStrategy` implements; the dependency direction is
one-way (this repo depends on the engine's interface, never the reverse).

## Reproducing the headline result
See `RESULTS.md` §3 for the exact train and backtest commands, the frozen thresholds, and the deterministic
(`--deterministic-latent`) headline.

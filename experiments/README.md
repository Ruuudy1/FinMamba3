# experiments/

Reproduction scaffolding, **not** part of the packaged `finmamba3` library. These shell
scripts and ad-hoc Python drivers reproduce the campaign runs (cross-dataset FiLM null,
backbone ablation, gauge/in-kernel/LM-FiLM demonstrations, vintage reconciliation). They
are untested and not held to the house code style, and they hard-code paths and run ids
for specific machines.

- `altdata/` — cross-dataset (Kaggle crypto LOB + FI-2010) FiLM-null and backbone-ablation runs,
  plus the vintage-reconciliation drivers. See `altdata/README.md`.
- `inkernel/` — the in-kernel vs input-affine gauge-absorption boundary demonstration.
- `lm_film/` — the Mamba character-LM FiLM control (regime-decodability bound on the null).

Treat anything here as a recipe to read and adapt, not a supported API. The packaged
entrypoints are `finmamba3-train`, `finmamba3-backtest`, and the `python -m finmamba3.*`
modules documented in the top-level `readme.md`.

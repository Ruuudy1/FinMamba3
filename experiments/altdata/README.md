# Alternative-data campaign — experiment scripts (reproducibility)

Exact orchestration for the cross-dataset regime-FiLM null + backbone ablation (driver
`alternative-data-goal.md`; results in `RESULTS.md` §5/§5.1, `fi2010-film.md`, `kaggle-film.md`).

**Environment.** GPU work runs on the RTX 4080 via WSL: `wsl.exe bash -c "cd /mnt/host/c/Users/ruuud/algoverse/Drama
&& <cmd>"`. The `*.sh` files `cd` into the repo themselves; strip CR first (`sed -i 's/\r$//' <script>`) when invoking
from a Windows-checked-out tree. Data: `data/<ASSET>_<RES>.csv` (extracted from `data/archive.zip`, gitignored) and
`data/fi2010/{train,validation}`. Checkpoints land in `saved_models/{lob,kaggle}/LOB/<run_id>/ckpt/world_model_final.pth`
(gitignored); each script captures the run-specific paths into a `reports/*_manifest.txt`.

## Phase 1 — G1 FiLM A/B (predictability primary, spot-vol secondary)

| script | what it does | output |
|---|---|---|
| `fi2010_ab.sh 3000` | FI-2010 baseline+treatment × seeds 0,1 (`fi2010_studentt.yaml`) | `reports/fi2010_ab_manifest.txt` |
| `kaggle_ab.sh 3000` | Kaggle BTC baseline+treatment × seeds 0,1 (`kaggle_btc.yaml`) | `reports/kaggle_ab_manifest.txt` |
| `eval_ab.sh <tag> <config> <data-val> <win> <thr>` | the 8 gap evals (direction_macro_f1 + recon_nll × predictability + spot_vol × 2 seeds) | `reports/<tag>_ab_gaps.txt` + per-eval `.md` |
| `gap_stats.py <label> <g0> <g1> ...` | aggregate gaps → mean ± sd, SE, t, p, 95% CI (host, scipy) | stdout |
| `kaggle_escalation.sh 3000 0` | predictability-supervised strongest-null escalation (Kaggle) | `reports/kaggle_escalation_s0.log` + film_g trajectory |
| `run_kaggle_phase1.sh` | chains Kaggle `eval_ab` + `kaggle_escalation` | — |
| `robustness.sh 3000` | FI-2010 predictability escalation + ETH/ADA cross-asset A/B (seed 0) + evals | `reports/kaggle_crossasset_*` |
| `btc5min.sh 3000` / `btc1sec.sh 3000` | resolution-robustness A/B (`kaggle_btc_5min.yaml` / `_1sec.yaml`) | `reports/kaggle_btc{5min,1sec}_*` |

Eval invocations: `eval_ab.sh fi2010 configs/fi2010_studentt.yaml data/fi2010/validation 512 0.0` and
`eval_ab.sh kaggle configs/kaggle_btc.yaml data 128 0.01`.

## Phase 2 — G2 backbone ablation

| script | what it does | output |
|---|---|---|
| `probe_params.py <config> <overrides>` | total/sequence-model param count for a backbone (CPU build) — used to set matched depths | stdout |
| `probe_batch.sh` | sweeps Mamba/Mamba2/Transformer depths to match the 10.28M Mamba-3 SISO budget | stdout |
| `phase2_ablation.sh <tag> <basecfg> <data-train> <data-val> <steps> <thr> <seed> <reuse_m3> <reuse_norm>` | trains Mamba-1/2/Transformer (`{tag}_{mamba1,mamba2,transformer}.yaml`), reuses the Mamba-3 SISO baseline; evals recon NLL + direction F1 via `eval_backbone_metrics` | `reports/<tag>_backbone_ablation.txt` |
| `run_phase2.sh` | chains both datasets' ablations + the FI-2010 DeepLOB/majority/linear-AR benchmark | — |

`run_phase2.sh` hard-codes this run's FiLM-off baseline checkpoint IDs (`fqv6n5ls` fi2010, `yc3rhs5n` kaggle) as the
reused Mamba-3 SISO cell; on a fresh reproduction re-run `fi2010_ab.sh`/`kaggle_ab.sh` first and substitute the new
baseline `world_model_final.pth` paths from `reports/<tag>_ab_manifest.txt`.

Mamba-3 MIMO is **not** run (the 4080's ~100 KB/block SMEM cap core-dumps the TileLang kernel at the first forward;
verified — see `RESULTS.md` §5.1) — it is the one A100-flagged headline cell.

## Mechanism / diagnostics (reuse existing checkpoints; no training)

| script | question | output |
|---|---|---|
| `plot_film_g.py` | the headline figure: film_g decay + reg_H trajectories of both escalations | `reports/film_g_escalation_decay.png` |
| `plot_film_g_settings.py` | summary figure: treatment film_g inert (~0.004) across all 10 A/B settings | `reports/film_g_across_settings.png` |
| `decay_fit.py` | exponential fit of the escalation film_g decay — asymptote (≤0 ⇒ to identity), time-constant, R² | stdout |
| `pe_hurst_corroboration.py` | does the ER split separate forecastable from random-walk windows (PE/Hurst)? | stdout |
| `frozen_capacity_probe.py --bucket {predictability,vol} --vol-window N` | is the regime linearly decodable from frozen baseline features? (vol vs ER, window-length sweep) | stdout |
| `probe_windows.sh` / `probe_vol_vs_er.sh` | the decodability sweeps that produced the capacity numbers | stdout |
| `finmamba3.eval.diag_router --dataset kaggle --supervise-axis predictability --feed-obs-vol ...` | does the (supervised) joint router commit per-step / agree with the regime bucket? | stdout |

## Class-balanced direction control (de-degenerate the primary metric)

The world-model next-tick head collapses to the flat majority (degenerate macro-F1), so a class-balanced re-run
confirms the gap stays null with a non-degenerate metric.

| script | what it does | output |
|---|---|---|
| `fi2010_cb_ab.sh 3000` / `kaggle_cb_ab.sh 3000` | baseline+treatment seeds 0,1 with `Direction.ClassBalanced=True`; eval direction macro-F1 both axes | `reports/{fi2010,kaggle}_cb_manifest.txt` |
| `kaggle_cb_seed2.sh 3000` | protocol third seed for the borderline Kaggle spot-vol gap (`add 2 if borderline`) | stdout |
| `cb_recon_eval.sh` | recon-NLL (secondary metric) on the CB checkpoints — confirms it is null too | stdout |

## Paper

These draft LaTeX subsections are **already integrated** into `finmamba3-paper.tex` (G1 as paragraphs + table
inside `\subsection{Regime-FiLM is a decisive null}`; G2 as the new `\subsection{Backbone ablation under matched
parameters}`, `sec:results_ablation`), verified well-formed. The standalone files are kept as the source/provenance
for the author to revise; final numbers, draft prose, paper notation:
- `paper_draft_cross_dataset.tex` — extends `\subsection{Regime-FiLM is a decisive null}` with the cross-dataset
  evidence (G1): the null table, the optimiser-driven decay-to-identity (exponential fit), the decodable-yet-unused
  capacity argument, the class-balanced control, and the real-gap-FiLM-fails-to-close motivation.
- `paper_draft_backbone_ablation.tex` — a `\subsection{Backbone ablation under matched parameters}` (G2): the
  matched-budget recon-NLL table + DeepLOB/linear-AR benchmark + the structural MIMO-infeasibility argument.
- `paper_fig.py` — generates the paper figure `figures/film_g_escalation_decay.pdf` (title-less, vector, LaTeX-math
  labels), integrated into the paper as `Fig.~\ref{fig:escalation_decay}` in `sec:results_null`.

## Headline result

Regime-FiLM is a decisive, dataset/asset/resolution-independent null: the optimizer drives the in-scan modulation to
identity (`film_g`→~0, `reg_H`=`ln R`) even when forced active and ER-supervised, and the null holds even on the
volatility axis the representation encodes best — it is the modulation collapsing, not a missing regime signal. Mamba-3
SISO is competitive-not-dominant under matched parameters; the MIMO headline cell awaits an A100.

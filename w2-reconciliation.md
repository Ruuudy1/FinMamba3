# W2 — Settlement/EV reconciliation on one pinned vintage (2026-06-20)

Goal: every settlement/EV number in `finmamba3-paper.tex` (Tables 7–10 + abstract + conclusion)
computed on ONE reproducible vintage = **current `data/validation`**, so nothing disagrees. The drift
the helper flagged (recorded +$3,946 vs re-run ~+$5,157) is real and *consequential*, not rounding.

## Canonical recipe (verbatim from `edge-architecture.md`, verified)

Command (per arm), no `--calibration-temperature` (default 1.0) and no `--intervals`:
```
python3 -m finmamba3.eval.pnl_backtest --config <cfg> --checkpoint <ckpt> \
  --data-val data/validation --norm-path <norm> --assets BTC \
  --predictability-gate --predictability-threshold 0.60 --edge-threshold 0.03 \
  --deterministic-latent --hours-val 9999 [--prob-source edge] [stress flags] --out <json>
```
| arm | config | checkpoint (run id) | norm |
|---|---|---|---|
| settlement s0 | `lob_spot.yaml` | `5rkphy4i` | `norm_spot_seed0_base.json` |
| edge_residual s0 | `lob_edge_residual.yaml` | `61kb278p` | `norm_edge_residual_s0.json` |
| edge_residual s1 | `lob_edge_residual.yaml` | `1lukxyyv` | `norm_edge_residual_s0.json` |
| edge_residual s2 | `lob_edge_residual.yaml` | `9qk67y37` | `norm_edge_residual_s0.json` |
| edge_full s0 | `lob_edge_full.yaml` | `8nti99zj` | `norm_edge_full_s0.json` |

Stress (edge_residual s0): slippage `--slippage-per-share 0.01`; depth band `--min-book-depth 60000
--max-book-depth 130000`; TTE `--min-tte-frac 0.25`. Driver: `experiments/altdata/recon_w2.sh`.

Market count on current data = **645** (high_pred 322 + low_pred 323), so the "646" scare was a
mis-count; the count is stable. The *PnL values* drifted because the validation `polymarket.db` tick
data itself changed.

**Provenance (HF):** the canonical dataset is `sj-hryi/FinMamba3` (HF), whose `data/` folder was
updated ~10 days ago; local `data/validation` reflects that release. The recorded paper numbers
(+$3,946 etc.) predate the update. Therefore the **pinned vintage = current `data/validation` (= recent
HF release)**, and the re-run numbers here are the correct, reproducible canonical values; the older
figures were a pre-update vintage. This is the vintage every table is being reconciled to.

## Finding 1 — settlement drift (DONE, `reports/recon_settlement_s0.json`)

| metric | recorded (paper) | current re-run |
|---|---:|---:|
| total | +$3,946 | **+$5,147** |
| high_pred | +$4,772 (P=1.0) | +$5,096 (P=1.0, dd95=0.021, 322 mkts) |
| low_pred | **−$826 (P=0.276, dd95=0.257)** | **+$51 (P=0.63, dd95=0.185, 323 mkts)** |
| hi market-block CI | [+$2,686,+$7,951] / 129 | **[+$2,445,+$7,747] / 127, P≈1.0** |

**Consequence (important):** on current data the settlement head's low-predictability failure is *much
milder* — P(profit) 0.276→0.63, low_pred PnL −$826→+$51. The paper's "settlement **catastrophically**
fails low-pred; the EV head rescues it (+41%, 12× regime collapse)" framing is **not reproduced** on the
current vintage. The EV head's advantage must be re-measured and the narrative re-scoped honestly to
whatever the current edge_residual re-run shows. This fits the paper's direction (the economic edge is a
COLM *coda*, reported honestly), so the coda gets re-scoped, not inflated.

## Operational note — run backtests SEQUENTIALLY (OOM)

WSL has 15 GiB RAM. Each `pnl_backtest`/`simple_baseline` builds a full Polymarket timeline (the val
build alone ≈ 641k ticks / 2833 markets); `simple_baseline` holds BOTH train and val timelines at once.
Running the edge batch concurrently with the logistic baseline exhausted RAM and the OOM killer stopped
the edge batch mid-arm. Lesson: **never run two timeline-building jobs at once** — one process at a time.

## Finding 2 — figure overlay (DONE)

`experiments/altdata/paper_fig.py` extended to overlay the post-scan (output-side) escalation curves
(dotted) on `figures/film_g_escalation_decay.pdf`; they track the input-affine curves exactly, the visual
of the overdetermined null. Caption updated. Paper recompiles (23 pp, 0 errors).

## Finding 3 — EV arms (DONE, current vintage) — THE NARRATIVE SHIFTED

All on 645 markets (hi 322 / lo 323), ER gate 0.60, edge thr 0.03, deterministic. P = iid bootstrap.

| arm | total | hi | lo | P(lo) | dd95(lo) | regime Δ (hi−lo) |
|---|---:|---:|---:|---:|---:|---:|
| Settlement | +$5,147 | +$5,096 | **+$51** | **0.63** | 0.185 | 5,044 |
| EV Huber s0 | +$5,549 | +$3,182 | +$2,368 | 0.99 | 0.096 | **814** |
| EV Huber s1 | +$4,415 | +$2,828 | +$1,587 | 0.96 | 0.117 | 1,240 |
| EV Huber s2 | +$4,372 | +$2,376 | +$1,995 | 0.99 | 0.099 | 381 |
| **EV Huber mean** | **+$4,779** | +$2,795 | +$1,984 | 0.98 | 0.104 | 812 |
| EV+actionCE s0 | +$4,102 | +$3,996 | +$106 | 0.58 | 0.204 | 3,889 |
| stress slip 1¢ | — | — | +$2,833 | 0.98 | 0.106 | — |
| stress depth | — | — | +$1,048 | 0.92 | 0.110 | — |
| stress TTE≥.25 | — | — | +$956 | 0.90 | 0.128 | — |

**The headline changed and the paper was re-scoped honestly:**
- Settlement low-pred is **NO LONGER catastrophic** (was −$826/P=0.276 → now **+$51/P=0.63**, weak but net-positive).
- "EV lifts total PnL **+41%**" is **FALSE** now: EV mean total +$4,779 is *below* settlement +$5,147 (settlement's
  high-pred dominates). Seed-0 EV (+$5,549) only marginally beats settlement total.
- The EV head's real, honest value is **regime robustness**: low-pred P 0.63→0.99, low-pred PnL +$51→+$1,984 (mean),
  regime spread collapse **6.2×** (5,044→814) — not 12×. "Trades a little headline profit for genuine regime robustness."
- edge_full (action CE) still hurts low-pred (P=0.58) — finding holds.
- Stress tests still clear the (now-milder) settlement baseline comfortably.

Updated in paper: Tables 8/9/10 (body+captions), sec:results_ev title+intro, sec:results_mech limitation,
contributions (3), conclusion, tab:edge & tab:baseline settlement/EV rows, abstract+sec:pnl market-block CI.

## Finding 4 — logistic/GBT baseline RE-FIT on current vintage (DONE, `reports/recon_simple_baseline.json`)

The full re-fit had timed out twice (1500s/3000s): `LogisticRegression(max_iter=2000)` + GBT on **1.59M
training ticks** exceeds this box's budget. Tractability lever (least-invasive): **stride-subsample the
supervised training ticks** via a new `--train-stride` flag in `simple_baseline.py` that keeps every K-th tick
over the full concatenated, time-ordered series (unbiased over the whole period, no truncation). K=4 →
**397,324 of 1.59M ticks** (108 features), which makes the sklearn fit tractable. A second fix was needed:
`del bt_train` (+ `gc.collect()`) before the val timeline is built — holding both timelines at once OOMs the
15 GiB WSL box (the first stride run was SIGKILLed mid-val-build at anon-rss 15.2 GiB; the prior *timeouts*
had died earlier, inside the slow fit, so never reached this peak). Full run: **897s, RC=0**. Driver:
`experiments/altdata/recon_logistic_currentvintage.sh`.

| arm | metric | recorded (was in paper) | current re-fit (stride-4) |
|---|---|---:|---:|
| logistic | total | +$7,016 (Brier 0.146) | **+$6,914 (Brier 0.146)** |
| logistic | total mkt-block | [+$2,928,+$11,008]/243, P=0.9996 | **[+$2,844,+$10,880]/243, P=0.9994** |
| logistic | high_pred | +$2,398, P=0.994 | **+$2,448, P=0.995** |
| logistic | low_pred | +$4,619, P=0.994 | **+$4,466, P=0.993** |
| gbt | total | +$876, P=0.691 | **+$1,575, P=0.809** |
| gbt | low_pred | −$705, P=0.338 | **−$112, P=0.451** |
| naive | total | −$1,008 | **−$1,008** (bit-identical) |

The logistic reproduces almost exactly (+$6,914 vs +$7,016, −1.5%; Brier identical; naive bit-identical), so the
stride-4 subsample is not a distortion and the deflation is not a vintage artifact.

**Decision: REPLACE — the re-fit is strictly better for COLM.** It meets every decision-gate "better" criterion
and none of the "keep" criteria:
- The deflation AND the "reproduces and exceeds" framing both HOLD on one vintage: current logistic **+$6,914 >
  current settlement +$5,147 > EV mean +$4,779** — no softening needed.
- Removes the awkward mixed-vintage caveat: all of `tab:baseline` (logistic/GBT/settlement/EV/naive) is now the
  current vintage. The methodology note ("stride-4 subsample, 397k of 1.59M ticks, uniform over the period") is
  cleaner and more defensible than "kept stale data because the fit timed out."
- Architecture-independence is unweakened (logistic above both heads); regime-robustness holds (logistic low_pred
  +$4,466 at P=0.993 vs settlement low_pred +$51 at P=0.63); the GBT nuance holds (GBT low_pred still net-negative
  −$112 at P=0.451<0.5, i.e. "the model class matched to the near-linear settlement structure wins, not any simple
  model").

**Paper updated:** `tab:baseline` (logistic/GBT rows + methodology-note caption), `fig:baseline` (regenerated
`figures/simple_baseline_pnl.pdf` from the current-vintage JSON; WM reference lines in `make_paper_figures.py`
bumped 3946/4643 → 5147/4779; caption CI + vintage), the regime-split paragraph, the results prose, and the
conclusion (+$7,016 → +$6,914). `tab:calib` is unchanged — it is the settlement-HEAD reliability diagnostic (not
the simple-model baseline), already labeled recorded-vintage/structural. pdflatex: 23 pp, 0 errors, 0 undefined
refs; grep confirms no stale baseline numbers remain.

## W2 STATUS: COMPLETE

Final paper compiles clean (23 pp, 0 errors, 0 undefined refs); zero stale economic numbers remain (verified by
grep). Settlement/EV/seeds/stress + both market-block CIs on the current pinned vintage; logistic/GBT deflation
now ALSO on the current vintage (stride-4 re-fit, Finding 4); 408-market deterministic anchor
(tab:edge/conjunction) kept and labeled.

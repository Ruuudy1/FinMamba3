# Goal: Re-fit the logistic/GBT baseline on the current data vintage (finish W2's Table 7)

Run this with `/goal` on the **`logistic-refit-current-vintage`** branch.

## DECISION GATE (read first) — do not regress the paper
This is an *optional improvement*. The current paper already handles the simple-model baseline
honestly: logistic/GBT/naive are on the **recorded** vintage with explicit caveats, and the
deflationary conclusion is vintage-robust (recorded logistic +$7,016 > current settlement +$5,147).

**Only replace the current approach if the re-fit is genuinely BETTER for the COLM submission.**
After you have the new numbers, judge whether they improve the paper:
- BETTER (swap it in) if: the current-vintage logistic still clearly supports the deflation
  ("a simple model matches/beats the world model"), it removes the awkward mixed-vintage caveat,
  and the required methodology note (subsample/cap) is clean and defensible.
- NOT better (KEEP the current recorded-vintage version, discard the re-run, do not edit the paper)
  if: the capped/subsampled fit muddies the story, the logistic now lands below the world-model
  heads in a way that weakens the architecture-independence point, or the methodology caveat is
  uglier/harder to defend than the current "recorded vintage" caveat.

If you keep the current version, still commit your findings (the measured current-vintage numbers +
the rationale) to this branch and `w2-reconciliation.md` so the decision is documented — just leave
`finmamba3-paper.tex` unchanged. Either way, **do not touch `main`**; the user reviews/merges.

## Context
The FinMamba3 COLM paper (`finmamba3-paper.tex`) had all settlement/EV numbers reconciled to the
**current** `data/validation` vintage (the canonical HF `sj-hryi/FinMamba3` release; see
`w2-reconciliation.md`). The one exception is the **simple-model baseline** (Table 7,
`\label{tab:baseline}`): the logistic regression, gradient-boosted tree, and fair-naive rows are
still on the **recorded** vintage, because `simple_baseline.py`'s full re-fit on ~1.59M training
ticks (`data/train`, all hours) times out (>50 min twice on the 4080/WSL box) —
`LogisticRegression(max_iter=2000)` + `HistGradientBoostingClassifier(max_iter=300)` on that many
samples is the bottleneck.

Recorded values currently in the paper (carry explicit "recorded vintage" caveats):
- Table 7: logistic **+$7,016** (Brier 0.146), GBT **+$876** (0.149), naive **−$1,008**.
- Regime-split paragraph (sec:results_baseline): logistic **+$2,398** high-pred / **+$4,619**
  low-pred, both market-block P=0.994.
- `fig:baseline` (`figures/simple_baseline_pnl.pdf`): logistic market-block CI [+$2,928, +$11k] / 243 mkts.
- Settlement/EV (already current): settlement **+$5,147**; EV mean **+$4,779**; settlement low-pred
  is now weak not catastrophic (**+$51, P=0.63**).

## Make the fit tractable (pick the least-invasive lever; MEASURE before any long run)
The bottleneck is the sklearn fit on ~1.59M training samples. Options, in order of preference:
1. **Stride-subsample training ticks** (unbiased over time): edit `simple_baseline.py` to keep every
   K-th supervised training tick (K chosen to land ~300–500k samples). Preserves the full time range;
   document K and the resulting sample count. **Recommended.**
2. **Cap `--hours-train`** to a window yielding ~300–500k ticks (no code change, but biases toward
   one end of the period — less preferred).
3. Faster solver / lower `max_iter` (e.g. `saga`, `max_iter=300`, with a convergence check) — only if
   (1)/(2) are insufficient.

**Always do a quick timing probe first** (tiny cap → confirm it finishes in a few minutes → scale up)
so you don't burn another 50-min timeout. Target: full run completes in <20 min.

## Run (memory-safe: NEVER run two timeline-building jobs concurrently — 15 GiB WSL OOMs)
```
wsl.exe bash -c "cd /mnt/host/c/Users/ruuud/algoverse/Drama && timeout 1500 python3 -m finmamba3.eval.simple_baseline \
  --config configs/lob_spot.yaml --data-train data/train --data-val data/validation \
  --norm-path saved_models/lob/norm_spot_seed0_base.json --assets BTC \
  <tractability flag> --out reports/recon_simple_baseline.json 2>&1 | tee reports/recon_simple_baseline.log | grep -a '\[baseline\]'"
```
`sklearn`/`scipy` must be pip-installed in WSL (verify). Extract from
`reports/recon_simple_baseline.json`: per-arm `total/high_pred/low_pred` `pnl`,
`bootstrap_market.{frac_profitable,ci_low,ci_high,n_markets}`, `calibration.brier`, and `total.naive_pnl`.
`simple_baseline.py` is CPU-only (sklearn on pipeline features; no GPU forward).

## If you decide it IS better — update the paper
- **Table 7 (`tab:baseline`)**: replace logistic/GBT/naive rows; swap the "recorded vintage" caption
  sentence for a one-line methodology note (e.g. "logistic/GBT fit on a stride-$K$ subsample of $N$
  training ticks").
- **Regime-split paragraph** ("The same-feature logistic is also regime-robust"): update the logistic
  high/low numbers; drop the "(recorded vintage)" qualifier; keep the honest comparison to the current
  settlement low-pred (+$51, P=0.63).
- **`fig:baseline`**: regenerate `figures/simple_baseline_pnl.pdf` via
  `python3 -m finmamba3.eval.make_paper_figures --baseline-json reports/recon_simple_baseline.json`
  (check its flags); update the caption CI numbers; drop the "recorded vintage" note.
- **`w2-reconciliation.md`**: replace "Finding 4 (logistic on recorded vintage)" with the
  current-vintage result + the methodology note.

## Honesty guardrails
- The deflationary claim is "a simple model on the same features matches/beats the world model." If the
  re-fit logistic comes out **below** the current settlement total (+$5,147), DO NOT force the old
  "exceeds" framing — soften to "is competitive with / matches," or keep the recorded version per the
  decision gate. The contribution survives either way (it's a control on architecture-independence).
- Document the methodology change (subsample/cap) plainly.

## Acceptance criteria
- `reports/recon_simple_baseline.json` exists with logistic + GBT totals, regime split, market-block
  CIs, Brier; the new-vs-recorded comparison + the keep/replace decision are documented in
  `w2-reconciliation.md`.
- If replaced: Table 7, the regime-split paragraph, and `fig:baseline` are all current-vintage with a
  clean methodology note and no remaining "recorded vintage" baseline caveats; `pdflatex` compiles
  clean (~23 pp, 0 errors/undefined); grep confirms no stale baseline numbers.
- Commit on **`logistic-refit-current-vintage`** with a clean message and **NO Claude co-author /
  trailer**; push the branch (do NOT push `main`). Leave a one-line summary (new numbers + keep/replace
  decision) for the user to review before merging.

## Reference
- Recipe + checkpoint IDs + vintage analysis: `w2-reconciliation.md`.
- Memory: `project_postscan_nongauge_result.md` (the COLM reframe shipped on commit `1ac9214`).

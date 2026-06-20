#!/bin/bash
# W2 follow-up: re-fit the logistic/GBT simple-model baseline (paper Table 7) on the current
# data/validation vintage. This is the one baseline still on the recorded vintage, because the full
# ~1.59M-tick sklearn fit timed out twice (1500s and 3000s) on the 4080/WSL box. Tractability lever:
# stride-subsample the supervised training ticks (--train-stride 4), keeping every 4th tick over the
# full time range (~397k of 1.59M ticks) so the fit cost is bounded without truncating any part of the
# training period. Gate, sizing, engine, predictability split, and bootstrap are otherwise unchanged.
set -uo pipefail
cd /mnt/host/c/Users/ruuud/algoverse/Drama
S=$(date +%s)
timeout 1700 python3 -m finmamba3.eval.simple_baseline \
  --config configs/lob_spot.yaml --data-train data/train --data-val data/validation \
  --norm-path saved_models/lob/norm_spot_seed0_base.json --assets BTC \
  --train-stride 4 \
  --out reports/recon_simple_baseline.json > reports/recon_simple_baseline.log 2>&1
RC=$?
E=$(date +%s)
echo "RC=$RC ELAPSED=$((E-S))s"
echo "---BASELINE LINES---"
grep -a '\[baseline\]' reports/recon_simple_baseline.log || true

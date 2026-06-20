#!/bin/bash
# W2 reconciliation: re-run every settlement/EV arm on ONE pinned vintage (current data/validation) so the
# paper's Tables 7-10 + abstract + conclusion all agree. Settlement (5rkphy4i) is run separately; this does
# the EV arms (edge_residual seeds 0/1/2, edge_full s0) and the edge_residual seed-0 stress tests. Verbatim
# canonical command from edge-architecture.md (no --calibration-temperature, default 1.0; no --intervals).
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
COMMON="--data-val data/validation --assets BTC --predictability-gate --predictability-threshold 0.60 --edge-threshold 0.03 --deterministic-latent --hours-val 9999"
SUMM=reports/recon_w2_summary.txt
: > $SUMM

run() {  # name config checkpoint norm extra
  local name=$1 config=$2 ckpt=$3 norm=$4; shift 4; local extra="$*"
  echo "### $name ###"
  python3 -m finmamba3.eval.pnl_backtest --config $config \
    --checkpoint saved_models/lob/LOB/$ckpt/ckpt/world_model_final.pth \
    --norm-path saved_models/lob/$norm $COMMON $extra \
    --out reports/recon_${name}.json > reports/recon_${name}.log 2>&1
  echo "== $name ==" >> $SUMM
  grep -aE '^\[pnl\]|predictability split' reports/recon_${name}.log >> $SUMM
  echo "  [$name done]"
}

run edge_residual_s0 configs/lob_edge_residual.yaml 61kb278p norm_edge_residual_s0.json --prob-source edge
run edge_residual_s1 configs/lob_edge_residual.yaml 1lukxyyv norm_edge_residual_s0.json --prob-source edge
run edge_residual_s2 configs/lob_edge_residual.yaml 9qk67y37 norm_edge_residual_s0.json --prob-source edge
run edge_full_s0     configs/lob_edge_full.yaml     8nti99zj norm_edge_full_s0.json     --prob-source edge
run edge_s0_slip   configs/lob_edge_residual.yaml 61kb278p norm_edge_residual_s0.json --prob-source edge --slippage-per-share 0.01
run edge_s0_depth  configs/lob_edge_residual.yaml 61kb278p norm_edge_residual_s0.json --prob-source edge --min-book-depth 60000 --max-book-depth 130000
run edge_s0_tte    configs/lob_edge_residual.yaml 61kb278p norm_edge_residual_s0.json --prob-source edge --min-tte-frac 0.25
echo "RECON_W2_DONE"
cat $SUMM

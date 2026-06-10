#!/bin/bash
# Run the regime-FiLM gap evals for one dataset's A/B: per seed, both metrics x both axes.
# Reads reports/<tag>_ab_manifest.txt (lines "<tag> <seed> <arm> <ckpt>") and writes a compact
# GAP line per (seed, metric, axis) plus per-arm markdown tables.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
TAG=$1          # fi2010 | kaggle
CONFIG=$2       # configs/fi2010_studentt.yaml | configs/kaggle_btc.yaml
DATAVAL=$3      # data/fi2010/validation | data
WINLEN=$4       # 512 | 128
THRESH=$5       # 0.0 | 0.01
SEEDS=${6:-"0 1"}
MANIFEST=reports/${TAG}_ab_manifest.txt
OUT=reports/${TAG}_ab_gaps.txt
: > $OUT
for SEED in $SEEDS; do
  BASE=$(awk -v s=$SEED '$2==s && $3=="baseline"{print $4}' $MANIFEST)
  TREAT=$(awk -v s=$SEED '$2==s && $3=="treatment"{print $4}' $MANIFEST)
  NORM=saved_models/lob/${TAG}_studentt_norm_s${SEED}.json
  for METRIC in direction_macro_f1 recon_nll; do
    for AXIS in predictability spot_vol; do
      MD=reports/${TAG}_${METRIC}_${AXIS}_s${SEED}.md
      python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config $CONFIG \
        --dataset $TAG --regime-axis $AXIS --metric $METRIC \
        --baseline-checkpoint $BASE --treatment-checkpoint $TREAT \
        --data-val $DATAVAL --norm-path $NORM --window-len $WINLEN --threshold $THRESH \
        --out $MD > /dev/null 2>&1
      GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
      echo "GAP $TAG $SEED $METRIC $AXIS $GAP" | tee -a $OUT
    done
  done
done
echo "${TAG}_EVAL_DONE"

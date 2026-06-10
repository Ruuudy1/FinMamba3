#!/bin/bash
# Kaggle BTC class-balanced direction A/B (symmetric with the FI-2010 CB run): inverse-frequency CE so the
# head does not collapse to the flat majority at threshold 0.01. Confirms a non-degenerate primary metric on
# the headline dataset with the gap still ~0 (FiLM inert). seeds 0,1; eval direction macro-F1 on both axes.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
MANIFEST=reports/kaggle_cb_manifest.txt
: > $MANIFEST
for SEED in 0 1; do
  for ARM in baseline treatment; do
    FLAG=""
    if [ "$ARM" = "treatment" ]; then FLAG="--Models.WorldModel.RegimeFiLM.Enabled True"; fi
    NORM=saved_models/lob/kaggle_cb_norm_s${SEED}.json
    LOG=reports/kaggle_cb_${ARM}_s${SEED}.log
    echo "=== TRAIN kaggle CLASS-BALANCED seed=$SEED arm=$ARM ==="
    python3 -m finmamba3.train --config configs/kaggle_btc.yaml \
      --data-train data --data-val data --dataset kaggle \
      --BasicSettings.Seed $SEED $FLAG --Models.WorldModel.Direction.ClassBalanced True \
      --norm-path $NORM --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
    CKPT=$(ls -t saved_models/kaggle/LOB/*/ckpt/world_model_final.pth | head -1)
    echo "kaggle $SEED $ARM $CKPT" >> $MANIFEST
    echo "  done $SEED $ARM: $(grep -a '\[loss\]' $LOG | tail -1 | grep -oE 'dir=[0-9.]+ .*film_g=[0-9.]+ film_b=[0-9.]+ reg_H=[0-9.]+')"
  done
done
cat $MANIFEST
for SEED in 0 1; do
  BASE=$(awk -v s=$SEED '$2==s && $3=="baseline"{print $4}' $MANIFEST)
  TREAT=$(awk -v s=$SEED '$2==s && $3=="treatment"{print $4}' $MANIFEST)
  NORM=saved_models/lob/kaggle_cb_norm_s${SEED}.json
  for AXIS in predictability spot_vol; do
    MD=reports/kaggle_cb_direction_${AXIS}_s${SEED}.md
    python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/kaggle_btc.yaml \
      --dataset kaggle --regime-axis $AXIS --metric direction_macro_f1 \
      --baseline-checkpoint $BASE --treatment-checkpoint $TREAT --data-val data \
      --norm-path $NORM --window-len 128 --threshold 0.01 --out $MD > /dev/null 2>&1
    GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
    echo "CB-GAP kaggle $SEED direction $AXIS $GAP :: $(grep -E '^\| (baseline|treatment)' $MD | tr '\n' ' ')"
  done
done
echo KAGGLE_CB_DONE

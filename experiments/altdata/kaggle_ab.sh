#!/bin/bash
# Train the Kaggle BTC regime-FiLM A/B: baseline (FiLM off) + treatment (FiLM on) on seeds 0,1.
# Mirrors fi2010_ab.sh but over the Kaggle loader (config-driven 168h/72h chronological split).
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
SEEDS=${2:-"0 1"}
MANIFEST=reports/kaggle_ab_manifest.txt
: > $MANIFEST
for SEED in $SEEDS; do
  for ARM in baseline treatment; do
    FLAG=""
    if [ "$ARM" = "treatment" ]; then FLAG="--Models.WorldModel.RegimeFiLM.Enabled True"; fi
    NORM=saved_models/lob/kaggle_studentt_norm_s${SEED}.json
    LOG=reports/kaggle_${ARM}_s${SEED}.log
    echo "=== TRAIN kaggle seed=$SEED arm=$ARM steps=$STEPS ==="
    python3 -m finmamba3.train --config configs/kaggle_btc.yaml \
      --data-train data --data-val data --dataset kaggle \
      --BasicSettings.Seed $SEED $FLAG --norm-path $NORM \
      --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
    CKPT=$(ls -t saved_models/kaggle/LOB/*/ckpt/world_model_final.pth | head -1)
    LASTLOSS=$(grep -a '\[loss\]' $LOG | tail -1)
    echo "kaggle $SEED $ARM $CKPT" >> $MANIFEST
    echo "    loss: $LASTLOSS" >> $MANIFEST
    echo "done seed=$SEED arm=$ARM -> $CKPT"
    echo "    $LASTLOSS"
  done
done
echo "KAGGLE_AB_TRAIN_DONE"
cat $MANIFEST

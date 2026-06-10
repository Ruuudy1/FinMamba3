#!/bin/bash
# Train the FI-2010 regime-FiLM A/B: baseline (FiLM off) + treatment (FiLM on) on seeds 0,1.
# Captures each final checkpoint and the last [loss] line (film_g/film_b/reg_H) into a manifest.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
MANIFEST=reports/fi2010_ab_manifest.txt
: > $MANIFEST
for SEED in 0 1; do
  for ARM in baseline treatment; do
    FLAG=""
    if [ "$ARM" = "treatment" ]; then FLAG="--Models.WorldModel.RegimeFiLM.Enabled True"; fi
    NORM=saved_models/lob/fi2010_studentt_norm_s${SEED}.json
    LOG=reports/fi2010_${ARM}_s${SEED}.log
    echo "=== TRAIN fi2010 seed=$SEED arm=$ARM steps=$STEPS ==="
    python3 -m finmamba3.train --config configs/fi2010_studentt.yaml \
      --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
      --BasicSettings.Seed $SEED $FLAG --norm-path $NORM \
      --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
    CKPT=$(ls -t saved_models/lob/LOB/*/ckpt/world_model_final.pth | head -1)
    LASTLOSS=$(grep -a '\[loss\]' $LOG | tail -1)
    echo "fi2010 $SEED $ARM $CKPT" >> $MANIFEST
    echo "    loss: $LASTLOSS" >> $MANIFEST
    echo "done seed=$SEED arm=$ARM -> $CKPT"
    echo "    $LASTLOSS"
  done
done
echo "FI2010_AB_TRAIN_DONE"
cat $MANIFEST

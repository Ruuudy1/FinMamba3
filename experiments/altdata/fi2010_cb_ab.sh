#!/bin/bash
# FI-2010 class-balanced direction A/B: inverse-frequency CE weighting (Direction.ClassBalanced=True) at
# threshold 0.0 so the head does not collapse to the 76%-flat majority. Confirms a non-degenerate primary
# metric with the gap still ~0 (FiLM inert). seeds 0,1; eval direction macro-F1 on both axes.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
MANIFEST=reports/fi2010_cb_manifest.txt
: > $MANIFEST
for SEED in 0 1; do
  for ARM in baseline treatment; do
    FLAG=""
    if [ "$ARM" = "treatment" ]; then FLAG="--Models.WorldModel.RegimeFiLM.Enabled True"; fi
    NORM=saved_models/lob/fi2010_cb_norm_s${SEED}.json
    LOG=reports/fi2010_cb_${ARM}_s${SEED}.log
    echo "=== TRAIN fi2010 CLASS-BALANCED seed=$SEED arm=$ARM ==="
    python3 -m finmamba3.train --config configs/fi2010_studentt.yaml \
      --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
      --BasicSettings.Seed $SEED $FLAG --Models.WorldModel.Direction.ClassBalanced True \
      --norm-path $NORM --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
    CKPT=$(ls -t saved_models/lob/LOB/*/ckpt/world_model_final.pth | head -1)
    echo "fi2010 $SEED $ARM $CKPT" >> $MANIFEST
    echo "  done $SEED $ARM: $(grep -a '\[loss\]' $LOG | tail -1 | grep -oE 'dir=[0-9.]+ .*film_g=[0-9.]+ film_b=[0-9.]+ reg_H=[0-9.]+')"
  done
done
cat $MANIFEST
for SEED in 0 1; do
  BASE=$(awk -v s=$SEED '$2==s && $3=="baseline"{print $4}' $MANIFEST)
  TREAT=$(awk -v s=$SEED '$2==s && $3=="treatment"{print $4}' $MANIFEST)
  NORM=saved_models/lob/fi2010_cb_norm_s${SEED}.json
  for AXIS in predictability spot_vol; do
    MD=reports/fi2010_cb_direction_${AXIS}_s${SEED}.md
    python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/fi2010_studentt.yaml \
      --dataset fi2010 --regime-axis $AXIS --metric direction_macro_f1 \
      --baseline-checkpoint $BASE --treatment-checkpoint $TREAT --data-val data/fi2010/validation \
      --norm-path $NORM --window-len 512 --threshold 0.0 --out $MD > /dev/null 2>&1
    GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
    echo "CB-GAP fi2010 $SEED direction $AXIS $GAP :: $(grep -E '^\| (baseline|treatment)' $MD | tr '\n' ' ')"
  done
done
echo FI2010_CB_DONE

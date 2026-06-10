#!/bin/bash
# Robustness extensions: FI-2010 predictability escalation (symmetry with Kaggle) + ETH/ADA cross-asset
# Kaggle A/B (the null is not BTC-specific). Trains then evals; prints film_g diagnostics throughout.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}

echo "### FI2010 PREDICTABILITY-SUPERVISED ESCALATION ###"
ELOG=reports/fi2010_escalation_s0.log
python3 -m finmamba3.train --config configs/fi2010_studentt.yaml \
  --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
  --BasicSettings.Seed 0 \
  --Models.WorldModel.RegimeFiLM.Enabled True --Models.WorldModel.RegimeFiLM.InitScale 0.1 \
  --Models.WorldModel.RegimeFiLM.LRMult 50 --Models.WorldModel.RegimeFiLM.SuperviseVol True \
  --Models.WorldModel.RegimeFiLM.SuperviseAxis predictability --Models.WorldModel.RegimeFiLM.SupervisionWeight 30 \
  --Models.WorldModel.RegimeFiLM.FeedObsVol True --Models.WorldModel.RegimeFiLM.DecoupleRouterFromFiLM True \
  --Models.WorldModel.RegimeFiLM.EntropyCoef 0.0 \
  --norm-path saved_models/lob/fi2010_escalation_norm.json --JointTrainAgent.SampleMaxSteps $STEPS > $ELOG 2>&1
echo "fi2010 escalation film_g trajectory:"
grep -a '\[loss\]' $ELOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+).*/\1 \2 \3/' | awk 'NR==1 || $1%500==0 {printf "  step=%s film_g=%s reg_H=%s\n",$1,$2,$3}'

MANIFEST=reports/kaggle_crossasset_manifest.txt
: > $MANIFEST
for ASSET in eth ada; do
  for ARM in baseline treatment; do
    FLAG=""
    if [ "$ARM" = "treatment" ]; then FLAG="--Models.WorldModel.RegimeFiLM.Enabled True"; fi
    NORM=saved_models/lob/kaggle_${ASSET}_norm_s0.json
    LOG=reports/kaggle_${ASSET}_${ARM}_s0.log
    echo "### TRAIN kaggle $ASSET arm=$ARM ###"
    python3 -m finmamba3.train --config configs/kaggle_${ASSET}.yaml \
      --data-train data --data-val data --dataset kaggle \
      --BasicSettings.Seed 0 $FLAG --norm-path $NORM --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
    CKPT=$(ls -t saved_models/kaggle/LOB/*/ckpt/world_model_final.pth | head -1)
    echo "$ASSET 0 $ARM $CKPT" >> $MANIFEST
    echo "  done $ASSET $ARM: $(grep -a '\[loss\]' $LOG | tail -1 | grep -oE 'film_g=[0-9.]+ film_b=[0-9.]+ reg_H=[0-9.]+')"
  done
done
echo "CROSSASSET_TRAIN_DONE"; cat $MANIFEST

GAPS=reports/kaggle_crossasset_gaps.txt
: > $GAPS
for ASSET in eth ada; do
  BASE=$(awk -v a=$ASSET '$1==a && $3=="baseline"{print $4}' $MANIFEST)
  TREAT=$(awk -v a=$ASSET '$1==a && $3=="treatment"{print $4}' $MANIFEST)
  NORM=saved_models/lob/kaggle_${ASSET}_norm_s0.json
  for METRIC in direction_macro_f1 recon_nll; do
    MD=reports/kaggle_${ASSET}_${METRIC}_pred.md
    python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/kaggle_${ASSET}.yaml \
      --dataset kaggle --regime-axis predictability --metric $METRIC \
      --baseline-checkpoint $BASE --treatment-checkpoint $TREAT \
      --data-val data --norm-path $NORM --window-len 128 --threshold 0.01 --out $MD > /dev/null 2>&1
    GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
    echo "CROSSASSET-GAP $ASSET $METRIC predictability $GAP" | tee -a $GAPS
  done
done
echo "ROBUSTNESS_ALL_DONE"; cat $GAPS

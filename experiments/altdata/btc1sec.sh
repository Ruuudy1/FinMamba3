#!/bin/bash
# Resolution coverage: the Kaggle BTC FiLM A/B at 1sec bars (4h train + 1.5h val slice of the 1.8 GB file).
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
MANIFEST=reports/kaggle_btc1sec_manifest.txt
: > $MANIFEST
for ARM in baseline treatment; do
  FLAG=""
  if [ "$ARM" = "treatment" ]; then FLAG="--Models.WorldModel.RegimeFiLM.Enabled True"; fi
  NORM=saved_models/lob/kaggle_btc1sec_norm_s0.json
  LOG=reports/kaggle_btc1sec_${ARM}_s0.log
  echo "### TRAIN kaggle BTC 1sec arm=$ARM ###"
  python3 -m finmamba3.train --config configs/kaggle_btc_1sec.yaml --data-train data --data-val data --dataset kaggle \
    --BasicSettings.Seed 0 $FLAG --norm-path $NORM --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
  CKPT=$(ls -t saved_models/kaggle/LOB/*/ckpt/world_model_final.pth | head -1)
  echo "btc1sec 0 $ARM $CKPT" >> $MANIFEST
  echo "  done $ARM: $(grep -a '\[loss\]' $LOG | tail -1 | grep -oE 'film_g=[0-9.]+ film_b=[0-9.]+ reg_H=[0-9.]+')"
done
cat $MANIFEST
BASE=$(awk '$3=="baseline"{print $4}' $MANIFEST)
TREAT=$(awk '$3=="treatment"{print $4}' $MANIFEST)
NORM=saved_models/lob/kaggle_btc1sec_norm_s0.json
for METRIC in direction_macro_f1 recon_nll; do
  MD=reports/kaggle_btc1sec_${METRIC}_pred.md
  python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/kaggle_btc_1sec.yaml \
    --dataset kaggle --regime-axis predictability --metric $METRIC \
    --baseline-checkpoint $BASE --treatment-checkpoint $TREAT \
    --data-val data --norm-path $NORM --window-len 128 --threshold 0.01 --out $MD > /dev/null 2>&1
  GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
  echo "BTC1SEC-GAP $METRIC predictability $GAP"
done
echo "BTC1SEC_DONE"

#!/bin/bash
# Recon-NLL (secondary metric) on the class-balanced checkpoints — the CB runs are distinct models (the
# direction-CE reweighting changes the joint gradient), so confirm the secondary metric is also null on them.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
for SEED in 0 1; do
  BASE=$(awk -v s=$SEED '$2==s && $3=="baseline"{print $4}' reports/fi2010_cb_manifest.txt)
  TREAT=$(awk -v s=$SEED '$2==s && $3=="treatment"{print $4}' reports/fi2010_cb_manifest.txt)
  NORM=saved_models/lob/fi2010_cb_norm_s${SEED}.json
  for AXIS in predictability spot_vol; do
    MD=reports/fi2010_cb_recon_${AXIS}_s${SEED}.md
    python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/fi2010_studentt.yaml \
      --dataset fi2010 --regime-axis $AXIS --metric recon_nll --baseline-checkpoint $BASE \
      --treatment-checkpoint $TREAT --data-val data/fi2010/validation --norm-path $NORM --window-len 512 \
      --out $MD > /dev/null 2>&1
    GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
    echo "CB-RECON fi2010 $SEED $AXIS $GAP"
  done
done
for SEED in 0 1; do
  BASE=$(awk -v s=$SEED '$2==s && $3=="baseline"{print $4}' reports/kaggle_cb_manifest.txt)
  TREAT=$(awk -v s=$SEED '$2==s && $3=="treatment"{print $4}' reports/kaggle_cb_manifest.txt)
  NORM=saved_models/lob/kaggle_cb_norm_s${SEED}.json
  for AXIS in predictability spot_vol; do
    MD=reports/kaggle_cb_recon_${AXIS}_s${SEED}.md
    python3 -m finmamba3.eval.eval_regime_generalization_fi2010 --config configs/kaggle_btc.yaml \
      --dataset kaggle --regime-axis $AXIS --metric recon_nll --baseline-checkpoint $BASE \
      --treatment-checkpoint $TREAT --data-val data --norm-path $NORM --window-len 128 \
      --out $MD > /dev/null 2>&1
    GAP=$(grep 'generalization gap' $MD | sed -E 's/.*= ([+-]?[0-9.]+)\*\*.*/\1/')
    echo "CB-RECON kaggle $SEED $AXIS $GAP"
  done
done
echo CB_RECON_DONE

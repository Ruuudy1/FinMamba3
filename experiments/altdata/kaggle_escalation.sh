#!/bin/bash
# Strongest-null test on Kaggle BTC: force FiLM active (InitScale 0.1) and supervise the router on the
# predictability (Efficiency-Ratio) bucket with the prior campaign's strongest settings, then watch
# whether film_g stays > 0 or decays to identity. A monotone film_g decay despite forced activation +
# direct ER supervision is the strongest cross-dataset form of the regime-FiLM null.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
SEED=${2:-0}
NORM=saved_models/lob/kaggle_escalation_norm_s${SEED}.json
LOG=reports/kaggle_escalation_s${SEED}.log
echo "=== TRAIN kaggle ESCALATION seed=$SEED (forced-active FiLM + predictability-supervised router) ==="
python3 -m finmamba3.train --config configs/kaggle_btc.yaml \
  --data-train data --data-val data --dataset kaggle \
  --BasicSettings.Seed $SEED \
  --Models.WorldModel.RegimeFiLM.Enabled True \
  --Models.WorldModel.RegimeFiLM.InitScale 0.1 \
  --Models.WorldModel.RegimeFiLM.LRMult 50 \
  --Models.WorldModel.RegimeFiLM.SuperviseVol True \
  --Models.WorldModel.RegimeFiLM.SuperviseAxis predictability \
  --Models.WorldModel.RegimeFiLM.SupervisionWeight 30 \
  --Models.WorldModel.RegimeFiLM.FeedObsVol True \
  --Models.WorldModel.RegimeFiLM.DecoupleRouterFromFiLM True \
  --Models.WorldModel.RegimeFiLM.EntropyCoef 0.0 \
  --norm-path $NORM \
  --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
CKPT=$(ls -t saved_models/kaggle/LOB/*/ckpt/world_model_final.pth | head -1)
echo "escalation ckpt: $CKPT"
echo "=== film_g / reg_H trajectory (the decisive evidence) ==="
grep -a '\[loss\]' $LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=([0-9.]+) reg_H=([0-9.]+).*/step=\1 film_g=\2 reg_H=\4/'
echo "KAGGLE_ESCALATION_DONE"

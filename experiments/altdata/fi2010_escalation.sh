#!/bin/bash
# Strongest-null test on FI-2010: force FiLM active (InitScale 0.1) and supervise the router on the
# predictability (Efficiency-Ratio) bucket with the prior campaign's strongest settings, then watch
# whether film_g stays > 0 or decays to identity. Parametrized twin of kaggle_escalation.sh so both
# datasets can be run at additional seeds for tau error bars (the seed-0 run is reports/fi2010_escalation_s0.log,
# produced by robustness.sh under a different norm-path naming; this script is seed-parametrized for reruns).
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
SEED=${2:-0}
NORM=saved_models/lob/fi2010_escalation_norm_s${SEED}.json
LOG=reports/fi2010_escalation_s${SEED}.log
echo "=== TRAIN fi2010 ESCALATION seed=$SEED (forced-active FiLM + predictability-supervised router) ==="
python3 -m finmamba3.train --config configs/fi2010_studentt.yaml \
  --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
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
echo "=== film_g / reg_H trajectory (the decisive evidence) ==="
grep -a '\[loss\]' $LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=([0-9.]+) reg_H=([0-9.]+).*/step=\1 film_g=\2 reg_H=\4/'
echo "FI2010_ESCALATION_DONE"

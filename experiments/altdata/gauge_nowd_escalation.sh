#!/bin/bash
# B1a (gauge-mechanism.md): the SAME forced-active, ER-supervised strongest-null escalation as
# kaggle_escalation.sh, but with the LaProp weight decay set to ZERO on BOTH the base and FiLM param groups.
# Gauge absorption predicts the constant/gauge component of the scale sits in a flat loss valley, so with no
# weight decay there is nothing to pull that component to identity; if film_g STILL decays monotonically to
# identity with WD off, the decay is not pure gauge but reconstruction-driven. Compared against the WD=1e-4
# baseline reports/kaggle_escalation_s0.log so only the weight decay differs.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
SEED=${2:-0}
NORM=saved_models/lob/kaggle_nowd_norm_s${SEED}.json
LOG=reports/kaggle_escalation_nowd_s${SEED}.log
echo "=== TRAIN kaggle ESCALATION (NO WEIGHT DECAY) seed=$SEED ==="
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
  --Models.WorldModel.Weight_decay 0.0 \
  --norm-path $NORM \
  --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
echo "=== film_g / reg_H trajectory (NO WD) ==="
grep -a '\[loss\]' $LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=([0-9.]+) reg_H=([0-9.]+).*/step=\1 film_g=\2 reg_H=\4/'
echo "=== DECAY FIT: NO-WD vs WD=1e-4 baseline ==="
python3 -c "import sys; sys.path.insert(0,'experiments/altdata'); from decay_fit import fit; fit('Kaggle NO-WD', '$LOG'); fit('Kaggle WD=1e-4 baseline', 'reports/kaggle_escalation_s0.log')"
echo "GAUGE_NOWD_DONE"

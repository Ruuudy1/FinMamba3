#!/bin/bash
# Volatility-axis strongest-null test (the bridge experiment closing the axis-conflation): the SAME
# forced-active, router-supervised escalation as kaggle_escalation.sh, but supervising the router on the
# DECODABLE volatility bucket (SuperviseAxis vol) instead of the weakly-decodable efficiency-ratio
# (predictability) axis. A frozen-router head fits the vol bucket to ~0.92 (vol-scale.md), so if film_g
# still decays to identity here the collapse is gauge absorption, not missing regime signal. Input-side
# gate (no ApplyAfterScan). Only the supervised axis and the log/norm paths differ from the predictability
# escalation (kaggle_escalation.sh), so the comparison is clean. Neither config sets SuperviseChannelIndex,
# so the vol label and the FeedObsVol feature are both taken from the obs midprice (the axis the 0.92
# freeze-probe and the eval split use).
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
SEED=${2:-0}
ESCALATION="--Models.WorldModel.RegimeFiLM.Enabled True --Models.WorldModel.RegimeFiLM.InitScale 0.1 \
  --Models.WorldModel.RegimeFiLM.LRMult 50 --Models.WorldModel.RegimeFiLM.SuperviseVol True \
  --Models.WorldModel.RegimeFiLM.SuperviseAxis vol --Models.WorldModel.RegimeFiLM.SupervisionWeight 30 \
  --Models.WorldModel.RegimeFiLM.FeedObsVol True --Models.WorldModel.RegimeFiLM.DecoupleRouterFromFiLM True \
  --Models.WorldModel.RegimeFiLM.EntropyCoef 0.0"

echo "### FI2010 VOL-AXIS (decodable) ESCALATION seed=$SEED ###"
FI_LOG=reports/fi2010_volaxis_escalation_s${SEED}.log
python3 -m finmamba3.train --config configs/fi2010_studentt.yaml \
  --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
  --BasicSettings.Seed $SEED $ESCALATION \
  --norm-path saved_models/lob/fi2010_volaxis_norm_s${SEED}.json \
  --JointTrainAgent.SampleMaxSteps $STEPS > $FI_LOG 2>&1
echo "fi2010 vol-axis film_g trajectory:"
grep -a '\[loss\]' $FI_LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+).*/\1 \2 \3/' | awk 'NR==1 || $1%500==0 {printf "  step=%s film_g=%s reg_H=%s\n",$1,$2,$3}'

echo "### KAGGLE BTC VOL-AXIS (decodable) ESCALATION seed=$SEED ###"
KG_LOG=reports/kaggle_volaxis_escalation_s${SEED}.log
python3 -m finmamba3.train --config configs/kaggle_btc.yaml \
  --data-train data --data-val data --dataset kaggle \
  --BasicSettings.Seed $SEED $ESCALATION \
  --norm-path saved_models/lob/kaggle_volaxis_norm_s${SEED}.json \
  --JointTrainAgent.SampleMaxSteps $STEPS > $KG_LOG 2>&1
echo "kaggle vol-axis film_g trajectory:"
grep -a '\[loss\]' $KG_LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+).*/\1 \2 \3/' | awk 'NR==1 || $1%500==0 {printf "  step=%s film_g=%s reg_H=%s\n",$1,$2,$3}'

echo "### EXP-DECAY FIT (c<0.02 => approaches identity => gauge absorption on the decodable axis) ###"
python3 -c "import sys; sys.path.insert(0,'experiments/altdata'); from decay_fit import fit; fit('FI-2010 vol-axis', '$FI_LOG'); fit('Kaggle BTC vol-axis', '$KG_LOG')"
echo "VOLAXIS_ESCALATION_DONE"

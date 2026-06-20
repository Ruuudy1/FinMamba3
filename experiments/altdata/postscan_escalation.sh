#!/bin/bash
# Non-gauge positive control: the SAME forced-active, ER-supervised strongest-null escalation that drove
# the input-affine FiLM to identity (kaggle_escalation.sh / robustness.sh), but with ApplyAfterScan True so
# the gate sits on each block's OUTPUT past the selective scan instead of its input. An input-side affine is
# a gauge direction the host Delta/B/C projections absorb; this run tests whether a post-scan gate escapes
# that absorption. Outcome read off decay_fit: asymptote c plateauing above identity = the non-gauge boundary
# (input-affine folds, post-scan survives); c -> 0 = a broader null than gauge absorption alone. Only the
# ApplyAfterScan flag and the log paths differ from the input-affine escalation, so the comparison is clean.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}
SEED=${2:-0}
ESCALATION="--Models.WorldModel.RegimeFiLM.Enabled True --Models.WorldModel.RegimeFiLM.InitScale 0.1 \
  --Models.WorldModel.RegimeFiLM.LRMult 50 --Models.WorldModel.RegimeFiLM.SuperviseVol True \
  --Models.WorldModel.RegimeFiLM.SuperviseAxis predictability --Models.WorldModel.RegimeFiLM.SupervisionWeight 30 \
  --Models.WorldModel.RegimeFiLM.FeedObsVol True --Models.WorldModel.RegimeFiLM.DecoupleRouterFromFiLM True \
  --Models.WorldModel.RegimeFiLM.EntropyCoef 0.0 --Models.WorldModel.RegimeFiLM.ApplyAfterScan True"

echo "### FI2010 POST-SCAN (non-gauge) ESCALATION seed=$SEED ###"
FI_LOG=reports/fi2010_postscan_escalation_s${SEED}.log
python3 -m finmamba3.train --config configs/fi2010_studentt.yaml \
  --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
  --BasicSettings.Seed $SEED $ESCALATION \
  --norm-path saved_models/lob/fi2010_postscan_norm_s${SEED}.json \
  --JointTrainAgent.SampleMaxSteps $STEPS > $FI_LOG 2>&1
echo "fi2010 post-scan film_g trajectory:"
grep -a '\[loss\]' $FI_LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+).*/\1 \2 \3/' | awk 'NR==1 || $1%500==0 {printf "  step=%s film_g=%s reg_H=%s\n",$1,$2,$3}'

echo "### KAGGLE BTC POST-SCAN (non-gauge) ESCALATION seed=$SEED ###"
KG_LOG=reports/kaggle_postscan_escalation_s${SEED}.log
python3 -m finmamba3.train --config configs/kaggle_btc.yaml \
  --data-train data --data-val data --dataset kaggle \
  --BasicSettings.Seed $SEED $ESCALATION \
  --norm-path saved_models/lob/kaggle_postscan_norm_s${SEED}.json \
  --JointTrainAgent.SampleMaxSteps $STEPS > $KG_LOG 2>&1
echo "kaggle post-scan film_g trajectory:"
grep -a '\[loss\]' $KG_LOG | sed -E 's/.*step=([0-9]+).*film_g=([0-9.]+) film_b=[0-9.]+ reg_H=([0-9.]+).*/\1 \2 \3/' | awk 'NR==1 || $1%500==0 {printf "  step=%s film_g=%s reg_H=%s\n",$1,$2,$3}'

echo "### EXP-DECAY FIT (c<0.02 => approaches identity / broader null; c plateau => non-gauge boundary) ###"
python3 -c "import sys; sys.path.insert(0,'experiments/altdata'); from decay_fit import fit; fit('FI-2010 post-scan', '$FI_LOG'); fit('Kaggle BTC post-scan', '$KG_LOG')"
echo "POSTSCAN_ESCALATION_DONE"

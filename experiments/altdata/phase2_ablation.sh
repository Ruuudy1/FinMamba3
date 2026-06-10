#!/bin/bash
# G2 backbone ablation under matched parameters: report recon NLL + direction macro-F1 per backbone.
# Mamba-3 SISO reuses the FiLM-off baseline A/B checkpoint; Mamba-1/Mamba-2/Transformer train from
# matched per-backbone configs. MIMO is NOT run here (A100-only; flagged in the report).
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
TAG=$1          # fi2010 | kaggle
BASECFG=$2      # the Mamba-3 SISO config (configs/fi2010_studentt.yaml | configs/kaggle_btc.yaml)
DATATRAIN=$3; DATAVAL=$4; STEPS=$5; THRESH=$6; SEED=${7:-0}
REUSE_M3=$8     # baseline A/B checkpoint reused for the Mamba-3 SISO row
REUSE_NORM=$9   # its normalization json
RESULTS=reports/${TAG}_backbone_ablation.txt
: > $RESULTS
echo "=== EVAL $TAG Mamba3-SISO (reused baseline ckpt) ==="
python3 -m finmamba3.eval.eval_backbone_metrics --config $BASECFG --dataset $TAG --backbone Mamba3 \
  --checkpoint $REUSE_M3 --data-val $DATAVAL --norm-path $REUSE_NORM --threshold $THRESH \
  --out reports/${TAG}_Mamba3SISO_metrics.md 2>&1 | grep -E '^\| (Mamba|Transformer)' | tee -a $RESULTS
for BK in mamba1 mamba2 transformer; do
  CFG=configs/${TAG}_${BK}.yaml
  NORM=saved_models/lob/${TAG}_${BK}_norm.json
  LOG=reports/${TAG}_${BK}.log
  echo "=== TRAIN $TAG $BK ==="
  python3 -m finmamba3.train --config $CFG --data-train $DATATRAIN --data-val $DATAVAL --dataset $TAG \
    --BasicSettings.Seed $SEED --norm-path $NORM --JointTrainAgent.SampleMaxSteps $STEPS > $LOG 2>&1
  CKPT=$(ls -t saved_models/*/LOB/*/ckpt/world_model_final.pth | head -1)
  echo "=== EVAL $TAG $BK -> $CKPT ==="
  python3 -m finmamba3.eval.eval_backbone_metrics --config $CFG --dataset $TAG \
    --checkpoint $CKPT --data-val $DATAVAL --norm-path $NORM --threshold $THRESH \
    --out reports/${TAG}_${BK}_metrics.md 2>&1 | grep -E '^\| (Mamba|Transformer)' | tee -a $RESULTS
done
echo "${TAG}_ABLATION_DONE"
cat $RESULTS

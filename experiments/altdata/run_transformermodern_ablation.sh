#!/bin/bash
# TransformerModern row for the matched-parameter backbone ablation (paper Table 2).
# Same protocol as phase2_ablation.sh: train at $STEPS (default 3000), seed 0, then eval
# held-out recon NLL + direction macro-F1 via eval_backbone_metrics. Pre-norm transformer
# matched to the Mamba-3 block; NumLayers=2 keeps it at the ~10.3M matched-parameter budget.
set -e
cd /mnt/host/c/Users/ruuud/algoverse/Drama
STEPS=${1:-3000}

echo "=== TRAIN fi2010 TransformerModern ($STEPS steps) ==="
python3 -m finmamba3.train --config configs/fi2010_transformermodern.yaml \
  --data-train data/fi2010/train --data-val data/fi2010/validation --dataset fi2010 \
  --BasicSettings.Seed 0 --norm-path saved_models/lob/fi2010_transformermodern_norm.json \
  --JointTrainAgent.SampleMaxSteps "$STEPS" > reports/fi2010_transformermodern.log 2>&1
CKPT_FI=$(ls -t saved_models/*/LOB/*/ckpt/world_model_final.pth | head -1)
echo "fi2010 ckpt: $CKPT_FI"
python3 -m finmamba3.eval.eval_backbone_metrics --config configs/fi2010_transformermodern.yaml \
  --dataset fi2010 --checkpoint "$CKPT_FI" --data-val data/fi2010/validation \
  --norm-path saved_models/lob/fi2010_transformermodern_norm.json --threshold 0.0 \
  --out reports/fi2010_transformermodern_metrics.md

echo "=== TRAIN kaggle TransformerModern ($STEPS steps) ==="
python3 -m finmamba3.train --config configs/kaggle_transformermodern.yaml \
  --data-train data --data-val data --dataset kaggle \
  --BasicSettings.Seed 0 --norm-path saved_models/lob/kaggle_transformermodern_norm.json \
  --JointTrainAgent.SampleMaxSteps "$STEPS" > reports/kaggle_transformermodern.log 2>&1
CKPT_KG=$(ls -t saved_models/*/LOB/*/ckpt/world_model_final.pth | head -1)
echo "kaggle ckpt: $CKPT_KG"
python3 -m finmamba3.eval.eval_backbone_metrics --config configs/kaggle_transformermodern.yaml \
  --dataset kaggle --checkpoint "$CKPT_KG" --data-val data \
  --norm-path saved_models/lob/kaggle_transformermodern_norm.json --threshold 0.01 \
  --out reports/kaggle_transformermodern_metrics.md

echo "TRANSFORMERMODERN_ABLATION_DONE"
echo "=== fi2010 ==="; cat reports/fi2010_transformermodern_metrics.md
echo "=== kaggle ==="; cat reports/kaggle_transformermodern_metrics.md

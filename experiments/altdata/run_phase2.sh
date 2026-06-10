#!/bin/bash
# G2 backbone ablation on both datasets + the FI-2010 DeepLOB/majority/linear-AR benchmark.
set -e
cd /mnt/host/c/tmp
sed -i 's/\r$//' phase2_ablation.sh
echo "### FI2010 BACKBONE ABLATION ###"
bash phase2_ablation.sh fi2010 configs/fi2010_studentt.yaml data/fi2010/train data/fi2010/validation 3000 0.0 0 \
  saved_models/lob/LOB/fqv6n5ls/ckpt/world_model_final.pth saved_models/lob/fi2010_studentt_norm_s0.json
echo "### KAGGLE BACKBONE ABLATION ###"
bash phase2_ablation.sh kaggle configs/kaggle_btc.yaml data data 3000 0.01 0 \
  saved_models/kaggle/LOB/yc3rhs5n/ckpt/world_model_final.pth saved_models/lob/kaggle_studentt_norm_s0.json
echo "### FI2010 BENCHMARK (deeplob/majority/linear_ar) ###"
cd /mnt/host/c/Users/ruuud/algoverse/Drama
python3 -m finmamba3.eval.benchmark_fi2010 --config configs/fi2010_studentt.yaml \
  --data-train data/fi2010/train --data-val data/fi2010/validation --horizon 10 \
  --baselines deeplob,majority,linear_ar --epochs-deeplob 20 --out reports/fi2010_benchmark.md 2>&1 | grep -E 'method|^\|' | tail -8
echo "ALL_PHASE2_DONE"

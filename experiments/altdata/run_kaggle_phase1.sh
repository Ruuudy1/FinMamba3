#!/bin/bash
# Chain the Kaggle gap evals and the predictability-supervised escalation so the GPU stays busy.
set -e
cd /mnt/host/c/tmp
sed -i 's/\r$//' eval_ab.sh kaggle_escalation.sh
echo "### KAGGLE GAP EVALS ###"
bash eval_ab.sh kaggle configs/kaggle_btc.yaml data 128 0.01
echo "### KAGGLE PREDICTABILITY-SUPERVISED ESCALATION ###"
bash kaggle_escalation.sh 3000 0
echo "ALL_KAGGLE_PHASE1_DONE"

#!/bin/bash
# ER-bucket decodability vs ER window length, on the frozen baseline features (Kaggle BTC + FI-2010).
cd /mnt/host/c/Users/ruuud/algoverse/Drama
cp /mnt/host/c/tmp/frozen_capacity_probe.py .
sed -i 's/\r$//' frozen_capacity_probe.py
for VW in 16 64; do
  echo "=== Kaggle BTC, vol_window=$VW ==="
  python3 frozen_capacity_probe.py --config configs/kaggle_btc.yaml --dataset kaggle \
    --checkpoint saved_models/kaggle/LOB/yc3rhs5n/ckpt/world_model_final.pth --data-val data \
    --norm-path saved_models/lob/kaggle_studentt_norm_s0.json --window-len 128 --num-windows 300 \
    --vol-window $VW 2>&1 | grep -iE 'frozen-feature|error' | tail -2
  echo "=== FI-2010, vol_window=$VW ==="
  python3 frozen_capacity_probe.py --config configs/fi2010_studentt.yaml --dataset fi2010 \
    --checkpoint saved_models/lob/LOB/fqv6n5ls/ckpt/world_model_final.pth --data-val data/fi2010/validation \
    --norm-path saved_models/lob/fi2010_studentt_norm_s0.json --window-len 128 --num-windows 300 \
    --vol-window $VW 2>&1 | grep -iE 'frozen-feature|error' | tail -2
done
rm -f frozen_capacity_probe.py
echo PROBE_WINDOWS_DONE

#!/bin/bash
# Probe matched-budget depths for the G2 backbones against Mamba-3 SISO (total 10,281,997 / seq 7,132,288).
cd /mnt/host/c/Users/ruuud/algoverse/Drama
cp /mnt/host/c/tmp/probe_params.py .
sed -i 's/\r$//' probe_params.py
declare -a OVS=(
  "--Models.WorldModel.Backbone Mamba --Models.WorldModel.Mamba.n_layer 5 --Models.WorldModel.Mamba.ssm_cfg.d_state 128"
  "--Models.WorldModel.Backbone Mamba --Models.WorldModel.Mamba.n_layer 6 --Models.WorldModel.Mamba.ssm_cfg.d_state 128"
  "--Models.WorldModel.Backbone Mamba2 --Models.WorldModel.Mamba.n_layer 5 --Models.WorldModel.Mamba.ssm_cfg.d_state 128"
  "--Models.WorldModel.Backbone Mamba2 --Models.WorldModel.Mamba.n_layer 6 --Models.WorldModel.Mamba.ssm_cfg.d_state 128"
  "--Models.WorldModel.Backbone Transformer --Models.WorldModel.Transformer.NumLayers 3"
  "--Models.WorldModel.Backbone Transformer --Models.WorldModel.Transformer.NumLayers 4"
  "--Models.WorldModel.Backbone Transformer --Models.WorldModel.Transformer.NumLayers 5"
)
for ov in "${OVS[@]}"; do
  echo -n "[$ov] -> "
  python3 probe_params.py configs/fi2010_studentt.yaml $ov 2>/dev/null | tail -1
done
rm -f probe_params.py

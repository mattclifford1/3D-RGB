#!/bin/sh
source ~/.3D_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

echo "============ DEPTH ESTIMATION ============"
python est_depth.py --config $1 --vid $2

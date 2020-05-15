#!/bin/sh
source ~/.3D_env/bin/activate
export CUDA_VISIBLE_DEVICES=0

echo "============ LDI ESTIMATION ============"
python est_ldi.py --config $1 --vid $2

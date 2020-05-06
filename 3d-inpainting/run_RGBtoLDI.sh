#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ DEPTH ESTIMATION ============"
python est_depth.py --config argument.yml
echo "============ LDI ESTIMATION ============"
python est_ldi.py --config argument.yml

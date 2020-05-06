#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ DEPTH ESTIMATION ============"
python est_depth.py --config argument.yml

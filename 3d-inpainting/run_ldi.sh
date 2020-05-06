#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ LDI ESTIMATION ============"
python est_ldi.py --config argument.yml

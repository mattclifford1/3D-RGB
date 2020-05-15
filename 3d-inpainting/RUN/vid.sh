#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ Making Video Sample ============"
python construct_vid.py --config $1 --vid $2

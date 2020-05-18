#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ VIDEO POSTPROCESSING ============"
python postprocess_vid.py --config $1 --vid $2

#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ VIDEO POSTPROCESSING ============"
python process-data/postprocess_vid.py --config $1 --vid $2

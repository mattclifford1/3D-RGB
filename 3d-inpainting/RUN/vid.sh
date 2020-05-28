#!/bin/sh
source ~/.3D_env/bin/activate
module load Mesa/11.2.1-foss-2016a

echo "============ Making Video Sample ============"
python construct_vid.py --config $1 --vid $2

echo "============ VIDEO POSTPROCESSING ============"
python process-data/postprocess_vid.py --config $1 --vid $2

#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ VIDEO PREPROCESSING ============"
python process-data/preprocess_vid.py --config configs/preprocess_vid.yml

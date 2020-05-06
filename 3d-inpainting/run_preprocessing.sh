#!/bin/sh
source ~/.3D_env/bin/activate

echo "============ VIDEO PREPROCESSING ============"
python preprocess_vid.py --config preprocess_vid.yml

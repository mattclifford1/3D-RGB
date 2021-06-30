#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

# place videos in nu-data/vids/pre #####

# extract frames
python process-data/preprocess_vid.py --config configs/preprocess_vid.yml

VID=underwater-4
# estimate depth
python est_depth.py --config configs/desk.yml --vid $VID

# estimate LDI
python est_ldi.py --config configs/desk.yml --vid $VID

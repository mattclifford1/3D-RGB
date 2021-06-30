#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

# place videos in vids/input
# $1 is the vid prefix (project folder) name of vid (without .mp4)

# extract frames
python process-data/preprocess_vid.py --config configs/desk.yml --vid $1

# estimate depth
python est_depth.py --config configs/desk.yml --vid $1

# estimate LDI
python est_ldi.py --config configs/desk.yml --vid $1

# contruct video
./run_vid $1

#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

python construct_frame.py --config $1 --im $2 --depth $3 --ldi $4 --video_dir $5 --frame_num $6 --vid_length $7

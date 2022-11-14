#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

python est_ldi_single.py --config $1 --im $2 --depth $3 --ldi $4

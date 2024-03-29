#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

echo "============ Making Video Sample ============"
python scripts/construct_vid.py --config configs/desk.yml --vid $1

echo "============ VIDEO POSTPROCESSING ============"
python process-data/postprocess_vid.py --config configs/desk.yml --vid $1

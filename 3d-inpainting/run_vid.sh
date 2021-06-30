#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $VENV

VID=underwater-4
echo "============ Making Video Sample ============"
python construct_vid.py --config configs/desk.yml --vid $VID

echo "============ VIDEO POSTPROCESSING ============"
python process-data/postprocess_vid.py --config configs/desk.yml --vid $VID

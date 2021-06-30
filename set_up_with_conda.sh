#!/bin/bash
VENV=3DP
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda create -n $VENV python=3.7
conda activate $VENV

pip install --upgrade pip
pip install -r requirements_conda.txt
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit==10.1.243 -c pytorch

echo "============ Download Models ============"
chmod +x download.sh              # allow execution
./download.sh                     # download


# # sed is install from apt in google colab
# brew install gnu-sed                  # OSX install
# sed -i 's/offscreen_rendering: True/offscreen_rendering: False/g' argument.yml

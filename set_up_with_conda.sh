#!/bin/sh

# make conda virtual environment
echo "============ Setting up Python ============"
conda create -n 3DP python=3.7 anaconda
conda activate 3DP
pip install --upgrade pip
pip install -r requirements_conda.txt
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit==10.1.243 -c pytorch

echo "============ Download Models ============"
chmod +x download.sh              # allow execution
./download.sh                     # download


# # sed is install from apt in google colab
# brew install gnu-sed                  # OSX install
# sed -i 's/offscreen_rendering: True/offscreen_rendering: False/g' argument.yml

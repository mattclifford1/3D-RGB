#!/bin/sh

# make python3 virtual environment
echo "============ Setting up Python ============"
python3.7 -m venv ~/.3D_env         # make
source ~/.3D_env/bin/activate     # activate
pip --user install --upgrade pip
pip --user install -r requirements.txt  # install python dependancies

echo "============ Download Models ============"
chmod +x download.sh              # allow execution
./download.sh                     # download


# # sed is install from apt in google colab
# brew install gnu-sed                  # OSX install
# sed -i 's/offscreen_rendering: True/offscreen_rendering: False/g' argument.yml

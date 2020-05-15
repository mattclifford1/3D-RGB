#!/bin/bash
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
# SBATCH --account=comsm0018       # use the course account

module load languages/anaconda3/2019.07-3.6.5-tflow-1.14
# you might want to install some modules (only need to do this once as --user saves to own envirnomnet)
# pip install --user sklearn

srun $*    # run all command line inputs
wait

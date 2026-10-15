#!/bin/bash

#SBATCH --job-name=build
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH -p zen5_2304_b200x8
#SBATCH --qos zen5_2304_b200x8
#SBATCH --time=00:10:00

module load Miniforge3
module load CUDA/13.0.0

eval "$(conda shell.bash hook)"

conda activate $DATA/envs/AVR

which pip
pip install . --no-build-isolation

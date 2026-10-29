#!/bin/bash

#SBATCH --job-name=build-dwn
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --cpus-per-task 8
#SBATCH --time=00:10:00

module load Miniforge3
module load CUDA/13.0.0

eval "$(conda shell.bash hook)"

# activate the environment
conda activate $DATA/envs/AVR
pip install . --no-build-isolation

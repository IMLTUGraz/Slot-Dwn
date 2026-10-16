#!/bin/bash

#SBATCH --job-name=build
#SBATCH --gres=gpu:
#SBATCH --cpus-per-task 8
#SBATCH --time=00:30:00

module load Miniforge3
module load CUDA/13.0.0

eval "$(conda shell.bash hook)"

# activate the environment
conda activate AVR

# Build torch-dwn locally
pip install . --no-build-isolation

#!/bin/bash

#SBATCH --job-name=datagen
#SBATCH --output=log-%j.o
#SBATCH --error=log-%j.e
#SBATCH --cpus-per-task 8
#SBATCH --mem 4GB
#SBATCH --time=00:30:00

module load Miniforge3
eval "$(conda shell.bash hook)"

conda activate $DATA/envs/AVR

python generate_grid_choice.py --root $SCRATCH --folder deepiq_g_train --num-samples 5000 --num-relations 1 --split train
python generate_grid_choice.py --root $SCRATCH --folder deepiq_g_train --num-samples 5000 --num-relations 2 --split train
python generate_grid_choice.py --root $SCRATCH --folder deepiq_g_train --num-samples 5000 --num-relations 3 --split train
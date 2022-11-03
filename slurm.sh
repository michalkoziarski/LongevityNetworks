#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

conda activate sg

python -W ignore ${1} ${@:2}

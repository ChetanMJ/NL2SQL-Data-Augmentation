#!/bin/bash

#SBATCH -n 1 # Number of cores

#SBATCH -N 1 # Ensure that all cores are on one machine

#SBATCH --mem=16G # Memory - Use up to 8G

#SBATCH --time=0 # No time limit

#SBATCH --error=sampler.err

#SBATCH --output=sampler.out

python -u sampler.py

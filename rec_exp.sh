#!/bin/bash
#SBATCH --job-name=rec_exp
#SBATCH --account=fc_basics
#SBATCH --partition=savio2
#SBATCH --time=06:00:00
#SBATCH --output=slurm_outputs/rna_exp_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erginbas@berkeley.edu
which python
python capacity_aware_recommender_experiments.py $1
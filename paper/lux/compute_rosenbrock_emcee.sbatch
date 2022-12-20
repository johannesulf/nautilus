#!/bin/bash
#SBATCH --partition=leauthaud
#SBATCH --account=leauthaud
#SBATCH --job-name=compute_rosenbrock-10_emcee
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/compute_rosenbrock-10_emcee.out

source init.sh
python compute.py rosenbrock-10 emcee --verbose --full


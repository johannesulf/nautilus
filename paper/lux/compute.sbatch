#!/bin/bash
#SBATCH --partition=leauthaud
#SBATCH --account=leauthaud
#SBATCH --job-name=compute_LIKELIHOOD
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/compute_LIKELIHOOD.out

source init.sh
cd ..
for i in {0..39}; do
  OMP_NUM_THREADS=1 python compute.py $ARGS --n_run=5 &
done

wait

# example usage: ARGS="loggamma-30 dynesty-r"; STR=$(echo "$ARGS" | tr ' ' _); sbatch --job-name=compute_$STR --output=log/compute_$STR.out --export=ARGS="$ARGS" compute.sbatch

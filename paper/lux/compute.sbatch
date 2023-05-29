#!/bin/bash
#SBATCH --partition=leauthaud
#SBATCH --account=leauthaud
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jolange@ucsc.edu

source init.sh
cd ..
for i in {0..39}; do
  OMP_NUM_THREADS=1 python compute.py $ARGS &
  sleep 1
done

wait

# example usage: ARGS="loggamma-30 dynesty-r"; STR=$(echo "$ARGS" | tr ' ' _); sbatch --job-name=$STR --output=log/$STR.out --export=ARGS="$ARGS" compute.sbatch

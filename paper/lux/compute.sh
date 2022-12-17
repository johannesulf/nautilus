#!/bin/bash

TEMPLATE=$'#!/bin/bash
#SBATCH --partition=leauthaud
#SBATCH --account=leauthaud
#SBATCH --job-name=compute_LIKELIHOOD
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/compute_LIKELIHOOD.out

source init.sh
cd ..
for i in {0..39}; do
  OMP_NUM_THREADS=1 python compute.py LIKELIHOOD --n_run=5 &
done

wait
'

if [ -z ${1} ]; then
  echo "Please specify the likelihood problem."
  return 1
fi

SCRIPT="${TEMPLATE//LIKELIHOOD/$1}"
SCRIPT_NAME="compute_${1}.sh"
echo "$SCRIPT" > $SCRIPT_NAME
sbatch $SCRIPT_NAME
rm $SCRIPT_NAME

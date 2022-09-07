module load python
module load openmpi
module load hdf5

source env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/data/groups/leauthaud/jolange/Nautilus/"
export SPS_HOME="/data/groups/leauthaud/jolange/Nautilus/paper/fsps"

export HDF5_USE_FILE_LOCKING='FALSE'

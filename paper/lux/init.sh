module load python/3.8.6
module load hdf5

source env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/data/groups/leauthaud/jolange/Nautilus/"
export HDF5_USE_FILE_LOCKING='FALSE'

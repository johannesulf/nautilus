module load python
module load hdf5

source env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/data/groups/leauthaud/jolange/Nautilus/"

export HDF5_USE_FILE_LOCKING='FALSE'

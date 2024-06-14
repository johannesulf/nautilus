import numpy as np
import multiprocessing
import os
import pytest
import time

from multiprocessing import Pool
from dask.distributed import Client

from nautilus import Sampler
from nautilus.pool import NautilusPool


def prior(x):
    return x


def likelihood(x):
    time.sleep(0.001)
    return 1, os.getpid()


@pytest.mark.skipif(multiprocessing.get_start_method() == 'spawn',
                    reason=('pytest does not support spawning'))
@pytest.mark.parametrize("pool", [None, 1, 2, (1, 4), (4, 1), 'mp', 'dask'])
def test_pool(pool):
    # Test that the expected number of processes are run.

    if isinstance(pool, tuple):
        n_jobs = pool[0]
    else:
        if pool not in ['mp', 'dask']:
            n_jobs = 1 if pool is None else pool
        elif pool == 'mp':
            n_jobs = 4
            pool = Pool(n_jobs)
        else:
            pool = Client()
            n_jobs = NautilusPool(pool).size

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=50, n_networks=1,
                      pool=pool)
    sampler.run(f_live=1.0, n_eff=0)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    assert len(np.unique(blobs)) == n_jobs
    assert sampler.n_batch >= 100
    assert (sampler.n_batch % n_jobs) == 0

    try:
        pool.close()
    except AttributeError:
        pass

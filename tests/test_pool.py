import multiprocessing
import numpy as np
import os
import pytest
import time

from multiprocessing import Pool

from nautilus import Sampler


def prior(x):
    return x


def likelihood(x):
    time.sleep(0.001)
    return 1, os.getpid()


@pytest.mark.skipif(multiprocessing.get_start_method() == 'spawn',
                    reason=('pytest does not support spawning'))
@pytest.mark.parametrize("pool", [1, 2, Pool(2), Pool(3), None,
                                  (2, 2), (None, 1), (2, Pool(2))])
def test_pool(pool):
    # Test that the expected number of processes are run.

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=50, n_networks=1,
                      pool=pool)
    sampler.run(f_live=1.0, n_eff=0)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    if isinstance(pool, tuple):
        pool = pool[0]

    if isinstance(pool, int):
        n_jobs = pool
    elif pool is None:
        n_jobs = 1
    else:
        n_jobs = pool._processes

    assert len(np.unique(blobs)) == n_jobs

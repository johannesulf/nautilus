import numpy as np
import os
import pytest

from multiprocessing import Pool
from nautilus import Sampler


def prior(x):
    return x


def likelihood(x):
    return 1, os.getpid()


@pytest.mark.parametrize("pool", [1, 2, Pool(2), Pool(3)])
def test_pool(pool):
    # Test that the expected number of processes are run.

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=10, pool=pool)
    sampler.run(f_live=1.0, n_eff=0)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    if isinstance(pool, int):
        assert len(np.unique(blobs)) == pool
    else:
        assert len(np.unique(blobs)) == pool._processes

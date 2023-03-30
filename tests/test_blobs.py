import numpy as np
import pytest

from nautilus import Sampler


@pytest.fixture
def prior():
    def prior(x):
        return x
    return prior


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("discard_exploration", [True, False])
def test_blobs_single(prior, dtype, vectorized, discard_exploration):
    # Test that blobs work and their data types and values match.

    def likelihood(x):
        if vectorized:
            return np.ones(len(x)), (10 * x[:, 0]).astype(dtype)
        else:
            return 1, dtype(10 * x[0])

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=10,
                      vectorized=vectorized, n_jobs=1)
    sampler.run(f_live=1.0, n_eff=200, discard_exploration=discard_exploration)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    assert len(points) == len(blobs)
    assert blobs.dtype == dtype
    assert np.all((10 * points[:, 0]).astype(dtype) == blobs)


@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("discard_exploration", [True, False])
def test_blobs_multi(prior, vectorized, discard_exploration):
    # Test that blobs work and their data types and values match.

    def likelihood(x):
        if vectorized:
            return (np.ones(len(x)), x[:, 0].astype(np.float32),
                    x[:, 1].astype(np.float32))
        else:
            return 1, np.float32(x[0]), np.float32(x[1])

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=10,
                      vectorized=vectorized, n_jobs=1)
    sampler.run(f_live=1.0, n_eff=200, discard_exploration=discard_exploration)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    assert len(points) == len(blobs)
    assert blobs['blob_0'].dtype == np.float32
    assert blobs['blob_1'].dtype == np.float32
    assert np.all(points[:, 0].astype(np.float32) == blobs['blob_0'])
    assert np.all(points[:, 1].astype(np.float32) == blobs['blob_1'])


@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("discard_exploration", [True, False])
def test_blobs_dtype(prior, vectorized, discard_exploration):
    # Test that blobs work and their data types and values match.

    def likelihood(x):
        if vectorized:
            return np.ones(len(x)), x[:, 0], x[:, 1]
        else:
            return 1, x[0], x[1]

    blobs_dtype = [('a', '|S10'), ('b', np.int16)]
    sampler = Sampler(prior, likelihood, n_dim=2, n_live=10,
                      vectorized=vectorized, n_jobs=1, blobs_dtype=blobs_dtype)
    sampler.run(f_live=1.0, n_eff=200, discard_exploration=discard_exploration)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    assert len(points) == len(blobs)
    assert blobs['a'].dtype == blobs_dtype[0][1]
    assert blobs['b'].dtype == blobs_dtype[1][1]
    assert np.all(points[:, 0].astype(blobs_dtype[0][1]) == blobs['a'])
    assert np.all(points[:, 1].astype(blobs_dtype[1][1]) == blobs['b'])


@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("discard_exploration", [True, False])
def test_blobs_single_dtype(prior, vectorized, discard_exploration):
    # Test that blobs work even when returning specyfing a single dtype like
    # CosmoSIS does.

    def likelihood(x):
        if vectorized:
            return np.ones(len(x)), x[:, 0], x[:, 1]
        else:
            return 1, x[0], x[1]

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=10,
                      vectorized=vectorized, n_jobs=1, blobs_dtype=float)
    sampler.run(f_live=1.0, n_eff=200, discard_exploration=discard_exploration)
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)
    assert len(points) == len(blobs)
    assert np.all(points[:, 0] == blobs[:, 0])
    assert np.all(points[:, 1] == blobs[:, 1])

import h5py
import numpy as np
import pytest

from pathlib import Path

from nautilus import Sampler
from nautilus.bounds import (UnitCube, Ellipsoid, MultiEllipsoid, NeuralBound,
                             NautilusBound)
from nautilus.neural import NeuralNetworkEmulator


@pytest.fixture
def h5py_group():
    with h5py.File('test.hdf5', 'w') as file:
        yield file.create_group('test')
    Path('test.hdf5').unlink()


def test_neural_io(h5py_group):
    # Test that we can write and read a neural network emulator correctly.

    points = np.random.random((100, 2))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    emulator_write = NeuralNetworkEmulator.train(points, log_l)
    emulator_write.write(h5py_group)
    emulator_read = NeuralNetworkEmulator.read(h5py_group)
    assert np.all(emulator_write.predict(points) ==
                  emulator_read.predict(points))


@pytest.mark.parametrize("bound_class", [UnitCube, Ellipsoid, MultiEllipsoid,
                                         NeuralBound, NautilusBound])
@pytest.mark.parametrize("random_state_sync", [True, False])
def test_bounds_io(h5py_group, bound_class, random_state_sync):
    # Test that we can write and read a bound correctly. In particular, also
    # test that the random state is correctly set after writing and reading.

    n_dim = 5
    n_points = 100
    np.random.seed(0)
    points = np.random.random((n_points, n_dim))

    if bound_class == UnitCube:
        args = (n_dim, )
    elif bound_class in [Ellipsoid, MultiEllipsoid]:
        args = (points, )
    else:
        log_l = -np.linalg.norm(points - 0.5, axis=1)
        log_l_min = np.median(log_l)
        if bound_class == NeuralBound:
            args = (points, log_l, log_l_min)
        else:
            args = (points, log_l, log_l_min, np.log(0.5))

    bound_write = bound_class.compute(*args)
    if bound_class == MultiEllipsoid:
        bound_write.split_ellipsoid()

    bound_write.write(h5py_group)
    if random_state_sync:
        random_state = np.random.RandomState()
        random_state.set_state(bound_write.random_state.get_state())
    else:
        random_state = None
    bound_read = bound_class.read(h5py_group, random_state=random_state)

    if bound_class != NeuralBound:
        assert (np.all(bound_write.sample(10000) == bound_read.sample(10000))
                == random_state_sync)

    if ((random_state_sync or bound_class in [UnitCube, Ellipsoid]) and not
            bound_class == NeuralBound):
        assert bound_write.volume() == bound_read.volume()

    points = np.random.random((10000, n_dim))
    assert np.all(bound_write.contains(points) == bound_read.contains(points))


@pytest.mark.parametrize("blobs", [True, False])
def test_sampler_io(blobs):
    # Test that we can write and read a sampler correctly. In particular, also
    # test that the random state is correctly set after writing and reading.

    def prior(x):
        return x

    def likelihood(x):
        if blobs:
            return -np.linalg.norm(x - 0.5) * 0.001, x[0]
        else:
            return -np.linalg.norm(x - 0.5) * 0.001

    sampler_write = Sampler(prior, likelihood, n_dim=2, n_live=100,
                            filepath='test.hdf5', resume=False)
    sampler_write.run(f_live=0.45, n_eff=0)
    sampler_read = Sampler(prior, likelihood, n_dim=2, n_live=100,
                           filepath='test.hdf5', resume=True)

    sampler_write.run(n_eff=1000)
    sampler_read.run(n_eff=1000)

    posterior_write = sampler_write.posterior()
    posterior_read = sampler_read.posterior()

    for arr_write, arr_read in zip(posterior_write, posterior_read):
        assert np.all(arr_write == arr_read)

    assert sampler_write.evidence() == sampler_read.evidence()

    Path('test.hdf5').unlink()

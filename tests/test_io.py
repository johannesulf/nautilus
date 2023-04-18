import h5py
import numpy as np
import pytest

from pathlib import Path

from nautilus import Sampler
from nautilus.bounds import (
    UnitCube, Ellipsoid, Union, UnitCubeEllipsoidMixture, NeuralBound,
    NautilusBound)
from nautilus.neural import NeuralNetworkEmulator


@pytest.fixture
def h5py_group():
    with h5py.File('test.hdf5', 'w') as filepath:
        yield filepath.create_group('test')
    Path('test.hdf5').unlink()


def test_neural_io(h5py_group):
    # Test that we can write and read a neural network emulator correctly.

    points = np.random.random((100, 2))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    emulator_write = NeuralNetworkEmulator.train(points, log_l, n_networks=1,
                                                 pool=None)
    emulator_write.write(h5py_group)
    emulator_read = NeuralNetworkEmulator.read(h5py_group)
    assert np.all(emulator_write.predict(points) ==
                  emulator_read.predict(points))


@pytest.mark.parametrize("bound_class", [UnitCube, Ellipsoid, Union,
                                         UnitCubeEllipsoidMixture, NeuralBound,
                                         NautilusBound])
@pytest.mark.parametrize("rng_sync", [True, False])
def test_bounds_io(h5py_group, bound_class, rng_sync):
    # Test that we can write and read a bound correctly. In particular, also
    # test that the random number generator is correctly set after writing and
    # reading.

    n_dim = 5
    n_points = 100
    np.random.seed(0)
    points = np.random.random((n_points, n_dim))
    rng = np.random.default_rng(0)

    if bound_class == UnitCube:
        args = (n_dim, )
        kwargs = dict()
    elif bound_class in [Ellipsoid, Union, UnitCubeEllipsoidMixture]:
        args = (points, )
        kwargs = dict()
    else:
        log_l = -np.linalg.norm(points - 0.5, axis=1)
        log_l_min = np.median(log_l)
        if bound_class == NeuralBound:
            args = (points, log_l, log_l_min)
        else:
            args = (points, log_l, log_l_min, np.log(0.5))
        kwargs = dict(n_networks=1, pool=None)

    bound_write = bound_class.compute(*args, **kwargs, rng=rng)
    if bound_class == Union:
        bound_write.split_bound()

    bound_write.write(h5py_group)
    rng = np.random.default_rng(1)
    if rng_sync:
        if bound_class == NeuralBound:
            rng = None
        elif bound_class == UnitCubeEllipsoidMixture:
            rng.bit_generator.state = bound_write.cube.rng.bit_generator.state
        else:
            rng.bit_generator.state = bound_write.rng.bit_generator.state

    bound_read = bound_class.read(h5py_group, rng=rng)

    if bound_class != NeuralBound:
        assert (np.all(bound_write.sample(10000) == bound_read.sample(10000))
                == rng_sync)

    if ((rng_sync or bound_class in [UnitCube, Ellipsoid]) and not
            bound_class == NeuralBound):
        assert bound_write.volume() == bound_read.volume()

    points = np.random.random((10000, n_dim))
    assert np.all(bound_write.contains(points) == bound_read.contains(points))


@pytest.mark.parametrize("blobs", [True, False])
@pytest.mark.parametrize("discard_exploration", [True, False])
@pytest.mark.parametrize("n_networks", [0, 1, 2])
def test_sampler_io(blobs, discard_exploration, n_networks):
    # Test that we can write and read a sampler correctly. In particular, also
    # test that the random number generator is correctly set after writing and
    # reading. Also make sure that the sampler can print out the progress.

    def prior(x):
        return x

    def likelihood(x):
        if blobs:
            return -np.linalg.norm(x - 0.5) * 0.001, x[0]
        else:
            return -np.linalg.norm(x - 0.5) * 0.001

    sampler_write = Sampler(prior, likelihood, n_dim=2, n_live=100,
                            n_networks=n_networks, n_jobs=1,
                            filepath='test.hdf5', resume=False, seed=0)
    sampler_write.run(f_live=0.45, n_eff=0, verbose=True)
    sampler_write.explored = False
    sampler_read = Sampler(prior, likelihood, n_dim=2, n_live=100,
                           n_networks=n_networks, n_jobs=1,
                           filepath='test.hdf5', resume=True)
    sampler_read.explored = False

    sampler_write.run(f_live=0.45, n_eff=1000,
                      discard_exploration=discard_exploration, verbose=True)
    sampler_read.run(f_live=0.45, n_eff=1000,
                     discard_exploration=discard_exploration, verbose=True)

    posterior_write = sampler_write.posterior()
    posterior_read = sampler_read.posterior()

    for arr_write, arr_read in zip(posterior_write, posterior_read):
        assert np.all(arr_write == arr_read)

    assert sampler_write.evidence() == sampler_read.evidence()

    Path('test.hdf5').unlink()


def test_sampler_exploration_io():
    # Test that the sampler correctly creates a backup of the state after the
    # exploration stage.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5) * 0.001

    sampler = Sampler(prior, likelihood, n_dim=2, n_live=100,
                      n_networks=1, n_jobs=1, filepath='test.hdf5',
                      resume=False, seed=0)
    sampler.run(n_eff=1000, discard_exploration=True)

    assert Path('test.hdf5').is_file()
    assert Path('test_exp.hdf5').is_file()

    Path('test.hdf5').unlink()
    Path('test_exp.hdf5').unlink()

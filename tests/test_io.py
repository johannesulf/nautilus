import h5py
import numpy as np
import pytest

from nautilus.bounds import UnitCube
from nautilus.neural import NeuralNetworkEmulator
from pathlib import Path


@pytest.fixture
def h5py_group():
    with h5py.File('test.hdf5', 'w') as file:
        yield file.create_group('test')
    Path('test.hdf5').unlink()


def test_neural_io(h5py_group):
    points = np.random.random((100, 2))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    emulator_write = NeuralNetworkEmulator.train(points, log_l)
    emulator_write.write(h5py_group)
    emulator_read = NeuralNetworkEmulator.read(h5py_group)
    assert np.all(emulator_write.predict(points) ==
                  emulator_read.predict(points))


@pytest.mark.parametrize("random_state_sync", [True, False])
def test_unit_cube_io(h5py_group, random_state_sync):
    n_dim = 5
    cube_write = UnitCube.compute(n_dim)
    cube_write.write(h5py_group)
    if random_state_sync:
        random_state = np.random.RandomState()
        random_state.set_state(cube_write.random_state.get_state())
    else:
        random_state = None
    cube_read = UnitCube.read(h5py_group, random_state=random_state)
    assert (np.all(cube_write.sample(100) == cube_read.sample(100)) ==
            random_state_sync)

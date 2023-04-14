import numpy as np

from nautilus import neural


def test_neural_network_emulator():
    # Test that the emulator works and produces decent estimates.

    np.random.seed(0)
    n_dim, n_points = 5, 1000
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = neural.NeuralNetworkEmulator.train(x, y, n_networks=1, pool=None)
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)

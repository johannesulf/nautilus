import numpy as np
from nautilus import neural


def test_neural_network_emulator():
    np.random.seed(0)
    n_dim, n_points = 7, 300
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    neural_network_kwargs = {
        'alpha': 0, 'tol': 0, 'max_iter': 1000, 'n_iter_no_change': 50,
        'hidden_layer_sizes': (128, 128, 128)}
    emu = neural.NeuralNetworkEmulator(
        x, y, neural_network_kwargs=neural_network_kwargs)
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.1 * np.std(y)

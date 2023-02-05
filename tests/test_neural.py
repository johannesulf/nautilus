import numpy as np

from nautilus import neural


def test_neural_network_emulator():
    # Test that the emulator works and produces decent estimates.

    np.random.seed(0)
    n_dim, n_points = 5, 1000
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    neural_network_kwargs = {
        'hidden_layer_sizes': (100, 50, 20), 'alpha': 0,
        'learning_rate_init': 1e-2, 'max_iter': 10000,
        'random_state': 0, 'tol': 1e-4, 'n_iter_no_change': 20}
    emu = neural.NeuralNetworkEmulator.train(
        x, y, neural_network_kwargs=neural_network_kwargs)
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)

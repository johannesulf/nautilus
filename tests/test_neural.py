import numpy as np
import pytest

from nautilus import neural


@pytest.mark.parametrize("activation", ['relu', 'silu'])
def test_neural_network_emulator(activation):
    # Test that the emulator works and produces decent estimates.

    np.random.seed(0)
    n_dim, n_points = 5, 1000
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = neural.NeuralNetworkEmulator.train(
        x, y, n_networks=1, neural_network_kwargs=dict(activation=activation),
        pool=None)
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)


def test_silu_gradient():
    # Test the training gradient of SiLU networks against finite differences.
    # One gradient descent step with a learning rate of 1 changes the weights
    # by minus the gradient of the loss, mean((y - y_pred)^2) / 2.

    np.random.seed(0)
    x = np.random.normal(size=(50, 3))
    y = np.random.normal(size=50)
    net = neural.SiLUMLPRegressor(
        hidden_layer_sizes=(6, 4), activation='silu', solver='sgd',
        learning_rate_init=1, momentum=0, batch_size=len(x), alpha=0,
        shuffle=False, random_state=0)
    net.partial_fit(x, y)
    coefs = [coef.copy() for coef in net.coefs_]
    intercepts = [intercept.copy() for intercept in net.intercepts_]
    net.partial_fit(x, y)
    grad = coefs[0] - net.coefs_[0]
    net.intercepts_ = intercepts

    def loss(coef):
        net.coefs_ = [coef] + coefs[1:]
        return np.mean((y - net.predict(x))**2) / 2

    eps = 1e-6
    grad_num = np.zeros_like(grad)
    for i in np.ndindex(grad.shape):
        step = np.zeros_like(grad)
        step[i] = eps
        grad_num[i] = (loss(coefs[0] + step) - loss(coefs[0] - step)) / (
            2 * eps)
    assert np.allclose(grad, grad_num, rtol=1e-5, atol=1e-8)

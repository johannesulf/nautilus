"""Module implementing neural network emulators."""

import threading
import warnings
from functools import partial

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES
from sklearn.utils.extmath import safe_sparse_dot
from threadpoolctl import threadpool_limits

try:  # scikit-learn >= 1.2 validates parameters with these constraints
    from sklearn.utils._param_validation import StrOptions
except ImportError:
    StrOptions = None

# Per thread, sigmoid(x) of the SiLU hidden layers from the latest forward
# pass in training, keyed by the id of the layer output.
_silu_state = threading.local()


def inplace_silu(x):
    """Apply the SiLU activation, x * sigmoid(x), in place.

    Computing sigmoid(x) as 1 / (1 + exp(-x)) is faster than using
    scipy.special.expit.

    Parameters
    ----------
    x : numpy.ndarray
        Input values, overwritten with the output.

    Returns
    -------
    s : numpy.ndarray
        sigmoid(x), which the backward pass of training needs.

    """
    s = np.negative(x)
    with np.errstate(over='ignore'):  # exp(-x) = inf gives the right limit, 0
        np.exp(s, out=s)
    s += 1
    np.reciprocal(s, out=s)
    x *= s
    return s


def inplace_silu_derivative(z, delta):
    """Multiply the back-propagated error by the SiLU derivative in place.

    With s = sigmoid(x) and z = x * s, the derivative is s + z * (1 - s).
    scikit-learn passes only the layer output z, from which x cannot be
    recovered, so s is taken from the forward pass.

    Parameters
    ----------
    z : numpy.ndarray
        Output of the hidden layer.
    delta : numpy.ndarray
        Back-propagated error, modified in place.

    """
    # Each layer is visited once per backward pass, so s can be freed.
    s = _silu_state.sigmoid.pop(id(z))
    delta *= s + z * (1 - s)


# Register SiLU with scikit-learn's activation tables. Only SiLUMLPRegressor
# can be trained with it.
ACTIVATIONS['silu'] = inplace_silu
DERIVATIVES['silu'] = inplace_silu_derivative


class SiLUMLPRegressor(MLPRegressor):
    """MLPRegressor that also supports SiLU hidden activations.

    scikit-learn's MLPRegressor does not implement the SiLU (swish)
    activation, x * sigmoid(x). This subclass adds ``activation='silu'``.
    All other activations behave exactly as in MLPRegressor. It relies on
    private parts of scikit-learn: the activation tables and the
    ``_forward_pass`` and ``_backprop`` methods of MLPRegressor.

    """

    if StrOptions is not None:
        _parameter_constraints = MLPRegressor._parameter_constraints | {
            'activation': [StrOptions(
                {'identity', 'logistic', 'tanh', 'relu', 'silu'})]}

    def _forward_pass(self, activations):
        """Perform a forward pass, keeping sigmoid(x) of SiLU layers.

        Parameters
        ----------
        activations : list
            The ith element holds the values of the ith layer.

        Returns
        -------
        activations : list
            The ith element holds the values of the ith layer.

        """
        if self.activation != 'silu':
            return super()._forward_pass(activations)

        _silu_state.sigmoid = {}
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(
                activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]
            if i + 1 != self.n_layers_ - 1:
                _silu_state.sigmoid[id(activations[i + 1])] = inplace_silu(
                    activations[i + 1])
        ACTIVATIONS[self.out_activation_](activations[-1])

        return activations


def network_class(activation):
    """Return the network class for a hidden-layer activation.

    Parameters
    ----------
    activation : str
        Hidden-layer activation.

    Returns
    -------
    network_class : type
        SiLUMLPRegressor for 'silu', MLPRegressor otherwise.

    """
    return SiLUMLPRegressor if activation == 'silu' else MLPRegressor


@threadpool_limits.wrap(limits=1)
def train_network(x, y, neural_network_kwargs):
    """Train a network.

    Parameters
    ----------
    x : numpy.ndarray
        Input coordinates.
    y : numpy.ndarray
        Target values.
    neural_network_kwargs : dict
        Keyword arguments passed to the constructor of MLPRegressor, or of
        SiLUMLPRegressor if the activation is 'silu'.

    Returns
    -------
    network : MLPRegressor
        The trained network.

    """
    return network_class(neural_network_kwargs.get('activation'))(
        **neural_network_kwargs).fit(x, y)


class NeuralNetworkEmulator:
    """Likelihood neural network emulator.

    Attributes
    ----------
    mean : numpy.ndarray
        Mean of input coordinates used for normalizing coordinates.
    scale : numpy.ndarray
        Standard deviation of input coordinates used for normalizing
        coordinates.
    network : sklearn.neural_network.MLPRegressor
        Artifical neural network used for emulation.

    """

    @classmethod
    def train(cls, x, y, n_networks=4, neural_network_kwargs=None, pool=None):
        """Initialize and train the likelihood neural network emulator.

        Parameters
        ----------
        x : numpy.ndarray
            Input coordinates.
        y : numpy.ndarray
            Target values.
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict, list of dict or None, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor. A list gives one dictionary per network. An
            activation of 'silu' selects SiLUMLPRegressor. Default is `None`.
        pool : multiprocessing.Pool, optional
            Pool used for parallel processing.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()

        emulator.mean = np.mean(x, axis=0)
        emulator.scale = np.std(x, axis=0)

        if neural_network_kwargs is None:
            neural_network_kwargs = {}

        if isinstance(neural_network_kwargs, dict):
            neural_network_kwargs = [neural_network_kwargs] * n_networks
        elif len(neural_network_kwargs) != n_networks:
            raise ValueError("If 'neural_network_kwargs' is a list, it must " +
                             "have one dictionary per network.")

        if any('random_state' in kwargs for kwargs in neural_network_kwargs):
            warnings.warn("The 'random_state' keyword argument passed to the" +
                          " neural network is ignored.", Warning, stacklevel=2)

        default_kwargs = dict(  # noqa: C408
            hidden_layer_sizes=(100, 50, 20), alpha=0, learning_rate_init=1e-2,
            max_iter=10000, tol=0, n_iter_no_change=10)
        kwargs_list = [default_kwargs | kwargs | {'random_state': i}
                       for i, kwargs in enumerate(neural_network_kwargs)]

        f = partial(train_network, (x - emulator.mean) / emulator.scale, y)

        if pool is None:
            emulator.neural_networks = list(map(f, kwargs_list))
        else:
            emulator.neural_networks = list(pool.map(f, kwargs_list))

        return emulator

    def predict(self, x):
        """Calculate the emulator likelihood prediction for a group of points.

        Parameters
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.

        Returns
        -------
        y_emu : numpy.ndarray
            Emulated normalized likelihood value of the training points.

        """
        return np.mean(
            [network.predict((x - self.mean) / self.scale) for network in
             self.neural_networks], axis=0)

    def write(self, group):
        """Write the emulator to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_networks'] = len(self.neural_networks)

        for i, network in enumerate(self.neural_networks):

            for key in network.__dict__:
                if key in ['coefs_', 'intercepts_']:
                    continue
                try:
                    group.attrs[key + f'_{i}'] = getattr(network, key)
                except (TypeError, ValueError):
                    pass

            for k in range(network.n_layers_ - 1):
                group.create_dataset(f'coefs_{k}_{i}', data=network.coefs_[k])
                group.create_dataset(
                    f'intercepts_{k}_{i}', data=network.intercepts_[k])

        group.create_dataset('mean', data=self.mean)
        group.create_dataset('scale', data=self.scale)

    @classmethod
    def read(cls, group):
        """Read the emulator from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()

        emulator.mean = np.array(group['mean'])
        emulator.scale = np.array(group['scale'])

        emulator.neural_networks = []

        for i in range(group.attrs['n_networks']):

            network = network_class(group.attrs.get(f'activation_{i}'))()

            for key in group.attrs:
                if key.rsplit('_', 1)[1] == f'{i}':
                    setattr(network, key.rsplit('_', 1)[0], group.attrs[key])

            network.coefs_ = [
                np.array(group[f'coefs_{k}_{i}']) for k in
                range(network.n_layers_ - 1)]
            network.intercepts_ = [
                np.array(group[f'intercepts_{k}_{i}']) for k in
                range(network.n_layers_ - 1)]

            emulator.neural_networks.append(network)

        return emulator

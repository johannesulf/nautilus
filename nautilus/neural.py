"""Module implementing neural network emulators."""

import numpy as np
import warnings
from functools import partial
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=1)
def train_network(x, y, neural_network_kwargs, random_state):
    """Train a network.

    Parameters
    ----------
    x : numpy.ndarray
        Input coordinates.
    y : numpy.ndarray
        Target values.
    neural_network_kwargs : dict
        Keyword arguments passed to the constructor of MLPRegressor.
    random_state : int
        Determines random number generation.

    Returns
    -------
    network : MLPRegressor
        The trained network.

    """
    return MLPRegressor(
        random_state=random_state, **neural_network_kwargs).fit(x, y)


class NeuralNetworkEmulator():
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
    def train(cls, x, y, n_networks=4, neural_network_kwargs={}, pool=None):
        """Initialize and train the likelihood neural network emulator.

        Parameters
        ----------
        x : numpy.ndarray
            Input coordinates.
        y : numpy.ndarray
            Target values.
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor.
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

        default_neural_network_kwargs = dict(
            hidden_layer_sizes=(100, 50, 20), alpha=0, learning_rate_init=1e-2,
            max_iter=10000, tol=0, n_iter_no_change=10)
        default_neural_network_kwargs.update(neural_network_kwargs)
        neural_network_kwargs = default_neural_network_kwargs

        if 'random_state' in neural_network_kwargs:
            warnings.warn("The 'random_state' keyword argument passed to the" +
                          " neural network is ignored.", Warning, stacklevel=2)
            del neural_network_kwargs['random_state']

        f = partial(train_network, (x - emulator.mean) / emulator.scale, y,
                    neural_network_kwargs)

        if pool is None:
            emulator.neural_networks = list(map(f, range(n_networks)))
        else:
            emulator.neural_networks = list(pool.map(f, range(n_networks)))

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
                    group.attrs[key + '_{}'.format(i)] = getattr(network, key)
                except (TypeError, ValueError):
                    pass

            for k in range(network.n_layers_ - 1):
                group.create_dataset('coefs_{}_{}'.format(k, i),
                                     data=network.coefs_[k])
                group.create_dataset('intercepts_{}_{}'.format(k, i),
                                     data=network.intercepts_[k])

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

            network = MLPRegressor()

            for key in group.attrs:
                if key.rsplit('_', 1)[1] == '{}'.format(i):
                    setattr(network, key.rsplit('_', 1)[0], group.attrs[key])

            network.coefs_ = [
                np.array(group['coefs_{}_{}'.format(k, i)]) for k in
                range(network.n_layers_ - 1)]
            network.intercepts_ = [
                np.array(group['intercepts_{}_{}'.format(k, i)]) for k in
                range(network.n_layers_ - 1)]

            emulator.neural_networks.append(network)

        return emulator

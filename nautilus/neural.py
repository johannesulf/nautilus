"""Module implementing neural network emulators."""

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPRegressor


class NeuralNetworkEmulator():
    """Likelihood neural network emulator.

    Attributes
    ----------
    network : sklearn.neural_network.MLPRegressor
        Artifical neural network used for emulation.
    neural_network_thread_limit : int
        Maximum number of threads used by `sklearn`. If None, no limits
        are applied.

    """

    @classmethod
    def train(cls, x, y, neural_network_kwargs={},
              neural_network_thread_limit=1):
        """Initialize and train the likelihood neural network emulator.

        Parameters
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.
        y : numpy.ndarray
            Normalized likelihood value of the training points.
        neural_network_kwargs : dict, optional
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`. By default, no keyword
            arguments are passed to the constructor.
        neural_network_thread_limit : int or None, optional
            Maximum number of threads used by `sklearn`. If None, no limits
            are applied. Default is 1.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()

        emulator.neural_network = MLPRegressor(**neural_network_kwargs)
        emulator.neural_network_thread_limit = neural_network_thread_limit
        with threadpool_limits(limits=emulator.neural_network_thread_limit):
            emulator.neural_network.fit(x, y)

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
        with threadpool_limits(limits=self.neural_network_thread_limit):
            return self.neural_network.predict(x)

    def write(self, group):
        """Write the emulator to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['neural_network_thread_limit'] =\
            self.neural_network_thread_limit

        for key in self.neural_network.__dict__:

            if key in ['coefs_', 'intercepts_']:
                continue

            try:
                group.attrs[key] = getattr(self.neural_network, key)
            except TypeError:
                pass

        for i in range(self.neural_network.n_layers_ - 1):
            group.create_dataset('coefs_{}'.format(i),
                                 data=self.neural_network.coefs_[i])
            group.create_dataset('intercepts_{}'.format(i),
                                 data=self.neural_network.intercepts_[i])

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

        emulator.neural_network_thread_limit =\
            group.attrs['neural_network_thread_limit'].item()

        emulator.neural_network = MLPRegressor()

        for key in group.attrs:
            if key != 'neural_network_thread_limit':
                setattr(emulator.neural_network, key, group.attrs[key])

        emulator.neural_network.coefs_ = [
            np.array(group['coefs_{}'.format(i)]) for i in
            range(emulator.neural_network.n_layers_ - 1)]
        emulator.neural_network.intercepts_ = [
            np.array(group['intercepts_{}'.format(i)]) for i in
            range(emulator.neural_network.n_layers_ - 1)]

        return emulator

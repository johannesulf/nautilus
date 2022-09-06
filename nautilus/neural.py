"""Module implementing neural network emulators."""

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

    def __init__(self, x, y, neural_network_kwargs={},
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

        """
        self.network = MLPRegressor(**neural_network_kwargs)
        self.neural_network_thread_limit = neural_network_thread_limit
        with threadpool_limits(limits=self.neural_network_thread_limit):
            self.network.fit(x, y)

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
            return self.network.predict(x)

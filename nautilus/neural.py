"""Module implementing neural network emulators."""

from sklearn.neural_network import MLPRegressor


class NeuralNetworkEmulator():
    """Likelihood neural network emulator.

    Attributes
    ----------
    network : object
        Artifical neural network used for emulation.

    """

    def __init__(self, x, y, max_epochs=10000, min_delta_mse=1e-4,
                 patience=20):
        """Initialize and train the likelihood neural network emulator.

        Parameters
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.
        y : numpy.ndarray
            Normalized likelihood value of the training points.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 10000.
        min_delta_mse : float, optional
            Training stops if the mean squared error of the training set does
            not improve by at least `min_delta_mse` in `patience` epochs and
            the number of training epochs is at least `min_epochs`. Default is
            1e-4.
        patience : int, optional
            Training stops if the mean squared error of the training set does
            not improve by at least `min_delta_mse` in `patience` epochs and
            the number of training epochs is at least `min_epochs`. Default is
            20.

        """
        kwargs = {'hidden_layer_sizes': (128, 128, 128), 'alpha': 0,
                  'warm_start': True, 'random_state': 0,
                  'learning_rate_init': 1e-2, 'max_iter': max_epochs,
                  'tol': min_delta_mse, 'n_iter_no_change': patience}

        self.network = MLPRegressor(batch_size=min(128, len(x)), **kwargs)
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
        return self.network.predict(x)

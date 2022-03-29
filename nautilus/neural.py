import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor

try:
    from tensorflow.random import set_seed
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from keras.callbacks import Callback, EarlyStopping
    TENSORFLOW_INSTALLED = True
except ImportError:
    TENSORFLOW_INSTALLED = False

if TENSORFLOW_INSTALLED:
    class EarlyStoppingByLossValue(Callback):
        """Callback for Keras to stop training at a specified loss value.

        Attributes
        ----------
        value : float
            Value at which to stop training.
        """

        def __init__(self, value):
            """Initialize a callback for Keras to stop training at a specified
            loss value.

            Attributes
            ----------
            value : float
                Value at which to stop training.
            """

            self.value = value

        def on_batch_end(self, batch, logs={}):
            """Called at the end of each training batch.

            Attributes
            ----------
            batch : int
                Index of the training batch within the current training epoch.
            logs: dict
                Information about training progress.
            """

            if logs.get('loss') <= self.value:
                self.model.stop_training = True


class NeuralNetworkEmulator():
    """Likelihood neural network emulator.

    Attributes
    ----------
    backend : string
        Neural network Python package used for the neural network.
    network : object
        Artifical neural network used for emulation.
    """

    def __init__(self, x, y, backend='tensorflow', min_epochs=100,
                 max_epochs=10000, mse_target=1e-3, min_delta_mse=1e-4,
                 patience=100):
        """Initialize and train the likelihood neural network emulator.

        Attributes
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.
        y : numpy.ndarray
            Normalized likelihood value of the training points.
        backend : string, optional
            Neural network Python package used for the neural network. Options
            are 'tensorflow' and 'scikit-learn'. Default is 'tensorflow'.
        min_epochs : int, optional
            Minimum number of training epochs. Default is 100.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 10000.
        mse_target : float, optional
            Target mean squared error for the normalized likelihood. Training
            stops if this target is achieved for the training set and the
            number of training epochs is at least `min_epochs`. Default is
            1e-3.
        min_delta_mse : float, optional
            Training stops if the mean squared error of the training set does
            not improve by at least `min_delta_mse` in `patience` epochs and
            the number of training epochs is at least `min_epochs`. Default is
            1e-4.
        patience : int, optional
            Training stops if the mean squared error of the training set does
            not improve by at least `min_delta_mse` in `patience` epochs and
            the number of training epochs is at least `min_epochs`. Default is
            100.
        """

        if backend == 'tensorflow' and not TENSORFLOW_INSTALLED:
            warnings.warn("The tensorflow backend was requested but " +
                          "tensorflow is not installed. Falling back to the " +
                          "scikit-learn backend.")
            backend = 'scikit-learn'

        self.backend = backend

        if backend == 'tensorflow':
            set_seed(0)
            self.network = Sequential()
            self.network.add(Dense(units=128, activation='relu',
                                   input_dim=x.shape[1]))
            self.network.add(Dense(units=128, activation='relu'))
            self.network.add(Dense(units=128, activation='relu'))
            self.network.add(Dense(units=1))
            self.network.compile(loss='mean_squared_error', optimizer='adam')
            self.network.fit(x, y, epochs=min(50, min_epochs), batch_size=64,
                             verbose=0)
            self.network.fit(x, y, epochs=max(0, min_epochs - 50),
                             batch_size=len(x), verbose=0)
            callback_1 = EarlyStopping(
                monitor='loss', min_delta=min_delta_mse, patience=patience)
            callback_2 = EarlyStoppingByLossValue(mse_target)
            self.network.fit(
                x, y, epochs=max_epochs-min_epochs, batch_size=len(x),
                callbacks=[callback_1, callback_2], verbose=0)
        elif backend == 'scikit-learn':
            self.network = MLPRegressor(
                hidden_layer_sizes=(128, 128, 128), alpha=0,
                batch_size=len(x), warm_start=True, random_state=0)
            self.network.partial_fit(x, y)
            while self.network.n_iter_ < max_epochs:
                self.network.partial_fit(x, y)
                mse = self.network.loss_curve_
                if self.network.n_iter_ >= min_epochs and mse[-1] < mse_target:
                    break
                if (self.network.n_iter_ >= min_epochs and len(mse) > patience
                    and np.amin(mse[:-patience]) - np.amin(mse[-patience:]) <
                        min_delta_mse):
                    break

    def predict(self, x):
        """Calculate the emulator likelihood prediction for a group of points.

        Attributes
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.

        Returns
        -------
        numpy.ndarray
            Emulated normalized likelihood value of the training points.
        """

        if self.backend == 'tensorflow':
            return self.network(x).numpy().flatten()
        elif self.backend == 'scikit-learn':
            return self.network.predict(x)

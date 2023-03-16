"""Module implementing multi-dimensional neural network-based bounds."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import percentileofscore

from .basic import UnitCube, Ellipsoid, UnitCubeEllipsoidMixture
from .union import Union
from ..neural import NeuralNetworkEmulator


class NeuralBound():
    """Neural network-based bound.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    random_state : numpy.random or numpy.random.RandomState instance
        Determines random number generation.
    outer_bound : UnitCubeEllipsoidMixture
        Outer bound around the points above the likelihood threshold.
    emulator : object
        Emulator based on `sklearn.neural_network.MLPRegressor` used to fit and
        predict likelihood scores.
    score_predict_min : float
        Minimum score predicted by the emulator to be considered part of the
        bound.
    """

    @classmethod
    def compute(cls, points, log_l, log_l_min, enlarge_per_dim,
                neural_network_kwargs={}, neural_network_thread_limit=1,
                random_state=None):
        """Compute a neural network-based bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        log_l : numpy.ndarray of length m
            Likelihood of each point.
        log_l_min : float
            Target likelihood threshold of the bound.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid of the outer bound is enlarged
            by this factor. Default is 1.1.
        neural_network_kwargs : dict, optional
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`. By default, no keyword
            arguments are passed to the constructor.
        neural_network_thread_limit : int or None, optional
            Maximum number of threads used by `sklearn`. If None, no limits
            are applied. Default is 1.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NeuralBound
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        if random_state is None:
            bound.random_state = np.random
        else:
            bound.random_state = random_state

        # Determine the outer bound.
        bound.outer_bound = UnitCubeEllipsoidMixture.compute(
            points[log_l > log_l_min], enlarge_per_dim=enlarge_per_dim,
            random_state=bound.random_state)

        # Train the network.
        select = bound.outer_bound.contains(points)
        points = points[select]
        log_l = log_l[select]

        points_t = bound.outer_bound.transform(points)
        perc = np.argsort(np.argsort(log_l)) / float(len(log_l))
        perc_min = percentileofscore(log_l, log_l_min) / 100
        score = np.zeros(len(points))
        select = perc < perc_min
        if np.any(select):
            score[select] = 0.5 * (perc[select] / perc_min)
        score[~select] = 1 - 0.5 * (1 - perc[~select]) / (1 - perc_min)
        bound.emulator = NeuralNetworkEmulator.train(
            points_t, score, neural_network_kwargs=neural_network_kwargs,
            neural_network_thread_limit=neural_network_thread_limit)

        bound.score_predict_min = np.polyval(np.polyfit(
            score, bound.emulator.predict(points_t), 3), 0.5)

        return bound

    def contains(self, points):
        """Check whether points are contained in the bound.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        in_bound : bool or numpy.ndarray
            Bool or array of bools describing for each point if it is contained
            in the bound.

        """
        points = np.atleast_2d(points)

        in_bound = self.outer_bound.contains(points)
        if np.any(in_bound):
            points_t = self.outer_bound.transform(points)
            in_bound[in_bound] = (self.emulator.predict(points_t[in_bound]) >
                                  self.score_predict_min)

        return np.squeeze(in_bound)

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_dim'] = self.n_dim
        group.attrs['score_predict_min'] = self.score_predict_min
        self.outer_bound.write(group.create_group('outer_bound'))
        self.emulator.write(group.create_group('emulator'))

    @classmethod
    def read(cls, group, random_state=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NeuralBound
            The bound.

        """
        bound = cls()

        if random_state is None:
            bound.random_state = np.random
        else:
            bound.random_state = random_state

        bound.n_dim = group.attrs['n_dim']
        bound.score_predict_min = group.attrs['score_predict_min']
        bound.outer_bound = UnitCubeEllipsoidMixture.read(group['outer_bound'])
        bound.emulator = NeuralNetworkEmulator.read(group['emulator'])

        return bound


class NautilusBound():
    """Union of multiple non-overlapping neural network-based bounds.

    The bound is the overlap of the union of multiple neural network-based
    bounds and the unit hypercube.

    Attributes
    ----------
    log_v : list
        Natural log of the volume of the sampling bound.
    neural_bounds : list
        List of the individual neural network-based bounds.
    outer_bound : Union
        Outer bound used for sampling.
    random_state : None or numpy.random.RandomState instance
        Determines random number generation.
    points_sample : numpy.ndarray
        Points that a call to `sample` will return next.
    n_sample : int
        Number of points sampled from all ellipsoids.
    n_reject : int
        Number of points rejected due to overlap.
    """

    @classmethod
    def compute(cls, points, log_l, log_l_min, log_v_target,
                enlarge_per_dim=1.1, n_points_min=None, split_threshold=100,
                use_neural_networks=True, neural_network_kwargs={},
                neural_network_thread_limit=1, random_state=None):
        """Compute a union of multiple neural network-based bounds.

        Parameters
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        log_l : numpy.ndarray of length m
            Likelihood of each point.
        log_l_min : float
            Target likelihood threshold of the bound.
        log_v_target : float
            Expected volume of the bound. Used for multi-ellipsoidal
            decomposition.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid of the outer bound is enlarged
            by this factor. Default is 1.1.
        n_points_min : int or None, optional
            The minimum number of points each ellipsoid should have.
            Effectively, ellipsoids with less than twice that number will not
            be split further. If None, uses `n_points_min = n_dim + 1`. Default
            is None.
        split_threshold: float, optional
            Threshold used for splitting the multi-ellipsoidal bound used for
            sampling. If the volume of the bound is larger than
            `split_threshold` times the target volume, the multi-ellipsiodal
            bound is split further, if possible. Default is 100.
        use_neural_networks : bool, optional
            Whether to use neural network emulators in the construction of the
            bound. Default is True.
        neural_network_kwargs : dict, optional
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`. By default, no keyword
            arguments are passed to the constructor.
        neural_network_thread_limit : int or None, optional
            Maximum number of threads used by `sklearn`. If None, no limits
            are applied. Default is 1.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NautilusBound
            The bound.

        """
        bound = cls()

        mell = Union.compute(
            points[log_l >
                   log_l_min], enlarge_per_dim=enlarge**(1 / points.shape[1]),
            n_points_min=n_points_min, random_state=random_state)
        cube = UnitCube.compute(points.shape[-1], random_state=random_state)

        while mell.split_bound(allow_overlap=False):
            pass

        bound.n_networks = len(mell.bounds)
        bound.nbounds = []

        if use_neural_networks:
            for ell in mell.bounds:
                select = ell.contains(points)
                bound.nbounds.append(NeuralBound.compute(
                    points[select], log_l[select], log_l_min, ellipsoid=ell,
                    enlarge=enlarge,
                    neural_network_kwargs=neural_network_kwargs,
                    neural_network_thread_limit=neural_network_thread_limit,
                    random_state=random_state))

        while min(mell.volume(), 0.0) - log_v_target > np.log(
                split_threshold * enlarge):
            if mell.volume() >= 0:
                points_test = cube.sample(10000)
                log_v = cube.volume() + np.log(
                    np.mean(mell.contains(points_test)))
            else:
                points_test = mell.sample(10000)
                log_v = mell.volume() + np.log(
                    np.mean(cube.contains(points_test)))
            if log_v - log_v_target > np.log(split_threshold * enlarge):
                if not mell.split_bound():
                    break
            else:
                break

        if mell.volume() >= 0:
            bound.sample_bounds = (cube, mell)
        else:
            bound.sample_bounds = (mell, cube)

        bound.log_v = bound.sample_bounds[0].volume()

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        bound.points_sample = np.zeros((0, points.shape[1]))
        bound.n_sample = 0
        bound.n_reject = 0

        return bound

    def contains(self, points):
        """Check whether points are contained in the ellipsoid.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        in_bound : bool or numpy.ndarray
            Bool or array of bools describing for each point it is contained in
            the bound.

        """
        in_bound = (self.sample_bounds[0].contains(points) &
                    self.sample_bounds[1].contains(points))
        if len(self.nbounds) > 0:
            in_bound = in_bound & np.any(np.vstack(
                [bound.contains(points) for bound in self.nbounds]), axis=0)
        return in_bound

    def sample(self, n_points=1):
        """Sample points from the the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw.

        Returns
        -------
        points : numpy.ndarray
            If `n_points` is larger than 1, two-dimensional array of shape
            (n_points, n_dim). Otherwise, a one-dimensional array of shape
            (n_dim).

        """
        while len(self.points_sample) < n_points:
            n_sample = 10000
            points = self.sample_bounds[0].sample(n_sample)
            points = points[self.sample_bounds[1].contains(points)]
            if len(self.nbounds) > 0:
                in_bound = np.any(np.vstack(
                    [bound.contains(points) for bound in self.nbounds]),
                    axis=0)
                points = points[in_bound]
            self.points_sample = np.vstack([self.points_sample, points])
            self.n_sample += n_sample
            self.n_reject += n_sample - len(points)

        # Update volumes since they might have gotten more accurate.
        self.log_v = self.sample_bounds[0].volume()

        points = self.points_sample[:n_points]
        self.points_sample = self.points_sample[n_points:]
        return np.squeeze(points)

    def volume(self):
        """Return the natural log of the volume.

        Returns
        -------
        log_v : float
            An estimate of the natural log of the volume. Will become more
            accurate as more points are sampled.

        """
        if self.n_sample == 0:
            self.sample()

        return logsumexp(self.log_v) + np.log(
            1.0 - self.n_reject / self.n_sample)

    def number_of_networks_and_ellipsoids(self):
        """Return the number of neural networks and sample ellipsoids.

        Returns
        -------
        n_neural : int
            The number of neural networks.
        n_sample : int
            The number of sample ellipsoids.
        """
        n_neural = len(self.nbounds)

        if isinstance(self.sample_bounds[0], UnitCube):
            i = 1
        else:
            i = 0
        n_sample = len(self.sample_bounds[i].ells)

        return n_neural, n_sample

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'NautilusBound'

        for key in ['n_networks', 'log_v', 'n_sample', 'n_reject']:
            group.attrs[key] = getattr(self, key)

        for i, nbound in enumerate(self.nbounds):
            nbound.write(group.create_group('nbound_{}'.format(i)))

        self.sample_bounds[0].write(group.create_group('sample_bound_0'))
        self.sample_bounds[1].write(group.create_group('sample_bound_1'))

        group.create_dataset('points_sample', data=self.points_sample)

    @classmethod
    def read(cls, group, random_state=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NautilusBound
            The bound.

        """
        bound = cls()

        if random_state is None:
            bound.random_state = np.random
        else:
            bound.random_state = random_state

        for key in ['n_networks', 'log_v', 'n_sample', 'n_reject']:
            setattr(bound, key, group.attrs[key])

        bound.nbounds = [NeuralBound.read(
            group['nbound_{}'.format(i)], random_state=bound.random_state) for
            i in range(bound.n_networks)]
        if group['sample_bound_0'].attrs['type'] == 'UnitCube':
            classes = [UnitCube, Union]
        else:
            classes = [Union, UnitCube]
        bound.sample_bounds = tuple(
            classes[i].read(group['sample_bound_{}'.format(i)],
                            random_state=bound.random_state) for i in range(2))
        bound.points_sample = np.array(group['points_sample'])

        return bound

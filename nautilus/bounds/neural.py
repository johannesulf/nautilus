"""Module implementing multi-dimensional neural network-based bounds."""

import numpy as np
from functools import partial
from scipy.stats import percentileofscore
from threadpoolctl import threadpool_limits

from .basic import Ellipsoid, UnitCubeEllipsoidMixture
from .union import Union
from ..neural import NeuralNetworkEmulator


class NeuralBound():
    """Neural network-based bound.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
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
    def compute(cls, points, log_l, log_l_min, enlarge_per_dim=1.1,
                n_networks=4, neural_network_kwargs={}, pool=None,
                rng=None):
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
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor.
        pool : multiprocessing.Pool, optional
            Pool used for parallel processing.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NeuralBound
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        if rng is None:
            rng = np.random.default_rng()

        # Determine the outer bound.
        bound.outer_bound = Ellipsoid.compute(
            points[log_l > log_l_min], enlarge_per_dim=enlarge_per_dim,
            rng=rng)

        if n_networks == 0:
            bound.emulator = None
            bound.score_predict_min = 0
            return bound

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
            points_t, score, n_networks=n_networks,
            neural_network_kwargs=neural_network_kwargs, pool=pool)

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
        if np.any(in_bound) and self.emulator is not None:
            points_t = self.outer_bound.transform(points)
            in_bound[in_bound] = (self.emulator.predict(points_t[in_bound]) >
                                  self.score_predict_min)

        return in_bound

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
        if self.emulator is not None:
            self.emulator.write(group.create_group('emulator'))

    @classmethod
    def read(cls, group, rng=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NeuralBound
            The bound.

        """
        bound = cls()

        if rng is None:
            rng = np.random.default_rng()

        bound.n_dim = group.attrs['n_dim']
        bound.score_predict_min = group.attrs['score_predict_min']
        bound.outer_bound = Ellipsoid.read(group['outer_bound'], rng=rng)
        if 'emulator' in group:
            bound.emulator = NeuralNetworkEmulator.read(group['emulator'])
        else:
            bound.emulator = None

        return bound


class NautilusBound():
    """Union of multiple non-overlapping neural network-based bounds.

    The bound is the overlap of the union of multiple neural network-based
    bounds and the unit hypercube.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    neural_bounds : list
        List of the individual neural network-based bounds.
    outer_bound : Union
        Outer bound used for sampling.
    rng : None or numpy.random.Generator
        Determines random number generation.
    points : numpy.ndarray
        Points that a call to `sample` will return next.
    n_sample : int
        Number of points sampled from the outer bound.
    n_reject : int
        Number of points rejected due to not falling into the neural
        network-based bounds.
    """

    @classmethod
    def compute(cls, points, log_l, log_l_min, log_v_target,
                enlarge_per_dim=1.1, n_points_min=None, split_threshold=100,
                n_networks=4, neural_network_kwargs={}, pool=None,
                rng=None):
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
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor.
        pool : multiprocessing.Pool, optional
            Pool used for parallel processing.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NautilusBound
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        bound.neural_bounds = []

        multi_ellipsoid = Union.compute(
            points[log_l >= log_l_min], enlarge_per_dim=enlarge_per_dim,
            n_points_min=n_points_min, bound_class=Ellipsoid,
            rng=rng)

        while multi_ellipsoid.split_bound(allow_overlap=False):
            pass

        for ellipsoid in multi_ellipsoid.bounds:
            select = ellipsoid.contains(points)
            bound.neural_bounds.append(NeuralBound.compute(
                points[select], log_l[select], log_l_min,
                enlarge_per_dim=enlarge_per_dim, n_networks=n_networks,
                neural_network_kwargs=neural_network_kwargs, pool=pool,
                rng=rng))

        bound.outer_bound = Union.compute(
            points[log_l > log_l_min], enlarge_per_dim=enlarge_per_dim,
            n_points_min=n_points_min, bound_class=UnitCubeEllipsoidMixture,
            rng=rng)

        while bound.outer_bound.volume() - log_v_target > np.log(
                split_threshold * enlarge_per_dim**points.shape[1]):
            if not bound.outer_bound.split_bound():
                break

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

        bound.points = np.zeros((0, points.shape[1]))
        bound.n_sample = 0
        bound.n_reject = 0

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
            Bool or array of bools describing for each point it is contained in
            the bound.

        """
        in_bound = self.outer_bound.contains(points)
        if len(self.neural_bounds) > 0:
            in_bound = in_bound & np.any(
                [bound.contains(points) for bound in self.neural_bounds],
                axis=0)
        return in_bound

    @threadpool_limits.wrap(limits=1)
    def _reset_and_sample(self, n_points=100, rng=None):
        """Reset the bound, sample points internally and return the result.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw. Default is 100.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. If None, random number
            generation is not reset. Default is None.

        Returns
        -------
        bound : NautilusBound
            The bound.

        """
        self.reset(rng=rng)
        self.sample(n_points=n_points, return_points=False)
        return self

    def sample(self, n_points=100, return_points=True, pool=None):
        """Sample points from the the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw. Default is 100.
        return_points : bool, optional
            If True, return sampled points. Otherwise, sample internally until
            at least `n_points` are saved.
        pool : multiprocessing.Pool, optional
            Pool used for parallel processing.

        Returns
        -------
        points : numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        if len(self.points) < n_points:
            if pool is None:
                while len(self.points) < n_points:
                    n_sample = 1000
                    points = self.outer_bound.sample(n_sample)
                    in_bound = np.any([bound.contains(points) for bound in
                                       self.neural_bounds], axis=0)
                    points = points[in_bound]
                    self.points = np.vstack([self.points, points])
                    self.n_sample += n_sample
                    self.n_reject += n_sample - len(points)
            else:
                n_jobs = pool._processes
                n_points_per_job = (
                    (max(n_points - len(self.points), 10000)) // n_jobs) + 1
                func = partial(self._reset_and_sample, n_points_per_job)
                rngs = [np.random.default_rng(seed) for seed in
                        np.random.SeedSequence(self.rng.integers(
                            2**32 - 1)).spawn(n_jobs)]
                bounds = pool.map(func, rngs)
                for bound in bounds:
                    self.points = np.vstack([self.points, bound.points])
                    self.n_sample += bound.n_sample
                    self.n_reject += bound.n_reject
                    self.outer_bound.n_sample += bound.outer_bound.n_sample
                    self.outer_bound.n_reject += bound.outer_bound.n_reject

        if return_points:
            points = self.points[:n_points]
            self.points = self.points[n_points:]
            return points

    def volume(self):
        """Return the natural log of the volume.

        Returns
        -------
        log_v : float
            An estimate of the natural log of the volume. Will become more
            accurate as more points are sampled.

        """
        if self.n_sample == 0:
            self.sample(return_points=False)

        return self.outer_bound.volume() + np.log(
            1.0 - self.n_reject / self.n_sample)

    def number_of_networks_and_ellipsoids(self):
        """Return the number of neural networks and sample ellipsoids.

        Returns
        -------
        n_networks : int
            The number of neural networks.
        n_ellipsoids : int
            The number of sample ellipsoids.
        """
        if self.neural_bounds[0].emulator is not None:
            n_networks = len(self.neural_bounds) * len(
                self.neural_bounds[0].emulator.neural_networks)
        else:
            n_networks = 0

        n_ellipsoids = 0
        for bound in self.outer_bound.bounds:
            n_ellipsoids += np.any(~bound.dim_cube)

        return n_networks, n_ellipsoids

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'NautilusBound'
        group.attrs['n_dim'] = self.n_dim
        group.attrs['n_neural_bounds'] = len(self.neural_bounds)

        for i, neural_bound in enumerate(self.neural_bounds):
            neural_bound.write(group.create_group('neural_bound_{}'.format(i)))

        self.outer_bound.write(group.create_group('outer_bound'))

        group.create_dataset('points', data=self.points)
        group.attrs['n_sample'] = self.n_sample
        group.attrs['n_reject'] = self.n_reject

    @classmethod
    def read(cls, group, rng=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NautilusBound
            The bound.

        """
        bound = cls()

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

        bound.n_dim = group.attrs['n_dim']

        bound.neural_bounds = []
        i = 0
        while 'neural_bound_{}'.format(i) in group:
            bound.neural_bounds.append(NeuralBound.read(
                group['neural_bound_{}'.format(i)],
                rng=bound.rng))
            i += 1

        bound.outer_bound = Union.read(
            group['outer_bound'], rng=rng)

        bound.points = np.array(group['points'])
        bound.n_sample = group.attrs['n_sample']
        bound.n_reject = group.attrs['n_reject']

        return bound

    def reset(self, rng=None):
        """Reset random number generation and any progress, if applicable.

        Parameters
        ----------
        rng : None or numpy.random.Generator, optional
            Determines random number generation. If None, random number
            generation is not reset. Default is None.

        """
        self.points = np.zeros((0, self.n_dim))
        self.n_sample = 0
        self.n_reject = 0
        self.outer_bound.reset(rng)
        if rng is not None:
            self.rng = rng

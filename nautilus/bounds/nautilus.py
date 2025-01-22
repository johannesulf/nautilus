"""Module implementing the nautilus bound."""

import numpy as np
from functools import partial
from threadpoolctl import threadpool_limits

from .basic import Ellipsoid, UnitCubeEllipsoidMixture
from .neural import NeuralBound
from .periodic import PhaseShift
from .union import Union


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
                periodic=None, n_networks=4, neural_network_kwargs={},
                pool=None, rng=None):
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
        periodic : numpy.ndarray or None
            Indices of the parameters that are periodic.
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor.
        pool : nautilus.pool.NautilusPool or None, optional
            Pool used for parallel processing. Default is None.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NautilusBound
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        if periodic is not None:
            bound.shift = PhaseShift.compute(points[log_l >= log_l_min],
                                             periodic)
            points = bound.shift.transform(points)
        else:
            bound.shift = None

        bound.neural_bounds = []

        multi_ellipsoid = Union.compute(
            points[log_l >= log_l_min], enlarge_per_dim=enlarge_per_dim,
            n_points_min=n_points_min, bound_class=Ellipsoid,
            rng=rng)

        while multi_ellipsoid.split(allow_overlap=False):
            pass

        for ellipsoid in multi_ellipsoid.bounds:
            select = ellipsoid.contains(points)
            bound.neural_bounds.append(NeuralBound.compute(
                points[select], log_l[select], log_l_min,
                enlarge_per_dim=enlarge_per_dim, n_networks=n_networks,
                neural_network_kwargs=neural_network_kwargs, pool=pool,
                rng=rng))

        bound.outer_bound = Union.compute(
            points[log_l >= log_l_min], enlarge_per_dim=enlarge_per_dim,
            n_points_min=n_points_min, bound_class=UnitCubeEllipsoidMixture,
            rng=rng)

        # If the single bounding ellipsoid is too large, split ellipsoids
        # further.
        while bound.outer_bound.log_v - log_v_target > np.log(
                split_threshold * enlarge_per_dim**points.shape[1]):
            if not bound.outer_bound.split():
                break

        # If the ellipsoid union is still too large, check whether some
        # ellipsoids have too low densities and should be dropped.
        while bound.outer_bound.log_v - log_v_target > np.log(
                split_threshold * enlarge_per_dim**points.shape[1]):
            if not bound.outer_bound.trim():
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
        if self.shift is not None:
            points = self.shift.transform(points)
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
        pool : nautilus.pool.NautilusPool or None, optional
            Pool used for parallel processing. Default is None.

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
                n_jobs = pool.size
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
            if self.shift is not None:
                points = self.shift.transform(points, inverse=True)
            return points

    @property
    def log_v(self):
        """Return the natural log of the volume.

        Returns
        -------
        log_v : float
            An estimate of the natural log of the volume. Will become more
            accurate as more points are sampled.

        """
        if self.n_sample == 0:
            self.sample(return_points=False)

        return self.outer_bound.log_v + np.log(
            1.0 - self.n_reject / self.n_sample)

    @property
    def n_ell(self):
        """Return the number of ellipsoids in the bound.

        Returns
        -------
        n_ell : int
            The number of ellipsoids.

        """
        return np.sum([np.any(~bound.dim_cube) for bound in
                       self.outer_bound.bounds])

    @property
    def n_net(self):
        """Return the number of neural networks in the bound.

        Returns
        -------
        n_net : int
            The number of neural networks.

        """
        if self.neural_bounds[0].emulator is not None:
            return len(self.neural_bounds) * len(
                self.neural_bounds[0].emulator.neural_networks)
        else:
            return 0

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'NautilusBound'
        group.attrs['n_dim'] = self.n_dim

        if self.shift is not None:
            self.shift.write(group.create_group('shift'))

        group.attrs['n_neural_bounds'] = len(self.neural_bounds)

        for i, neural_bound in enumerate(self.neural_bounds):
            neural_bound.write(group.create_group('neural_bound_{}'.format(i)))

        self.outer_bound.write(group.create_group('outer_bound'))

        group.create_dataset('points', data=self.points,
                             maxshape=(None, self.n_dim))
        group.attrs['n_sample'] = self.n_sample
        group.attrs['n_reject'] = self.n_reject

    def update(self, group):
        """Update bound information previously written to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_sample'] = self.n_sample
        group.attrs['n_reject'] = self.n_reject
        self.outer_bound.update(group['outer_bound'])
        group['points'].resize(self.points.shape)
        group['points'][...] = self.points

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

        if 'shift' in group:
            bound.shift = PhaseShift.read(group['shift'])
        else:
            bound.shift = None

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

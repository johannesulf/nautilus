"""Module implementing unions of basic bounds."""

import itertools
import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .basic import UnitCube, Ellipsoid, UnitCubeEllipsoidMixture


def ellipsoids_overlap(ellipsoids):
    """Determine if ellipsoids overlap.

    This functions is based on ieeexplore.ieee.org/document/6289830.

    Parameters
    ----------
    ellipsoids : list
        List of ellipsoids.

    Returns
    -------
    overlapping : bool
        True, if there is overlap between ellipoids and False otherwise.

    """
    c = [ellipsoid.c for ellipsoid in ellipsoids]
    A_inv = [np.linalg.inv(ellipsoid.A) for ellipsoid in ellipsoids]

    for i_1, i_2 in itertools.combinations(range(len(c)), 2):
        d = c[i_1] - c[i_2]
        def k(s): return (1 - np.dot(np.dot(
            d, np.linalg.inv(A_inv[i_1] / (1 - s) + A_inv[i_2] / s)), d))
        if minimize(k, 0.5, bounds=[(1e-9, 1-1e-9)]).fun > 0:
            return True

    return False


class Union():
    r"""Union of multiple ellipsoids or unit cube-ellipsoid mixtures.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    enlarge_per_dim : float
        Along each dimension, ellipsoids enlarged by this factor.
    n_points_min : int or None
        The minimum number of points each ellipsoid should have. Effectively,
        ellipsoids with less than twice that number will not be split further.
    cube : UnitCube or None
        If not None, the bound is defined as the overlap with the unit cube.
    points_bounds : list
        The points used to create the individual bounds. Used to split bounds
        further.
    bounds : list
        List of individual bounds.
    log_v_all : numpy.ndarray
        Natural log of the volume of each individual bound.
    block : numpy.ndarray
        List indicating whether a bound should not be split further because
        the volume would not decrease or the number of points is too low.
    points : numpy.ndarray
        Points that a call to `sample` will return next.
    n_sample : int
        Number of points sampled from all bounds.
    n_reject : int
        Number of points rejected due to overlap.
    rng : numpy.random.Generator
        Determines random number generation.

    """

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, n_points_min=None,
                unit=True, bound_class=Ellipsoid, rng=None):
        """Compute the bound.

        Upon creation, the bound consists of a single individual bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid is enlarged by this factor.
            Default is 1.1.
        n_points_min : int or None, optional
            The minimum number of points each ellipsoid should have.
            Effectively, ellipsoids with less than twice that number will not
            be split further. If None, uses `n_points_min = n_dim + 1`. Default
            is None.
        unit : bool, optional
            If the bound is restricted to the overlap with the unit cube.
        bound_class : class, optional
            Type of the individual bounds, i.e. ellipsoids or unit
            cube-ellipsoid mixtures.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Raises
        ------
        ValueError
            If `n_points_min` is smaller than the number of dimensions plus
            one.

        Returns
        -------
        bound : Union
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]
        bound.enlarge_per_dim = enlarge_per_dim

        if n_points_min is None:
            bound.n_points_min = bound.n_dim + 1
        else:
            if n_points_min < bound.n_dim + 1:
                raise ValueError('The number of points per bound must be ' +
                                 'larger than the number of dimensions.')
            bound.n_points_min = n_points_min

        if not unit:
            bound.cube = None
        else:
            bound.cube = UnitCube.compute(
                bound.n_dim, rng=rng)

        bound.points_bounds = [points]
        bound.bounds = [bound_class.compute(
            points, enlarge_per_dim=enlarge_per_dim,
            rng=rng)]
        bound.log_v_all = np.array([bound.bounds[0].log_v])
        bound.block = np.atleast_1d(len(points) < 2 * bound.n_points_min)

        bound.points = np.zeros((0, points.shape[1]))
        bound.n_sample = 0
        bound.n_reject = 0

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

        return bound

    def split(self, allow_overlap=True):
        """Split the largest bound in the union.

        Parameters
        ----------
        allow_overlap : bool, optional
            Whether to allow splitting the largest bound if doing so creates
            overlaps between bounds. Cannot be False if the individual bounds
            are cube-ellipsoid mixtures. Default is True.

        Raises
        ------
        ValueError
            If `allow_overlap` is False and the individual bounds are
            cube-ellipsoid mixtures.

        Returns
        -------
        success : bool
            Whether it was possible to split any bound.

        """
        if not allow_overlap and not isinstance(self.bounds[0], Ellipsoid):
            raise ValueError("'allow_overlap' can only be False if " +
                             "bounds are ellipsoids.")

        if not np.any(~self.block):
            return False

        index = np.argmax(np.where(~self.block, self.log_v_all, -np.inf))
        points = self.bounds[index].transform(self.points_bounds[index])

        gmm = GaussianMixture(
            n_components=2, n_init=10,
            random_state=self.rng.integers(2**32 - 1)).fit(points)
        p = np.vstack([multivariate_normal.logpdf(
            points, mean=gmm.means_[i], cov=gmm.covariances_[i]) +
            np.log(gmm.weights_[i]) for i in range(2)]).T

        labels = np.argmax(p, axis=1)
        # If one of the clusters has less than n_points_min members, re-assign
        # the most likely members from the larger cluster to the smaller one.
        if not np.all(np.bincount(labels) >= self.n_points_min):
            label = np.argmin(np.bincount(labels))
            labels[np.argsort(-p[:, label])[:self.n_points_min]] = label

        new_bounds = []
        points = self.points_bounds[index]
        for label in [0, 1]:
            new_bounds.append(type(self.bounds[0]).compute(
                points[labels == label], enlarge_per_dim=self.enlarge_per_dim,
                rng=self.rng))

        if not allow_overlap and ellipsoids_overlap(
                self.bounds[:index] + self.bounds[index+1:] + new_bounds):
            return False

        if (logsumexp([new_bounds[0].log_v, new_bounds[1].log_v]) >
                self.bounds[index].log_v):
            self.block[index] = True
            return self.split(allow_overlap=allow_overlap)

        self.points_bounds.pop(index)
        self.points_bounds.append(points[labels == 0])
        self.points_bounds.append(points[labels == 1])
        self.bounds.pop(index)
        self.bounds = self.bounds + new_bounds
        self.log_v_all = np.array([bound.log_v for bound in self.bounds])
        self.block = np.concatenate(
            (np.delete(self.block, index),
             [len(self.points_bounds[-2]) < 2 * self.n_points_min,
              len(self.points_bounds[-1]) < 2 * self.n_points_min]))

        # Reset the sampling.
        self.reset()

        return True

    def trim(self, threshold=1e3):
        """Drop the lowest-density bound, if possible.

        Density is defined as the ratio of each bound's number of points to its
        volume.

        Parameters
        ----------
        threshold : float, optional
            Only drop the lowest-density bound if it has a density at least
            `threshold` times lower than the median of all other bounds.

        Returns
        -------
        success : bool
            Whether it was possible to drop a bound. Will always return False
            if there is only one bound in the union.

        """
        if len(self.bounds) == 1:
            return False

        log_n = np.array([np.log(len(points)) for points in
                          self.points_bounds])
        log_v = np.array([bound.log_v for bound in self.bounds])
        log_r = log_n - log_v
        index = np.argmin(log_r)

        if log_r[index] - np.median(np.delete(log_r, index)) < -np.log(
                threshold):
            self.points_bounds.pop(index)
            self.bounds.pop(index)
            self.log_v_all = np.array([bound.log_v for bound in self.bounds])
            self.reset()
            return True
        else:
            return False

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
            Bool or array of bools describing for each point whether it is
            contained in the bound.

        """
        in_bound = np.any([bound.contains(points) for bound in self.bounds],
                          axis=0)
        if self.cube is not None:
            in_bound = in_bound & self.cube.contains(points)
        return in_bound

    def sample(self, n_points=100):
        """Sample points from the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw. Default is 100.

        Returns
        -------
        points : numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        while len(self.points) < n_points:
            n_sample = 1000

            p = np.exp(np.array(self.log_v_all) - logsumexp(self.log_v_all))
            n_per_bound = self.rng.multinomial(n_sample, p)

            points = np.vstack([bound.sample(n) for bound, n in
                                zip(self.bounds, n_per_bound)])
            if self.cube is not None:
                points = points[self.cube.contains(points)]
            self.rng.shuffle(points)
            n_bound = np.sum([bound.contains(points) for bound in self.bounds],
                             axis=0)
            p = 1 - 1.0 / n_bound
            points = points[self.rng.random(size=len(points)) > p]
            self.points = np.vstack([self.points, points])

            self.n_sample += n_sample
            self.n_reject += n_sample - len(points)

        points = self.points[:n_points]
        self.points = self.points[n_points:]
        return points

    @property
    def log_v(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v : float
            The natural log of the volume.

        """
        if self.n_sample == 0:
            self.sample()

        return logsumexp(self.log_v_all) + np.log(
            1.0 - self.n_reject / self.n_sample)

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'MultiEllipsoid'
        for key in ['n_dim', 'log_v_all', 'enlarge_per_dim', 'n_points_min',
                    'n_sample', 'n_reject']:
            group.attrs[key] = getattr(self, key)

        group.attrs['unit'] = self.cube is not None
        if self.cube is not None:
            self.cube.write(group.create_group('cube'))

        group.attrs['bound_class'] = self.bounds[0].__class__.__name__
        for i, bound in enumerate(self.bounds):
            bound.write(group.create_group('bound_{}'.format(i)))

        for i, points in enumerate(self.points_bounds):
            group.create_dataset('points_bound_{}'.format(i), data=points)
        group.create_dataset('points', data=self.points,
                             maxshape=(None, self.n_dim))

    def update(self, group):
        """Update bound information previously written to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_sample'] = self.n_sample
        group.attrs['n_reject'] = self.n_reject
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
        bound : MultiEllipsoid
            The bound.

        """
        bound = cls()

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

        for key in ['n_dim', 'log_v_all', 'enlarge_per_dim', 'n_points_min',
                    'n_sample', 'n_reject']:
            setattr(bound, key, group.attrs[key])

        if group.attrs['unit']:
            bound.cube = UnitCube.read(group['cube'], rng=bound.rng)

        if group.attrs['bound_class'] == 'Ellipsoid':
            bound_class = Ellipsoid
        else:
            bound_class = UnitCubeEllipsoidMixture

        bound.bounds = [bound_class.read(
            group['bound_{}'.format(i)], rng=bound.rng)
            for i in range(len(bound.log_v_all))]
        bound.points_bounds = [np.array(group['points_bound_{}'.format(i)]) for
                               i in range(len(bound.log_v_all))]
        bound.points = np.array(group['points'])

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

        if rng is not None:
            self.rng = rng
            if self.cube is not None:
                self.cube.reset(rng)
            for bound in self.bounds:
                bound.reset(rng)

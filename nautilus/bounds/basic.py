"""Module implementing basic multi-dimensional bounds."""

import itertools
import numpy as np
from scipy.special import gammaln
from scipy.special import logsumexp
from scipy.linalg.lapack import dpotrf, dpotri
from scipy.optimize import minimize
from sklearn.cluster import KMeans


class UnitCube():
    r"""Unit (hyper)cube bound in :math:`n_{\rm dim}` dimensions.

    The :math:`n`-dimensional unit hypercube has :math:`n_{\rm dim}` parameters
    :math:`x_i` with :math:`0 \leq x_i < 1` for all :math:`x_i`.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    random_state : numpy.random.RandomState instance
        Determines random number generation.

    """

    @classmethod
    def compute(cls, n_dim, random_state=None):
        """Compute the bound.

        Parameters
        ----------
        n_dim : int
            Number of dimensions.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : UnitCube
            The bound.

        """
        bound = cls()
        bound.n_dim = n_dim

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

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
            Bool or array of bools describing for each point whether it is
            contained in the bound.

        """
        return np.all((points >= 0) & (points < 1), axis=-1)

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
        points = self.random_state.random(size=(n_points, self.n_dim))
        return points

    def volume(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        float
            The natural log of the volume of the bound.

        """
        return 0

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'UnitCube'
        group.attrs['n_dim'] = self.n_dim

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
        bound : UnitCube
            The bound.

        """
        bound = cls()

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        bound.n_dim = group.attrs['n_dim']

        return bound


def invert_symmetric_positive_semidefinite_matrix(m):
    """Invert a symmetric positive sem-definite matrix.

    This function is faster than numpy.linalg.inv but does not work for
    arbitrary matrices.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to be inverted.

    Returns
    -------
    m_inv : numpy.ndarray
        Inverse of the matrix.

    """
    m_inv = dpotri(dpotrf(m, False, False)[0])[0]
    m_inv = np.triu(m_inv) + np.triu(m_inv, k=1).T
    return m_inv


def minimum_volume_enclosing_ellipsoid(points, tol=0, max_iterations=1000):
    r"""Find an approximation to the minimum volume enclosing ellipsoid (MVEE).

    This functions finds an approximation to the MVEE for
    :math:`n_{\rm points}` points in :math:`n_{\rm dim}`-dimensional space
    using the Khachiyan algorithm.

    This function is based on
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.116.7691 but
    implemented independently of the corresponding MATLAP implementation or
    other Python ports of the MATLAP code.

    Parameters
    ----------
    points : numpy.ndarray with shape (n_points, n_dim)
        A 2-D array where each row represents a point.
    tol : float, optional
        Tolerance parameter for early stopping. Smaller value lead to more
        accurate results. Default is 0.
    max_iterations : int, optional
        Maximum number of iterations before the function is stopped. Default
        is 1000.

    Returns
    -------
    c : numpy.ndarray
        The position of the ellipsoid.
    A : numpy.ndarray
        The bounds of the ellipsoid in matrix form, i.e.
        :math:`(x - c)^T A (x - c) \leq 1`.

    """
    m, n = points.shape
    q = np.append(points, np.ones(shape=(m, 1)), axis=1)
    u = np.repeat(1.0 / m, m)
    q_outer = np.array([np.outer(q_i, q_i) for q_i in q])
    e = np.diag(np.ones(m))

    for i in range(max_iterations):
        if i % 1000 == 0:
            v = np.einsum('ji,j,jk', q, u, q)
        g = np.einsum('ijk,jk', q_outer,
                      invert_symmetric_positive_semidefinite_matrix(v))
        j = np.argmax(g)
        d_u = e[j] - u
        a = (g[j] - (n + 1)) / ((n + 1) * (g[j] - 1))
        shift = np.linalg.norm(a * d_u)
        v = v * (1 - a) + a * q_outer[j]
        u = u + a * d_u
        if shift <= tol:
            break

    c = np.einsum('i,ij', u, points)
    A_inv = (np.einsum('ji,j,jk', points, u, points) - np.outer(c, c)) * n
    A = np.linalg.inv(A_inv)

    scale = np.amax(np.einsum('...i,ij,...j', points - c, A, points - c))
    A /= scale

    return c, A


class Ellipsoid():
    r"""Ellipsoid in :math:`n_{\rm dim}` dimensions.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    c : numpy.ndarray
        The position of the ellipsoid.
    A : numpy.ndarray
        The bounds of the ellipsoid in matrix form, i.e.
        :math:`(x - c)^T A (x - c) \leq 1`.
    B : numpy.ndarray
        Cholesky decomposition of the inverse of A.
    B_inv : numpy.ndarray
        Inverse of B.
    log_v : float
        Natural log of the volume of the ellipsoid.
    random_state : numpy.random.RandomState instance
        Determines random number generation.

    """

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, random_state=None):
        """Compute the bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid is enlarged by this factor.
            Default is 1.1.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Raises
        ------
        ValueError
            If `enlarge_per_dim` is smaller than unity or the number of points
            does not exceed the number of dimensions.

        Returns
        -------
        bound : Ellipsoid
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        if enlarge_per_dim < 1.0:
            raise ValueError(
                "The 'enlarge_per_dim' factor cannot be smaller than unity.")

        if not points.shape[0] > bound.n_dim:
            raise ValueError('Number of points must be larger than number ' +
                             'dimensions.')

        bound.c, bound.A = minimum_volume_enclosing_ellipsoid(points)
        bound.A /= enlarge_per_dim**2.0
        bound.B = np.linalg.cholesky(np.linalg.inv(bound.A))
        bound.B_inv = np.linalg.inv(bound.B)
        bound.log_v = (np.linalg.slogdet(bound.B)[1] +
                       bound.n_dim * np.log(2.) +
                       bound.n_dim * gammaln(1.5) -
                       gammaln(bound.n_dim / 2.0 + 1))

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        return bound

    def transform(self, points, inverse=False):
        """Transform points into the coordinate frame of the ellipsoid.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.
        inverse : bool, optional
            By default, the coordinates are transformed from the regular
            coordinates to the coordinates in the ellipsoid. If `inverse` is
            set to true, this function does the inverse operation, i.e.
            transform from the ellipsoidal to the regular coordinates.

        Returns
        -------
        points_transformed : numpy.ndarray
            Transformed points.

        """
        if not inverse:
            return np.einsum('ij, ...j', self.B_inv, points - self.c)
        else:
            return np.einsum('ij, ...j', self.B, points) + self.c

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
        return np.sum(self.transform(points)**2, axis=-1) < 1

    def sample(self, n_points=100):
        """Sample points from the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw.

        Returns
        -------
        points : numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        points = self.random_state.normal(size=(n_points, self.n_dim))
        points = points / np.sqrt(np.sum(points**2, axis=1))[:, np.newaxis]
        points *= self.random_state.uniform(size=n_points)[:, np.newaxis]**(
            1.0 / self.n_dim)
        points = self.transform(points, inverse=True)
        return points

    def volume(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v : float
            The natural log of the volume.

        """
        return self.log_v

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'Ellipsoid'
        for key in ['n_dim', 'c', 'A', 'B', 'B_inv', 'log_v']:
            group.attrs[key] = getattr(self, key)

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
        bound : Ellipsoid
            The bound.

        """
        bound = cls()

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        for key in ['n_dim', 'c', 'A', 'B', 'B_inv', 'log_v']:
            setattr(bound, key, group.attrs[key])

        return bound


def ellipsoids_overlap(ells):
    """Determine if ellipsoids overlap.

    This functions is based on ieeexplore.ieee.org/document/6289830.

    Parameters
    ----------
    ells : list
        List of ellipsoids.

    Returns
    -------
    overlapping : bool
        True, if there is overlap between ellipoids and False otherwise.

    """
    c = [ell.c for ell in ells]
    A_inv = [np.linalg.inv(ell.A) for ell in ells]

    for i_1, i_2 in itertools.combinations(range(len(ells)), 2):
        d = c[i_1] - c[i_2]
        def k(s): return (1 - np.dot(np.dot(
            d, np.linalg.inv(A_inv[i_1] / (1 - s) + A_inv[i_2] / s)), d))
        if minimize(k, 0.5, bounds=[(1e-9, 1-1e-9)]).fun > 0:
            return True

    return False


class MultiEllipsoid():
    r"""Union of multiple ellipsoids in :math:`n_{\rm dim}` dimensions.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    ells : list
        List of ellipsoids.
    log_v : list
        Natural log of the volume of each ellipsoid in the union.
    points : list
        The points used to create the union. Used to add more ellipsoids.
    points_sample : numpy.ndarray
        Points that a call to `sample` will return next.
    n_sample : int
        Number of points sampled from all ellipsoids.
    n_reject : int
        Number of points rejected due to overlap.
    enlarge_per_dim : float
        Along each dimension, the ellipsoid is enlarged by this factor.
    n_points_min : int or None
        The minimum number of points each ellipsoid should have. Effectively,
        ellipsoids with less than twice that number will not be split further.
    random_state : numpy.random.RandomState instance
        Determines random number generation.
    """

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, n_points_min=None,
                random_state=None):
        """Compute the bound.

        Upon creation, the bound consists of a single ellipsoid.

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
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Raises
        ------
        ValueError
            If `n_points_min` is smaller than the number of dimensions plus
            one.

        Returns
        -------
        bound : MultiEllipsoid
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        bound.points = [points]
        bound.ells = [Ellipsoid.compute(
            points, enlarge_per_dim=enlarge_per_dim,
            random_state=random_state)]
        bound.log_v = np.array([bound.ells[0].volume()])

        bound.points_sample = np.zeros((0, points.shape[1]))
        bound.n_sample = 0
        bound.n_reject = 0

        bound.enlarge_per_dim = enlarge_per_dim
        if n_points_min is None:
            n_points_min = bound.n_dim + 1

        if n_points_min < bound.n_dim + 1:
            raise ValueError('The number of points per ellipsoid cannot be ' +
                             'smaller than the number of dimensions plus one.')

        bound.n_points_min = n_points_min

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        return bound

    def split_ellipsoid(self, allow_overlap=True):
        """Split the largest ellipsoid to the ellipsoid union.

        Parameters
        ----------
        allow_overlap : bool, optional
            Whether to allow splitting the largest ellipsoid if doing so
            creates overlaps between any ellipsoids. Default is True.

        Returns
        -------
        success : bool
            Whether it was possible to split any ellipsoid.

        """
        split_possible = (np.array([len(points) for points in self.points]) >=
                          2 * self.n_points_min)

        if not np.any(split_possible):
            return False

        index = np.argmax(np.where(split_possible, self.log_v, -np.inf))
        points = self.ells[index].transform(self.points[index])

        if self.random_state != np.random:
            random_state = self.random_state
        else:
            random_state = None
        d = KMeans(
            n_clusters=2, n_init=10, random_state=random_state).fit_transform(
            points)

        labels = np.argmin(d, axis=1)
        if not np.all(np.bincount(labels) >= self.n_points_min):
            label = np.argmin(np.bincount(labels))
            labels[np.argsort(d[:, label] - d[:, label - 1])[
                :self.n_points_min]] = label

        new_ells = self.ells.copy()
        new_ells.pop(index)
        points = self.points[index]
        for label in [0, 1]:
            new_ells.append(Ellipsoid.compute(
                points[labels == label], enlarge_per_dim=self.enlarge_per_dim,
                random_state=self.random_state))

        if not allow_overlap and ellipsoids_overlap(new_ells):
            return False

        self.ells = new_ells
        self.log_v = np.array([ell.volume() for ell in self.ells])
        self.points.pop(index)
        self.points.append(points[labels == 0])
        self.points.append(points[labels == 1])

        # Reset the sampling.
        self.points_sample = np.zeros((0, points.shape[1]))
        self.n_sample = 0
        self.n_reject = 0

        return True

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
        return np.sum([ell.contains(points) for ell in self.ells], axis=0) >= 1

    def sample(self, n_points=100):
        """Sample points from the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw.

        Returns
        -------
        points : numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        while len(self.points_sample) < n_points:
            n_sample = 10000

            p = np.exp(np.array(self.log_v) - logsumexp(self.log_v))
            n_per_ell = self.random_state.multinomial(n_sample, p)

            points = np.vstack([ell.sample(n) for ell, n in
                                zip(self.ells, n_per_ell)])
            self.random_state.shuffle(points)
            n_ell = np.sum([ell.contains(points) for ell in self.ells], axis=0)
            p = 1 - 1.0 / n_ell
            points = points[self.random_state.random(size=n_sample) > p]
            self.points_sample = np.vstack([self.points_sample, points])

            self.n_sample += n_sample
            self.n_reject += n_sample - len(points)

        points = self.points_sample[:n_points]
        self.points_sample = self.points_sample[n_points:]
        return points

    def volume(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v : float
            The natural log of the volume.

        """
        if self.n_sample == 0:
            self.sample()

        return logsumexp(self.log_v) + np.log(
            1.0 - self.n_reject / self.n_sample)

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'MultiEllipsoid'
        for key in ['n_dim', 'log_v', 'n_sample', 'n_reject',
                    'enlarge_per_dim', 'n_points_min']:
            group.attrs[key] = getattr(self, key)

        for i, ell in enumerate(self.ells):
            ell.write(group.create_group('ellipsoid_{}'.format(i)))

        for i in range(len(self.points)):
            group.create_dataset('points_{}'.format(i), data=self.points[i])
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
        bound : MultiEllipsoid
            The bound.

        """
        bound = cls()

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        for key in ['n_dim', 'log_v', 'n_sample', 'n_reject',
                    'enlarge_per_dim', 'n_points_min']:
            setattr(bound, key, group.attrs[key])

        bound.ells = [Ellipsoid.read(
            group['ellipsoid_{}'.format(i)], random_state=bound.random_state)
            for i in range(len(bound.log_v))]
        bound.points = [np.array(group['points_{}'.format(i)]) for i in
                        range(len(bound.log_v))]
        bound.points_sample = np.array(group['points_sample'])

        return bound

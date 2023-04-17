"""Module implementing basic multi-dimensional bounds."""

import numpy as np
from scipy.special import gammaln
from scipy.linalg.lapack import dpotrf, dpotri
from threadpoolctl import threadpool_limits


class UnitCube():
    r"""Unit (hyper)cube bound.

    The :math:`n`-dimensional unit hypercube has :math:`n_{\rm dim}` parameters
    :math:`x_i` with :math:`0 \leq x_i < 1` for all :math:`x_i`.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    rng : numpy.random.Generator
        Determines random number generation.

    """

    @classmethod
    def compute(cls, n_dim, rng=None):
        """Compute the bound.

        Parameters
        ----------
        n_dim : int
            Number of dimensions.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : UnitCube
            The bound.

        """
        bound = cls()
        bound.n_dim = n_dim

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

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

    def sample(self, n_points=100, pool=None):
        """Sample points from the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw. Default is 100.
        pool : ignored
            Not used. Present for API consistency.

        Returns
        -------
        points : numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        points = self.rng.random(size=(n_points, self.n_dim))
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
        bound : UnitCube
            The bound.

        """
        bound = cls()

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

        bound.n_dim = group.attrs['n_dim']

        return bound

    def reset(self, rng=None):
        """Reset random number generation and any progress, if applicable.

        Parameters
        ----------
        rng : None or numpy.random.Generator, optional
            Determines random number generation. If None, random number
            generation is not reset. Default is None.

        """
        if rng is not None:
            self.rng = rng


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
    m_inv_triangle = dpotri(dpotrf(m)[0])[0]
    return m_inv_triangle + m_inv_triangle.T - np.diag(np.diag(m_inv_triangle))


def minimum_volume_enclosing_ellipsoid(points, tol=0, max_iterations=1000):
    r"""Find an approximation to the minimum volume enclosing ellipsoid (MVEE).

    This functions finds an approximation to the MVEE using the Khachiyan
    algorithm.

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
    r"""Ellipsoid bound.

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
    rng : numpy.random.Generator
        Determines random number generation.

    """

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, fast=False,
                max_iterations=1000, rng=None):
        """Compute the bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid is enlarged by this factor.
            Default is 1.1.
        fast : bool, optional
            If True, calculate the bounding ellipsoid from the mean and
            covariance of the points. If False, the ellipsoid (ignoring
            `enlarge_per_dim`) is an approximation to a minimum volume
            enclosing ellipsoid. Default is False.
        max_iterations : int, optional
            Maximum number of iterations before the minimization algorithm for
            the minimum volume enclosing ellipsoid is stopped. Ignored if
            `fast` is True. Default is 1000.
        rng : None or numpy.random.Generator, optional
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

        if not fast:
            with threadpool_limits(limits=1):
                bound.c, bound.A = minimum_volume_enclosing_ellipsoid(
                    points, max_iterations=max_iterations)
        else:
            bound.c = np.mean(points, axis=0)
            bound.A = np.linalg.inv(np.atleast_2d(np.cov(
                points, rowvar=False)))
            scale = np.amax(np.einsum(
                '...i,ij,...j', points - bound.c, bound.A, points - bound.c))
            bound.A /= scale

        bound.A /= enlarge_per_dim**2.0
        bound.B = np.linalg.cholesky(np.linalg.inv(bound.A))
        bound.B_inv = np.linalg.inv(bound.B)
        bound.log_v = (np.linalg.slogdet(bound.B)[1] +
                       bound.n_dim * np.log(2.) +
                       bound.n_dim * gammaln(1.5) -
                       gammaln(bound.n_dim / 2.0 + 1))

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

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
            transform from the ellipsoidal to the regular coordinates. Default
            is False.

        Returns
        -------
        points_t : numpy.ndarray
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
            How many points to draw. Default is 100.

        Returns
        -------
        points : numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        points = self.rng.normal(size=(n_points, self.n_dim))
        points = points / np.sqrt(np.sum(points**2, axis=1))[:, np.newaxis]
        points *= self.rng.uniform(size=n_points)[:, np.newaxis]**(
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
        group: h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'Ellipsoid'
        for key in ['n_dim', 'c', 'A', 'B', 'B_inv', 'log_v']:
            group.attrs[key] = getattr(self, key)

    @classmethod
    def read(cls, group, rng=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group: h5py.Group
            HDF5 group to write to.
        rng: None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound: Ellipsoid
            The bound.

        """
        bound = cls()

        if rng is None:
            bound.rng = np.random.default_rng()
        else:
            bound.rng = rng

        for key in ['n_dim', 'c', 'A', 'B', 'B_inv', 'log_v']:
            setattr(bound, key, group.attrs[key])

        return bound

    def reset(self, rng=None):
        """Reset random number generation and any progress, if applicable.

        Parameters
        ----------
        rng : None or numpy.random.Generator, optional
            Determines random number generation. If None, random number
            generation is not reset. Default is None.

        """
        if rng is not None:
            self.rng = rng


class UnitCubeEllipsoidMixture():
    """Mixture of a unit cube and an ellipsoid.

    Dimensions along which an ellipsoid has a smaller volume than a unit cube
    are defined via ellipsoids and vice versa.

    Attributes
    ----------
    n_dim: int
        Number of dimensions.
    dim_cube: numpy.ndarray
        Whether the boundary in each dimension is defined via the unit cube.
    cube: UnitCube or None
        Unit cube defining the boundary along certain dimensions.
    ellipsoid: Ellipsoid or None
        Ellipsoid defining the boundary along certain dimensions.

    """

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, rng=None):
        """Compute the bound.

        Parameters
        ----------
        points: numpy.ndarray with shape(n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge_per_dim: float, optional
            Along each dimension, the ellipsoid is enlarged by this factor.
            Default is 1.1.
        rng: None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound: UnitCubeEllipsoidMixture
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        kwargs = dict(enlarge_per_dim=enlarge_per_dim, max_iterations=100,
                      rng=rng)

        # First, calculate a bounding ellipsoid along all dimensions.
        ellipsoid = Ellipsoid.compute(points, **kwargs)
        ellipsoid_fast = Ellipsoid.compute(points, fast=True, **kwargs)
        bound.dim_cube = np.zeros(bound.n_dim, dtype=bool)
        # Now, estimate which dimensions have a smaller volume when using the
        # cube.
        for dim in range(bound.n_dim):
            ellipsoid_dim_fast = Ellipsoid.compute(
                np.delete(points, dim, axis=1), fast=True, **kwargs)
            if ellipsoid_fast.volume() > ellipsoid_dim_fast.volume():
                ellipsoid_dim = Ellipsoid.compute(
                    np.delete(points, dim, axis=1), fast=False, **kwargs)
                if ellipsoid.volume() > ellipsoid_dim.volume():
                    bound.dim_cube[dim] = True

        kwargs['max_iterations'] = 1000

        if np.any(bound.dim_cube):
            bound.cube = UnitCube.compute(np.sum(bound.dim_cube), rng=rng)
        else:
            bound.cube = None

        if np.all(bound.dim_cube):
            bound.ellipsoid = None
        else:
            bound.ellipsoid = Ellipsoid.compute(
                points[:, np.arange(bound.n_dim)[~bound.dim_cube]], **kwargs)

        return bound

    def transform(self, points):
        """Transform points into the frame of the cube-ellipsoid mixture.

        Along dimensions where the boundary is not defined via the unit range,
        the coordinates are transformed into the range [-1, +1]. Along all
        other dimensions, they are transformed into the coordinate system of
        the bounding ellipsoid.

        Parameters
        ----------
        points: numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        points_t: numpy.ndarray
            Transformed points.

        """
        points_t = np.copy(points)
        if self.cube is not None:
            idx = np.arange(self.n_dim)[self.dim_cube]
            points_t[:, idx] = points[:, idx] * 2 - 1
        if self.ellipsoid is not None:
            idx = np.arange(self.n_dim)[~self.dim_cube]
            points_t[:, idx] = self.ellipsoid.transform(points[:, idx])
        return points_t

    def contains(self, points):
        """Check whether points are contained in the bound.

        Parameters
        ----------
        points: numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        in_bound: bool or numpy.ndarray
            Bool or array of bools describing for each point whether it is
            contained in the bound.

        """
        in_bound = np.ones(points.shape[:-1], dtype=bool)
        if self.cube is not None:
            idx = np.arange(self.n_dim)[self.dim_cube]
            in_bound = in_bound & self.cube.contains(points[..., idx])
        if self.ellipsoid is not None:
            idx = np.arange(self.n_dim)[~self.dim_cube]
            in_bound = in_bound & self.ellipsoid.contains(points[..., idx])
        return in_bound

    def sample(self, n_points=100):
        """Sample points from the bound.

        Parameters
        ----------
        n_points: int, optional
            How many points to draw. Default is 100.

        Returns
        -------
        points: numpy.ndarray
            Points as two-dimensional array of shape (n_points, n_dim).

        """
        points = np.zeros((n_points, self.n_dim))
        if self.cube is not None:
            idx = np.arange(self.n_dim)[self.dim_cube]
            points[:, idx] = self.cube.sample(n_points)
        if self.ellipsoid is not None:
            idx = np.arange(self.n_dim)[~self.dim_cube]
            points[:, idx] = self.ellipsoid.sample(n_points)
        return points

    def volume(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v: float
            The natural log of the volume.

        """
        if self.ellipsoid is None:
            return 0
        else:
            return self.ellipsoid.volume()

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group: h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'UnitCubeEllipsoidMixture'
        group.attrs['n_dim'] = self.n_dim

        group.create_dataset('dim_cube', data=self.dim_cube)
        if self.cube is not None:
            self.cube.write(group.create_group('cube'))
        if self.ellipsoid is not None:
            self.ellipsoid.write(group.create_group('ellipsoid'))

    @classmethod
    def read(cls, group, rng=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group: h5py.Group
            HDF5 group to write to.
        rng: None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound: UnitCubeEllipsoidMixture
            The bound.

        """
        bound = cls()

        if rng is None:
            rng = np.random.default_rng()

        bound.n_dim = group.attrs['n_dim']
        bound.dim_cube = np.array(group['dim_cube'])

        if np.any(bound.dim_cube):
            bound.cube = UnitCube.read(group['cube'], rng=rng)
        else:
            bound.cube = None

        if not np.all(bound.dim_cube):
            bound.ellipsoid = Ellipsoid.read(group['ellipsoid'], rng=rng)
        else:
            bound.ellipsoid = None

        return bound

    def reset(self, rng=None):
        """Reset random number generation and any progress, if applicable.

        Parameters
        ----------
        rng : None or numpy.random.Generator, optional
            Determines random number generation. If None, random number
            generation is not reset. Default is None.

        """
        if rng is not None:
            if self.ellipsoid is not None:
                self.ellipsoid.reset(rng)
            if self.cube is not None:
                self.cube.reset(rng)

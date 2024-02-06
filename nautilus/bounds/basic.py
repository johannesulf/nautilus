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

    @property
    def log_v(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v : float
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


def minimum_volume_enclosing_ellipsoid(points, n_max=100, n_batch=20):
    r"""Find an approximation to the minimum volume enclosing ellipsoid (MVEE).

    This functions finds an approximation to the MVEE using a modified version
    of the Khachiyan algorithm.

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
    n_max : int, optional
        Maximum number of iterations before the function is stopped. Default
        is 50.
    n_batch : int, optional
        The number of points to evaluate simultaneously. If 1, the algorithm
        is the same as the original Khachiyan algorithm. Default is 20.

    Returns
    -------
    c : numpy.ndarray
        The position of the ellipsoid.
    A : numpy.ndarray
        The bounds of the ellipsoid in matrix form, i.e.
        :math:`(x - c)^T A (x - c) \leq 1`.
    A_inv : numpy.ndarray
        The inverse of `A`.

    """
    n_points, n_dim = points.shape
    q = np.append(points, np.ones(shape=(n_points, 1)), axis=1)
    u = np.repeat(1.0 / n_points, n_points)
    q_outer = np.array([np.outer(q_i, q_i) for q_i in q])

    for i in range(n_max):
        if i % 1000 == 0:
            v = np.einsum('ji,j,jk', q, u, q)
            v_inv = invert_symmetric_positive_semidefinite_matrix(v)
        g = np.einsum('ijk,jk', q_outer, v_inv)
        for j in np.argsort(g)[-n_batch:][::-1]:
            try:
                g = g[j]
            except IndexError:
                g = np.einsum('jk,jk', q_outer[j], v_inv)
            if g < n_dim + 1:
                continue
            a = (g - (n_dim + 1)) / ((n_dim + 1) * (g - 1))
            v = v * (1 - a) + a * q_outer[j]
            v_inv = invert_symmetric_positive_semidefinite_matrix(v)
            u = u * (1 - a) + a * (np.arange(n_points) == j)

    c = np.atleast_1d(np.average(points, weights=u, axis=0))
    A_inv = np.atleast_2d(np.cov(points, aweights=u, rowvar=False, bias=True))
    A = np.linalg.inv(A_inv)

    scale = np.amax(np.einsum('...i,ij,...j', points - c, A, points - c))
    A /= scale
    A_inv *= scale

    return c, A, A_inv


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
    rng : numpy.random.Generator
        Determines random number generation.

    """

    @classmethod
    def compute(cls, points, enlarge_per_dim=1.1, rng=None):
        """Compute the bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid is enlarged by this factor.
            Default is 1.1.
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

        with threadpool_limits(limits=1):
            bound.c, bound.A, A_inv = minimum_volume_enclosing_ellipsoid(
                points)

        bound.A /= enlarge_per_dim**2.0
        A_inv *= enlarge_per_dim**2.0
        bound.B = np.linalg.cholesky(A_inv)
        bound.B_inv = np.linalg.inv(bound.B)

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

    @property
    def log_v(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v : float
            The natural log of the volume.

        """
        return (np.linalg.slogdet(self.B)[1] + self.n_dim * np.log(2.) +
                self.n_dim * gammaln(1.5) - gammaln(self.n_dim / 2.0 + 1))

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group: h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'Ellipsoid'
        for key in ['n_dim', 'c', 'A', 'B', 'B_inv']:
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

        for key in ['n_dim', 'c', 'A', 'B', 'B_inv']:
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

        kwargs = dict(enlarge_per_dim=enlarge_per_dim, rng=rng)

        # First, start by sampling all dimensions using an ellipsoid..
        ellipsoid = Ellipsoid.compute(points, **kwargs)
        bound.dim_cube = np.zeros(bound.n_dim, dtype=bool)

        # Sample dimensions from the unit cube until volume doesn't decrease.
        while np.sum(~bound.dim_cube) > 1:
            c = ellipsoid.c
            A_inv = np.linalg.inv(ellipsoid.A)

            # Start by projecting the current ellipsoid along each dimension to
            # estimate dropping which dimension is most likely to lead to the
            # biggest volume reduction.
            log_v = np.zeros(np.sum(~bound.dim_cube))
            for i in range(np.sum(~bound.dim_cube)):
                points_proj = np.delete(points[:, ~bound.dim_cube], i, axis=1)
                c_proj = np.delete(c, i)
                A_inv_proj = np.delete(np.delete(A_inv, i, axis=0), i, axis=1)
                A_proj = np.linalg.inv(A_inv_proj)
                scale = np.amax(np.einsum('...i,ij,...j', points_proj - c_proj,
                                A_proj, points_proj - c_proj))
                A_proj /= scale
                log_v[i] = np.linalg.slogdet(np.linalg.inv(A_proj))[1]

            dim = np.arange(bound.n_dim)[~bound.dim_cube][np.argmin(log_v)]
            bound.dim_cube[dim] = True
            ellipsoid_proj = Ellipsoid.compute(
                points[:, ~bound.dim_cube], **kwargs)

            if ellipsoid_proj.log_v < ellipsoid.log_v:
                ellipsoid = ellipsoid_proj
            else:
                bound.dim_cube[dim] = False
                break

        # The above algorithm will not necessarily find the optimal combination
        # of which dimensions to sample from the unit cube. In particular, the
        # ellipsoid may have a volume larger than the unit cube. If that's the
        # case, start from the unit cube and add dimensions to sample from
        # ellipsoids until volume doesn't decrease.
        if ellipsoid.log_v > 0:
            ellipsoid = UnitCube.compute(points)
            bound.dim_cube = np.ones(bound.n_dim, dtype=bool)
            # Check which dimensions are better sampled from an ellipsoid,
            # i.e., have smaller volumes.
            tested = np.zeros(bound.n_dim, dtype=bool)
            while ~np.all(tested):
                for dim in np.arange(bound.n_dim)[~tested]:
                    bound.dim_cube[dim] = False
                    tested[dim] = True
                    ellipsoid_test = Ellipsoid.compute(
                        points[:, ~bound.dim_cube], **kwargs)
                    if ellipsoid.log_v > ellipsoid_test.log_v:
                        ellipsoid = ellipsoid_test
                        tested[bound.dim_cube] = False
                    else:
                        bound.dim_cube[dim] = True

        if np.any(bound.dim_cube):
            bound.cube = UnitCube.compute(np.sum(bound.dim_cube), rng=rng)
        else:
            bound.cube = None

        if np.all(bound.dim_cube):
            bound.ellipsoid = None
        else:
            bound.ellipsoid = ellipsoid

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

    @property
    def log_v(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        log_v: float
            The natural log of the volume.

        """
        if self.ellipsoid is None:
            return 0
        else:
            return self.ellipsoid.log_v

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

"""Module implementing multi-dimensional mixture bounds."""

import numpy as np
from scipy.special import logsumexp

from .basic import Ellipsoid


class UnitCubeEllipsoidMixture():
    r"""Overlap of the unit cube and, along certain dimensions, an ellipsoid.

    This bound only draws ellipsoids along those dimensions where an ellipsoid
    would have a smaller volume than the unit range.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    dim_cube : numpy.ndarray
        Dimensions along which the bound is not defined via an ellipsoid.
    dim_ellipsoidipsoid : numpy.ndarray
        Dimensions along which the bound is defined via an ellipsoid.
    ellipsoid : Ellipsoid
        Ellipsoid defining the boundary along certain dimensions.
    n_sample : int
        Number of points sampled.
    n_reject : int
        Number of points rejected because they fall outside the unit cube.
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

        Returns
        -------
        bound : UnitCubeEllipsoidMixture
            The bound.

        """
        bound = cls()
        bound.n_dim = points.shape[1]

        kwargs = dict(enlarge_per_dim=enlarge_per_dim,
                      random_state=random_state)

        # First, calculate a bounding ellipsoid along all dimensions.
        ellipsoid_all = Ellipsoid.compute(points, **kwargs)
        bound.dim_cube = []
        bound.dim_ellipsoid = []
        # Now, determine which dimensions can be dropped.
        for dim in range(bound.n_dim):
            ellipsoid_dim = Ellipsoid.compute(
                np.delete(points, dim, axis=1), **kwargs)
            if ellipsoid_all.volume() > ellipsoid_dim.volume():
                bound.dim_ellipsoid.append(dim)
            else:
                bound.dim_cube.append(dim)

        bound.dim_cube = np.array(bound.dim_cube, dtype=int)
        bound.dim_ellipsoid = np.array(bound.dim_ellipsoid, dtype=int)

        if len(bound.dim_cube) == 0:
            bound.ellipsoid = ellipsoid_all
        else:
            bound.ellipsoid = Ellipsoid.compute(
                points[:, bound.dim_ellipsoid], **kwargs)

        bound.points_sample = np.zeros((0, bound.n_dim))
        bound.n_sample = 0
        bound.n_reject = 0

        if random_state is None:
            bound.random_state = np.random
        else:
            bound.random_state = random_state

        return bound

    def transform(self, points, inverse=False):
        """Transform points into the frame of the cube-ellipsoid overlap.

        Along dimensions where the boundary is not defined via ellipsoids, the
        coordinates are not transformed. Along all other dimensions, they are
        transformed into the coordinate system of the bounding ellipsoid.

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
        points_transformed = np.copy(points)
        points_transformed[:, self.dim_ellipsoid] = self.ellipsoid.transform(
            points[:, self.dim_ellipsoid], inverse=inverse)
        return points_transformed

    def contains(self, points):
        """Check whether points are contained in the cube-ellipsoid overlap.

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
        return (np.all((points >= 0) & (points < 1), axis=-1) &
                (self.ellipsoid.contains(points[:, self.dim_ellipsoid])))

    def sample(self, n_points=100):
        """Sample points from the the ellipsoid.

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

            points = np.zeros((n_points, self.n_dim))
            points[:, self.dim_cube] = self.random_state.random(
                size=(n_points, len(self.dim_cube)))
            points[:, self.dim_ellipsoid] = self.ellipsoid.sample(n_points)
            points = points[np.all((points >= 0) & (points < 1), axis=-1)]
            self.points_sample = np.vstack([self.points_sample, points])

            self.n_sample += n_sample
            self.n_reject += n_sample - len(points)

        points = self.points_sample[:n_points]
        self.points_sample = self.points_sample[n_points:]
        return points

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

        return (self.ellipsoid.volume() + np.log(
            1.0 - self.n_reject / self.n_sample))

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'UnitCubeEllipsoidMixture'
        for key in ['n_dim', 'n_sample', 'n_reject']:
            group.attrs[key] = getattr(self, key)

        group.create_dataset('dim_cube', data=self.dim_cube)
        group.create_dataset('dim_ellipsoid', data=self.dim_ellipsoid)
        self.ellipsoid.write(group.create_group('ellipsoid'))
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
        bound : UnitCubeEllipsoidMixture
            The bound.

        """
        bound = cls()

        if random_state is None:
            bound.random_state = np.random.RandomState()
        else:
            bound.random_state = random_state

        for key in ['n_dim', 'n_sample', 'n_reject']:
            setattr(bound, key, group.attrs[key])

        bound.dim_cube = np.array(group['dim_cube'])
        bound.dim_ellipsoid = np.array(group['dim_ellipsoid'])
        bound.ellipsoid = Ellipsoid.read(
            group['ellipsoid'], random_state=bound.random_state)
        bound.points_sample = np.array(group['points_sample'])

        return bound

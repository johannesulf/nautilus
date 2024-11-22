"""Module implementing periodic boundary conditions."""

import numpy as np


class PhaseShift():
    r"""A simple phase shift to center points inside the unit cube.

    This class analyses points in each periodic dimenions and shifts them
    such that the largest gap between points extends over the boundary.

    Attributes
    ----------
    periodic : numpy.ndarray
        Indices of the parameters that are periodic.
    centers : numpy.ndarray
        The new centers along each periodic dimension.

    """

    @classmethod
    def compute(cls, points, periodic):
        """Compute the phase shift.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        periodic : numpy.ndarray or None
            Indices of the parameters that are periodic.

        Returns
        -------
        bound : PhaseShift
            The bound.

        """
        bound = cls()
        if periodic is None:
            periodic = np.zeros(0, dtype=bool)
        bound.periodic = periodic
        bound.centers = np.zeros(len(periodic))

        for dim in periodic:
            x = np.sort(points[:, dim])
            dx = np.append(np.diff(x), x[0] - (x[-1] - 1))
            bound.centers[dim] = (
                x[np.argmax(dx)] + np.amax(dx) / 2.0 + 0.5) % 1

        return bound

    def transform(self, points, inverse=False):
        """Apply the phase shift.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.
        inverse : bool, optional
            If False, to apply the phase shift. If True, reverse it. Default
            is False.

        Returns
        -------
        points_t : numpy.ndarray
            Transformed points.

        """
        if not np.any(self.periodic):
            return points

        points_t = np.copy(points)
        for dim in self.periodic:
            points_t[:, dim] = (points_t[:, dim] + (-1 if inverse else +1) *
                                (self.centers[dim] + 0.5)) % 1
        return points_t

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['type'] = 'PhaseShift'
        group.attrs['periodic'] = self.periodic
        group.attrs['centers'] = self.centers

    @classmethod
    def read(cls, group, rng=None):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        Returns
        -------
        bound : PhaseShift
            The bound.

        """
        bound = cls()

        bound.periodic = group.attrs['periodic']
        bound.centers = group.attrs['centers']

        return bound

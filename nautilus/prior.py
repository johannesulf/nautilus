"""Module implementing the prior bounds and convencience functions."""

import numpy as np
from scipy.stats import uniform
from scipy.stats.distributions import rv_frozen


class Prior():
    """Helper class to construct prior bounds.

    Attributes
    ----------
    keys : list
        List of model parameters.
    dists : list
        List of distributions each model parameter follows.

    """

    def __init__(self):
        """Initialize a prior without any parameters."""
        self.keys = []
        self.dists = []

    def add_parameter(self, key=None, dist=(0, 1)):
        """Add a model parameter to the prior.

        Parameters
        ----------
        key : str
            Name of the model parameter.
        dist : float, tuple, str or scipy.stats.distributions.rv_frozen
            Distribution the parameter should follow. If a float, the parameter
            is fixed to this value and will not be fitted in any analysis. If
            a tuple, it gives the lower and upper bound of a uniform
            distribution. If a string, the parameter will always be equal
            to the named model parameter. Finally, if rv_frozen, it will
            follow the specified scipy distribution.

        Raises
        ------
        ValueError
            If key already exists in the prior key list.

        """
        if key is None:
            self.keys.append('x_{}'.format(len(self.keys)))
        elif str(key) in self.keys:
            raise ValueError("Key '{}' already in key list.".format(key))
        else:
            self.keys.append(str(key))

        if type(dist) is tuple:
            self.dists.append(uniform(loc=dist[0], scale=dist[1] - dist[0]))
        else:
            self.dists.append(dist)

    def dimensionality(self):
        """Determine the number of free model parameters.

        Returns
        -------
        n_dim : int
            The number of free model parameters.
        """
        return sum(type(dist) is rv_frozen for dist in self.dists)

    def unit_to_physical(self, points):
        """Transfer points from the unit hypercube to the prior volume.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points in the unit hypercube. If more than one-dimensional, each
            row represents a point.

        Returns
        -------
        phys_points : numpy.ndarray
            Points transferred into the prior volume. Has the same shape as
            `points`.

        Raises
        ------
        ValueError
            If dimensionality of `points` does not match the prior.
        """
        phys_points = np.zeros_like(points)

        try:
            assert self.dimensionality() == points.shape[-1]
        except AssertionError:
            raise ValueError('Dimensionality of points does not match prior.')

        i = 0
        for dist in self.dists:
            if type(dist) is rv_frozen:
                phys_points[..., i] = dist.isf(1 - points[..., i])
                i = i + 1

        return phys_points

    def physical_to_structure(self, phys_points):
        """Express points in the prior volume as a structured data type.

        Parameters
        ----------
        phys_points : numpy.ndarray
            Points in the prior volume. If more than one-dimensional, each
            row represents a point.

        Returns
        -------
        struct_point : dict or numpy.ndarray
            Points as a structured data type. If `phys_points` has one
            dimension, this will be a dictionary. Otherwise, it will be a
            structured numpy array. Each model parameter, including fixed ones,
            can be accessed via their key.

        """
        if phys_points.ndim == 1:
            struct_points = {}
        else:
            struct_points = np.zeros(
                phys_points.shape[0], dtype=[(key, np.double) for key in
                                             self.keys])

        i = 0
        for key, dist in zip(self.keys, self.dists):
            if type(dist) is rv_frozen:
                struct_points[key] = phys_points[..., i]
                i = i + 1
            elif type(dist) is int or type(dist) is float:
                struct_points[key] = np.ones(phys_points[..., 0].shape) * dist

        for key, dist in zip(self.keys, self.dists):
            if type(dist) is str:
                struct_points[key] = struct_points[dist]

        return struct_points

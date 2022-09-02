"""Module implementing the prior bounds and convencience functions."""

import numbers
import numpy as np
from scipy.stats import uniform


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
        key : str or None
            Name of the model parameter. If None, the key name will be `x_i`,
            where i is a number.
        dist : float, tuple, str or object
            Distribution the parameter should follow. If a float, the parameter
            is fixed to this value and will not be fitted in any analysis. If
            a tuple, it gives the lower and upper bound of a uniform
            distribution. If a string, the parameter will always be equal
            to the named model parameter. Finally, if an object, it must have
            a `isf` attribute, i.e. the inverse survival function.

        Raises
        ------
        TypeError
            If `key` or `dist` is not the correct type.
        ValueError
            If a new key already exists in the key list or if `dist` is a
            string but does not refer to a previously defined key.
        """
        if key is None:
            self.keys.append('x_{}'.format(len(self.keys)))
        elif not isinstance(key, str):
            raise TypeError("Keyword argument 'key' must be a string.")
        elif key in self.keys:
            raise ValueError("Key '{}' already in key list.".format(key))
        else:
            self.keys.append(key)

        if isinstance(dist, tuple):
            self.dists.append(uniform(loc=dist[0], scale=dist[1] - dist[0]))
        elif isinstance(dist, numbers.Number) or hasattr(dist, 'isf'):
            self.dists.append(dist)
        elif isinstance(dist, str):
            if dist not in self.keys or dist == str(key):
                raise ValueError('Key {} not defined previously.'.format(dist))
            while isinstance(self.dists[self.keys.index(dist)], str):
                dist = self.dists[self.keys.index(dist)]
            self.dists.append(dist)
        else:
            raise TypeError("Keyword argument 'dist' does not have the " +
                            "correct type")

    def dimensionality(self):
        """Determine the number of free model parameters.

        Returns
        -------
        n_dim : int
            The number of free model parameters.
        """
        return sum(not isinstance(dist, (numbers.Number, str)) for dist in
                   self.dists)

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
            if hasattr(dist, 'isf'):
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
        struct_points : dict or numpy.ndarray
            Points as a structured data type. If `phys_points` has one
            dimension, this will be a dictionary. Otherwise, it will be a
            structured numpy array. Each model parameter, including fixed ones,
            can be accessed via their key.

        Raises
        ------
        ValueError
            If dimensionality of `points` does not match the prior.

        """
        try:
            assert self.dimensionality() == phys_points.shape[-1]
        except AssertionError:
            raise ValueError('Dimensionality of points does not match prior.')

        if phys_points.ndim == 1:
            struct_points = {}
        else:
            struct_points = np.zeros(
                phys_points.shape[0], dtype=[(key, float) for key in
                                             self.keys])

        i = 0
        for key, dist in zip(self.keys, self.dists):
            if hasattr(dist, 'isf'):
                struct_points[key] = phys_points[..., i]
                i = i + 1
            elif isinstance(dist, numbers.Number):
                struct_points[key] = np.ones(phys_points[..., 0].shape) * dist

        for key, dist in zip(self.keys, self.dists):
            if isinstance(dist, str):
                struct_points[key] = struct_points[dist]

        return struct_points

"""Module implementing multi-dimensional bounds."""

import itertools
import numpy as np
from scipy.special import gammaln
from scipy.special import logsumexp
from scipy.stats import percentileofscore
from scipy.linalg.lapack import dpotrf, dpotri
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from .neural import NeuralNetworkEmulator


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

    def __init__(self, n_dim, random_state=None):
        r"""Initialize the :math:`n_{\rm dim}`-dimensional unit hypercube.

        Parameters
        ----------
        n_dim : int
            Number of dimensions.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        """
        self.n_dim = n_dim

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

    def contains(self, points):
        """Check whether points are contained in the unit hypercube.

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

    def sample(self, n_points=1):
        """Sample points from the unit hypercube.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw. Default is 1.

        Returns
        -------
        points : numpy.ndarray
            If `n_points` is larger than 1, two-dimensional array of shape
            (n_points, n_dim). Otherwise, a one-dimensional array of shape
            (n_dim).

        """
        points = self.random_state.random(size=(n_points, self.n_dim))
        return np.squeeze(points)

    def volume(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        float
            The natural log of the volume of the bound.

        """
        return 0


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

    def __init__(self, points, enlarge=2.0, random_state=None):
        r"""Initialize an :math:`n_{\rm dim}`-dimensional ellipsoid.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoid around the points
            is increased by this factor. Default is 2.0
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        Raises
        ------
        ValueError
            If `enlarge` is smaller than unity or the number of points does
            not exceed the number of dimensions.

        """
        self.n_dim = points.shape[1]

        try:
            assert enlarge >= 1
        except AssertionError:
            raise ValueError(
                "The 'enlarge' factor cannot be smaller than unity.")

        if not points.shape[0] > self.n_dim:
            raise ValueError('Number of points must be larger than number ' +
                             'dimensions.')

        self.c, self.A = minimum_volume_enclosing_ellipsoid(points)
        self.A /= enlarge**(2.0 / self.n_dim)
        self.B = np.linalg.cholesky(np.linalg.inv(self.A))
        self.B_inv = np.linalg.inv(self.B)
        self.log_v = (np.linalg.slogdet(self.B)[1] +
                      self.n_dim * np.log(2.) +
                      self.n_dim * gammaln(1.5) -
                      gammaln(self.n_dim / 2.0 + 1))

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

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
        """Check whether points are contained in the ellipsoid.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        in_bound : bool or numpy.ndarray
            Bool or array of bools describing for each point whether it is
            contained in the ellipsoid.

        """
        return np.sum(self.transform(points)**2, axis=-1) < 1

    def sample(self, n_points=1):
        """Sample points from the the ellipsoid.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw.

        Returns
        -------
        points : numpy.ndarray
            If `n_points` is larger than 1, two-dimensional array of shape
            (n_points, n_dim). Otherwise, a one-dimensional array of shape
            (n_dim).

        """
        points = self.random_state.normal(size=(n_points, self.n_dim))
        points = points / np.sqrt(np.sum(points**2, axis=1))[:, np.newaxis]
        points *= self.random_state.uniform(size=n_points)[:, np.newaxis]**(
            1.0 / self.n_dim)
        points = self.transform(points, inverse=True)
        return np.squeeze(points)

    def volume(self):
        """Return the natural log of the volume of the ellipsoid.

        Returns
        -------
        log_v : float
            The natural log of the volume.

        """
        return self.log_v


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
        k = lambda s: (1 - np.dot(np.dot(
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
    points : numpy.ndarray
        The points used to create the union. Used to add more ellipsoids.
    points_sample : numpy.ndarray
        Points that a call to `sample` will return next.
    n_sample : int
        Number of points sampled from all ellipsoids.
    n_reject : int
        Number of points rejected due to overlap.
    enlarge : float
        The factor by which the volume of the minimum enclosing ellipsoids is
        increased.
    n_points_min : int or None
        The minimum number of points each ellipsoid should have. Effectively,
        ellipsoids with less than twice that number will not be split further.
    random_state : numpy.random.RandomState instance
        Determines random number generation.
    """

    def __init__(self, points, enlarge=2.0, n_points_min=None,
                 random_state=None):
        r"""Initialize a union of :math:`n_{\rm dim}`-dimensional ellipsoids.

        Upon creation, the union consists of a single ellipsoid.

        Parameters
        ----------
        points : numpy.ndarray with shape (n_points, n_dim)
            A 2-D array where each row represents a point.
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoids around the points
            is increased by this factor. Default is 2.0.
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

        """
        self.n_dim = points.shape[1]

        self.points = [points]
        self.ells = [Ellipsoid(points, enlarge=enlarge,
                               random_state=random_state)]
        self.log_v = [self.ells[0].volume()]

        self.points_sample = np.zeros((0, points.shape[1]))
        self.n_sample = 0
        self.n_reject = 0

        self.enlarge = enlarge
        if n_points_min is None:
            n_points_min = self.n_dim + 1

        if n_points_min < self.n_dim + 1:
            raise ValueError('The number of points per ellipsoid cannot be ' +
                             'smaller than the number of dimensions plus one.')

        self.n_points_min = n_points_min

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

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
        split_possible = (np.array([len(points) for points in self.points]) >
                          2 * self.n_points_min)

        if not np.any(split_possible):
            return False

        index = np.argmax(np.where(split_possible, self.log_v, -np.inf))
        points = self.ells[index].transform(self.points[index])

        d = KMeans(n_clusters=2, random_state=self.random_state).fit_transform(
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
            new_ells.append(Ellipsoid(
                points[labels == label], enlarge=self.enlarge,
                random_state=self.random_state))

        if not allow_overlap and ellipsoids_overlap(new_ells):
            return False

        self.ells = new_ells
        self.log_v = [ell.volume() for ell in self.ells]
        self.points.pop(index)
        self.points.append(points[labels == 0])
        self.points.append(points[labels == 1])

        # Reset the sampling.
        self.points_sample = np.zeros((0, points.shape[1]))
        self.n_sample = 0
        self.n_reject = 0

        return True

    def contains(self, points):
        """Check whether points are contained in the ellipsoid union.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        in_bound : bool or numpy.ndarray
            Bool or array of bools describing for each point whether it is
            contained in the ellipsoid union.

        """
        return np.sum([ell.contains(points) for ell in self.ells], axis=0) >= 1

    def sample(self, n_points=1):
        """Sample points from the the ellipsoid.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw.

        Returns
        -------
        points : numpy.ndarray
            If `n_points` is larger than 1, two-dimensional array of shape
            (n_points, n_dim). Otherwise, a one-dimensional array of shape
            (n_dim).

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
        return np.squeeze(points)

    def volume(self):
        """Return the natural log of the volume of the ellipsoid union.

        Returns
        -------
        log_v : float
            The natural log of the volume.

        """
        if self.n_sample == 0:
            self.sample()

        return logsumexp(self.log_v) + np.log(
            1.0 - self.n_reject / self.n_sample)


class NeuralBound():
    r"""Neural network-based bound in :math:`n_{\rm dim}` dimensions.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    random_state : numpy.random or numpy.random.RandomState instance
        Determines random number generation.
    ellipsoid : Ellipsoid
        Bounding ellipsoid drawn around the live points.
    emulator : object
        Emulator based on `sklearn.neural_network.MLPRegressor` used to fit and
        predict likelihood scores.
    score_predict_min : float
        Minimum score predicted by the emulator to be considered part of the
        bound.
    """

    def __init__(self, points, log_l, log_l_min, ellipsoid=None, enlarge=2.0,
                 neural_network_kwargs={}, neural_network_thread_limit=1,
                 random_state=None):
        """Initialize a neural network-based bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        log_l : numpy.ndarray of length m
            Likelihood of each point.
        log_l_min : float
            Target likelihood threshold of the bound.
        ellipsoid : Ellipsoid or None, optional
            If given, use the provided ellipsoid to enclose the points. If
            None, determine a bounding ellipsoid from the given points. Default
            is None.
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoid around the points
            with likelihood larger than the target likelihood is increased by
            this factor. Default is 2.0.
        neural_network_kwargs : dict, optional
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`. By default, no keyword
            arguments are passed to the constructor.
        neural_network_thread_limit : int or None, optional
            Maximum number of threads used by `sklearn`. If None, no limits
            are applied. Default is 1.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        """
        self.n_dim = points.shape[1]

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        # Determine the outer bound.
        if ellipsoid is None:
            self.ellipsoid = Ellipsoid(
                points[log_l > log_l_min], enlarge=enlarge,
                random_state=self.random_state)
        else:
            self.ellipsoid = ellipsoid

        # Train the network.
        select = self.ellipsoid.contains(points)
        points = points[select]
        log_l = log_l[select]

        points_t = self.ellipsoid.transform(points)
        perc = np.argsort(np.argsort(log_l)) / float(len(log_l))
        perc_min = percentileofscore(log_l, log_l_min) / 100
        score = np.zeros(len(points))
        select = perc < perc_min
        if np.any(select):
            score[select] = 0.5 * (perc[select] / perc_min)
        score[~select] = 1 - 0.5 * (1 - perc[~select]) / (1 - perc_min)
        self.emulator = NeuralNetworkEmulator(
            points_t, score, neural_network_kwargs=neural_network_kwargs,
            neural_network_thread_limit=neural_network_thread_limit)


        self.score_predict_min = np.polyval(np.polyfit(
            score, self.emulator.predict(points_t), 3), 0.5)

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

        points = self.ellipsoid.transform(points)
        in_bound = np.sum(points**2, axis=-1) < 1
        if np.any(in_bound):
            in_bound[in_bound] = (self.emulator.predict(points[in_bound]) >
                                  self.score_predict_min)

        return np.squeeze(in_bound)


class NautilusBound():
    """Union of multiple non-overlapping neural network-based bounds.

    The bound is the overlap of the union of multiple neural network-based
    bounds and the unit hypercube.

    Attributes
    ----------
    log_v : list
        List of the natural log of the volumes of each bound.
    nbounds : list
        List of the individual neural network-based bounds.
    sample_bounds : tuple
        Outer bounds used for sampling. The first one is used for sampling
        while any points must lie in both to be part of the bound.
    random_state : None or numpy.random.RandomState instance
        Determines random number generation.
    """

    def __init__(self, points, log_l, log_l_min, log_v_target, enlarge=2.0,
                 n_points_min=None, split_threshold=100,
                 use_neural_networks=True, neural_network_kwargs={},
                 neural_network_thread_limit=1, random_state=None):
        """Initialize a union of multiple neural network-based bounds.

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
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoid around the points
            with likelihood larger than the target likelihood is increased by
            this factor. Default is 2.0.
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
        use_neural_networks : bool, optional
            Whether to use neural network emulators in the construction of the
            bound. Default is True.
        neural_network_kwargs : dict, optional
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`. By default, no keyword
            arguments are passed to the constructor.
        neural_network_thread_limit : int or None, optional
            Maximum number of threads used by `sklearn`. If None, no limits
            are applied. Default is 1.
        random_state : None or numpy.random.RandomState instance, optional
            Determines random number generation. Default is None.

        """
        mell = MultiEllipsoid(points[log_l > log_l_min], enlarge=enlarge,
                              random_state=random_state)
        cube = UnitCube(points.shape[-1])

        while mell.split_ellipsoid(allow_overlap=False):
            pass

        self.nbounds = []

        if use_neural_networks:
            for ell in mell.ells:
                select = ell.contains(points)
                self.nbounds.append(NeuralBound(
                    points[select], log_l[select], log_l_min, ellipsoid=ell,
                    enlarge=enlarge,
                    neural_network_kwargs=neural_network_kwargs,
                    neural_network_thread_limit=neural_network_thread_limit,
                    random_state=random_state))

        while min(mell.volume(), 0.0) - log_v_target > np.log(
                split_threshold * enlarge):
            if mell.volume() >= 0:
                points_test = cube.sample(10000)
                log_v = cube.volume() + np.log(
                    np.mean(mell.contains(points_test)))
            else:
                points_test = mell.sample(10000)
                log_v = mell.volume() + np.log(
                    np.mean(cube.contains(points_test)))
            if log_v - log_v_target > np.log(100 * enlarge):
                if not mell.split_ellipsoid():
                    break
            else:
                break

        if mell.volume() >= 0:
            self.sample_bounds = (cube, mell)
        else:
            self.sample_bounds = (mell, cube)

        self.log_v = self.sample_bounds[0].volume()

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        self.points_sample = np.zeros((0, points.shape[1]))
        self.n_sample = 0
        self.n_reject = 0

    def contains(self, points):
        """Check whether points are contained in the ellipsoid.

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
        in_bound = (self.sample_bounds[0].contains(points) &
                    self.sample_bounds[1].contains(points))
        if len(self.nbounds) > 0:
            in_bound = in_bound & np.any(np.vstack(
                [bound.contains(points) for bound in self.nbounds]), axis=0)
        return in_bound

    def sample(self, n_points=1):
        """Sample points from the the bound.

        Parameters
        ----------
        n_points : int, optional
            How many points to draw.

        Returns
        -------
        points : numpy.ndarray
            If `n_points` is larger than 1, two-dimensional array of shape
            (n_points, n_dim). Otherwise, a one-dimensional array of shape
            (n_dim).

        """
        while len(self.points_sample) < n_points:
            n_sample = 10000
            points = self.sample_bounds[0].sample(n_sample)
            points = points[self.sample_bounds[1].contains(points)]
            points = points[self.contains(points)]
            self.points_sample = np.vstack([self.points_sample, points])
            self.n_sample += n_sample
            self.n_reject += n_sample - len(points)

        # Update volumes since they might have gotten more accurate.
        self.log_v = self.sample_bounds[0].volume()

        points = self.points_sample[:n_points]
        self.points_sample = self.points_sample[n_points:]
        return np.squeeze(points)

    def volume(self):
        """Return the natural log of the volume.

        Returns
        -------
        log_v : float
            An estimate of the natural log of the volume. Will become more
            accurate as more points are sampled.

        """
        if self.n_sample == 0:
            self.sample()

        return logsumexp(self.log_v) + np.log(
            1.0 - self.n_reject / self.n_sample)

    def number_of_networks_and_ellipsoids(self):
        """Return the number of neural networks and sample ellipsoids.

        Returns
        -------
        n_neural : int
            The number of neural networks.
        n_sample : int
            The number of sample ellipsoids.
        """
        n_neural = len(self.nbounds)

        if isinstance(self.sample_bounds[0], UnitCube):
            i = 1
        else:
            i = 0
        n_sample = len(self.sample_bounds[i].ells)

        return n_neural, n_sample

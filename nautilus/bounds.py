import itertools
import numpy as np
from scipy.special import gammaln
from scipy.special import logsumexp
from scipy.stats import percentileofscore
from scipy.linalg.lapack import dpotrf, dpotri
from sklearn.mixture import GaussianMixture

from .neural import NeuralNetworkEmulator


class UnitCube():
    r"""Unit (hyper)cube bound in :math:`n` dimensions.

    The :math:`n`-dimensional unit hypercube has :math:`n` parameters
    :math:`x_i` with :math:`0 \leq x_i < 1` for all :math:`x_i`.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    """

    def __init__(self, n):
        """Initialize the :math:`n`-dimensional unit hypercube.

        Attributes
        ----------
        n : int
            Number of dimensions.
        """

        self.n_dim = n

    def contains(self, points):
        """Check whether points are contained in the unit hypercube.

        Attributes
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        bool or numpy.ndarray
            Bool or array of bools describing for each point it is contained in
            the bound.
        """

        return np.all((points >= 0) & (points < 1), axis=-1)

    def sample(self, batch_size=1000, random_state=np.random):
        """Sample points from the unit hypercube.

        Attributes
        ----------
        batch_size : int, optional
            How many points to draw.
        random_state : numpy.random or numpy.random.RandomState instance
            Determines random number generation.

        Returns
        -------
        numpy.ndarray with shape (batch_size, n)
            Array where each row represents a point.
        """

        points = random_state.random(size=(batch_size, self.n_dim))
        return points

    def volume(self):
        """Return the natural log of the volume of the bound.

        Returns
        -------
        float
            The natural log of the volume of the bound.
        """

        return 0


def invert_symmetric_positive_semidefinite_matrix(m):
    """Invert a symmetric positive sem-definite matrix. This function is
    faster than numpy.linalg.inv but does not work for arbitrary matrices.

    Attributes
    ----------
    m : numpy.ndarray
        Matrix to be inverted.

    Returns
    -------
    numpy.ndarray
        Inverse of the matrix.
    """

    m_inv = dpotri(dpotrf(m, False, False)[0])[0]
    m_inv = np.triu(m_inv) + np.triu(m_inv, k=1).T
    return m_inv


def minimum_volume_enclosing_ellipsoid(points, tol=0, max_iterations=1000):
    r"""Find an approximation to the minimum bounding ellipsoid to:math:`m`
    points in :math:`n`-dimensional space using the Khachiyan algorithm.

    This function is based on
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.116.7691 but
    implemented independently of the corresponding MATLAP implementation or
    other Python ports of the MATLAP code.

    Attributes
    ----------
    points : numpy.ndarray with shape (m, n)
        A 2-D array where each row represents a point.

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
    r"""Union of an ellipsoid and the unit hypercube in :math:`n` dimensions.

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
    sample_ellipsoid : bool
        Whether sampling starts from the unit hypercube or the ellipsoid.
    """

    def __init__(self, points, enlarge=1.1):
        """Initialize an :math:`n`-dimensional overlap of an ellipsoid and the
        unit hypercube.

        Attributes
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoid around the points
            is increased by this factor. Default is 1.1
        """

        self.n_dim = points.shape[1]

        try:
            assert enlarge >= 1
        except AssertionError:
            raise Exception(
                "The 'enlarge' factor cannot be smaller than unity.")

        if points.shape[0] <= self.n_dim + 1:
            self.log_v = -np.inf
        else:
            try:
                self.c, self.A = minimum_volume_enclosing_ellipsoid(points)
                self.A /= enlarge**(1.0 / self.n_dim)
                self.B = np.linalg.cholesky(np.linalg.inv(self.A))
                self.B_inv = np.linalg.inv(self.B)
                self.log_v = (np.linalg.slogdet(self.B)[1] +
                              self.n_dim * np.log(2.) +
                              self.n_dim * gammaln(1.5) -
                              gammaln(self.n_dim / 2.0 + 1))
            except np.linalg.LinAlgError:
                self.log_v = -np.inf

        if self.log_v > 0:
            self.sample_ellipsoid = False
            self.log_v = 0
        else:
            self.sample_ellipsoid = True

    def transform(self, points):
        """Transform points into the coordinate frame of the ellipsoid.

        Attributes
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        numpy.ndarray
            Transformed points.
        """

        return np.einsum('ij, ...j', self.B_inv, points - self.c)

    def contains(self, points):
        """Check whether points are contained in the ellipsoid and the unit
        hypercube.

        Attributes
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        bool or numpy.ndarray
            Bool or array of bools describing for each point it is contained in
            the bound.
        """

        if self.log_v != -np.inf:
            return (np.all((points >= 0) & (points < 1), axis=-1) &
                    (np.sum(self.transform(points)**2, axis=-1) < 1))
        else:
            return np.zeros(len(points), dtype=bool)

    def sample(self, batch_size=1000, random_state=np.random):
        """Sample points from the overlap of the ellipsoid and the unit
        hypercube.

        Attributes
        ----------
        batch_size : int, optional
            How many points to draw at most.
        random_state : numpy.random or numpy.random.RandomState instance
            Determines random number generation.

        Returns
        -------
        numpy.ndarray with shape (batch_size, n)
            Array where each row represents a point. Sometimes, :math:`m` will
            be smaller than `batch_size`. In particular, :math:`m` /
            `batch_size` indicates the ratio of the true volume of the bound
            compared to what is returned by
            :func:`~nautilus.bounds.Ellipsoid.volume`.
        """

        if self.log_v != -np.inf:
            if self.sample_ellipsoid:
                points = random_state.normal(size=(batch_size, self.n_dim))
                points = points / np.sqrt(np.sum(points**2, axis=1))[
                    :, np.newaxis]
                points *= random_state.uniform(size=batch_size)[
                    :, np.newaxis]**(1.0 / self.n_dim)
                points = np.einsum('ij, ...j', self.B, points) + self.c
            else:
                points = random_state.random(size=(batch_size, self.n_dim))
            if len(points) == 0:
                return points
            return points[self.contains(points)]
        else:
            raise Exception('Cannot sample from zero-volume ellipsoid.')

    def volume(self):
        """Return the natural log of the volume of either the unit hypercube
        or the ellipsoid, whichever has the smaller volume.

        Returns
        -------
        float
            The natural log of the volume.
        """

        return self.log_v


class NeuralBound():
    """Neural network-based bound in :math:`n` dimensions.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    ellipsoid : Ellipsoid
        Bounding ellipsoid drawn around the live points.
    log_v : float
        Natural log of the volume of the enclosing volume.
    emulator : object
        Emulator used to fit and predict likelihoods.
    """

    def __init__(self, points, log_l, log_l_min, enlarge=1.5):
        """Initialize a neural network-based bound in :math:`n` dimensions.

        Attributes
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        log_l : numpy.ndarray of length m
            Likelihood of each point.
        log_l_min : float
            Target likelihood threshold of the bound.
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoid around the points
            with likelihood larger than the target likelihood is increased by
            this factor. Default is 1.5.
        """

        self.n_dim = points.shape[1]

        self.ellipsoid = Ellipsoid(points[log_l > log_l_min], enlarge=enlarge)
        self.log_v = self.ellipsoid.volume()

        use = self.ellipsoid.contains(points)
        points = points[use]
        log_l = log_l[use]

        points = self.ellipsoid.transform(points)
        perc = np.argsort(np.argsort(log_l)) / float(len(log_l))
        self.emulator = NeuralNetworkEmulator(points, perc)

        perc_predict_min = percentileofscore(log_l, log_l_min) / 100
        self.perc_predict_min = np.polyval(np.polyfit(
            perc, self.emulator.predict(points), 3), perc_predict_min)

    def contains(self, points):
        """Check whether points are contained in the bound.

        Attributes
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        bool or numpy.ndarray
            Bool or array of bools describing for each point if it is contained
            in the bound.
        """

        if self.log_v != -np.inf:

            in_cube = np.all((points >= 0) & (points < 1), axis=-1)
            points_transformed = self.ellipsoid.transform(points[in_cube])
            in_ell = np.sum(points_transformed**2, axis=-1) < 1
            points_transformed = points_transformed[in_ell]
            if np.any(in_ell):
                in_emu = (self.emulator.predict(points_transformed) >
                          self.perc_predict_min)

            in_bound = in_cube
            in_bound[in_bound] = in_ell
            if np.any(in_ell):
                in_bound[in_bound] = in_emu

            return in_bound
        else:
            return np.zeros(len(points), dtype=bool)

    def sample(self, batch_size=1000, random_state=np.random):
        """Sample points from the bound.

        Attributes
        ----------
        batch_size : int, optional
            How many points to draw at most.
        random_state : numpy.random or numpy.random.RandomState instance
            Determines random number generation.

        Returns
        -------
        numpy.ndarray with shape (m, n)
            Array where each row represents a point. Sometimes, :math:`m` will
            be smaller than `batch_size`. In particular, :math:`m` /
            `batch_size` indicates the ratio of the true volume of the bound
            compared to what is returned by
            :func:`~nautilus.bounds.NeuralBound.volume`.
        """

        if self.log_v != -np.inf:
            points = self.ellipsoid.sample(
                batch_size, random_state=random_state)
            if len(points) == 0:
                return points
            mask = (self.emulator.predict(self.ellipsoid.transform(points)) <
                    self.perc_predict_min)
            return points[~mask]
        else:
            raise Exception('Cannot sample from zero-volume ellipsoid.')

    def volume(self):
        """Return the natural log of the volume of the enclosing ellipsoid or
        unit hypercube, whichever is smaller. This does not account for the
        additional reduction of the volume by the use of the internal emulator.

        Returns
        -------
        float
            The natural log of the volume.
        """

        return self.log_v


def fast_enclosing_ellipsoid(points):
    r"""Find an enclosing ellipsoid based on the mean and covariance of the
    sample.

    Attributes
    ----------
    points : numpy.ndarray with shape (m, n)
        A 2-D array where each row represents a point.

    Returns
    -------
    c : numpy.ndarray
        The position of the ellipsoid.
    A : numpy.ndarray
        The bounds of the ellipsoid in matrix form, i.e.
        :math:`(x - c)^T A (x - c) \leq 1`.
    """

    c = np.mean(points, axis=0)
    A = invert_symmetric_positive_semidefinite_matrix(
        np.cov(points, rowvar=False))
    A /= np.sqrt(np.amax(np.einsum(
        '...j, ...j', np.einsum('ij,kj', c - points, A), c - points)))

    return c, A


def split_clusters(points, d_min=10, random_state=np.random):
    """Split points into multiple, non-overlapping clusters, if possible.
    Non-overlapping means that the enclosing ellipsoids of the groups do not
    overlap with each other.

    Attributes
    ----------
    points : numpy.ndarray
        A 2-D array containing a collection of points. Each row represents a
        point.
    d_min : float, optional
        The minimum separation between the centers of two enclosing
        ellipsoids in units of the sum of the radii of the two encluding
        ellpsoids. Default is 10.
    random_state : numpy.random or numpy.random.RandomState instance
        Determines random number generation.

    Returns
    -------
    list of numpy.ndarray
        List of groups of points. Each entry corresponds to one group.
    """

    n_cluster = 1
    labels = np.zeros(len(points), dtype=int)
    n_cluster_test = 2

    while n_cluster_test < n_cluster + 3:

        gmm = GaussianMixture(n_cluster_test, random_state=random_state)
        labels_test = gmm.fit_predict(points)

        replace = True

        for la in range(n_cluster_test):
            replace &= np.sum(labels_test == la) > points.shape[1] * 2

        if not replace:
            n_cluster_test += 1
            continue

        c = []
        A = []
        for la in range(n_cluster_test):
            c_l, A_l = fast_enclosing_ellipsoid(points[labels_test == la])
            c.append(c_l)
            A.append(A_l)

        for l_1, l_2 in itertools.combinations(range(n_cluster_test), 2):
            d = c[l_1] - c[l_2]
            x_1 = np.sqrt(np.dot(np.dot(d, A[l_1]), d))
            x_2 = np.sqrt(np.dot(np.dot(d, A[l_2]), d))
            replace &= 1.0 / x_1 + 1.0 / x_2 <= 1.0 / d_min

        n_cluster_test += 1

        if not replace:
            continue

        labels = labels_test
        n_cluster = n_cluster_test

    return labels


class MultiNeuralBound():

    """Union of multiple neural network-based bound in :math:`n` dimensions.

    Attributes
    ----------
    log_v : list
        List of the natural log of the volumes of the enclosing volumes.
    bounds : list
        List of the individual neural network-based bounds.
    """

    def __init__(self, points, log_l, log_l_min, enlarge=1.5, d_min=10):
        """Initialize a union of multiple neural network-based bound in
        :math:`n` dimensions.

        Attributes
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        log_l : numpy.ndarray of length m
            Likelihood of each point.
        log_l_min : float
            Target likelihood threshold of the bound.
        enlarge : float, optional
            The volume of the minimum enclosing ellipsoid around the points
            with likelihood larger than the target likelihood is increased by
            this factor. Default is 1.5.
        d_min : float, optional
            The minimum separation between the centers of two enclosing
            ellipsoids in units of the sum of the radii of the two encluding
            ellpsoids. Default is 10.
        """

        labels = split_clusters(points[log_l > log_l_min], d_min=d_min)
        self.log_v = []
        self.bounds = []

        for la in range(len(np.unique(labels))):
            use = np.ones(len(points), dtype=bool)
            use[log_l > log_l_min] = (labels == la)
            self.bounds.append(NeuralBound(
                points[use], log_l[use], log_l_min, enlarge=enlarge))
            self.log_v.append(self.bounds[-1].volume())

    def contains(self, points):
        """Check whether points are contained in the ellipsoid.

        Attributes
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        bool or numpy.ndarray
            Bool or array of bools describing for each point it is contained in
            the bound.
        """
        return np.any(np.vstack(
            [bound.contains(points) for bound in self.bounds]), axis=0)

    def sample(self, batch_size=1000, random_state=np.random):
        r"""Sample points from the union of multiple regression-based bounds.

        Attributes
        ----------
        batch_size : int, optional
            How many points to draw at most.
        random_state : numpy.random or numpy.random.RandomState instance
            Determines random number generation.

        Returns
        -------
        numpy.ndarray with shape (batch_size, n)
            Array where each row represents a point. Sometimes, :math:`m` will
            be smaller than `batch_size`. In particular, :math:`m` /
            `batch_size` indicates the ratio of the true volume of the bound
            compared to what is returned by
            :func:`~nautilus.bounds.MultiNeuralBound.volume`.
        """

        p = np.exp(self.log_v - logsumexp(self.log_v))
        n_per_ell = random_state.multinomial(batch_size, p)

        points = []
        for i in range(len(self.bounds)):
            points_i = self.bounds[i].sample(
                batch_size=n_per_ell[i], random_state=random_state)
            n_in_other_bounds = np.zeros(len(points_i))
            for k in range(len(self.bounds)):
                if k != i:
                    n_in_other_bounds += self.bounds[k].contains(points_i)
            points_i = points_i[random_state.random(len(points_i)) <
                                1.0 / (1 + n_in_other_bounds)]
            points.append(points_i)

        points = np.vstack(points)

        # shuffle
        points = points[random_state.choice(
            len(points), size=len(points), replace=False)]

        return points

    def volume(self):
        """Raw volume estimate. This does not account for the internal
        regression or overlaps between different sub-bounds.

        Returns
        -------
        float
            The natural log of the volume.

        """
        return logsumexp(self.log_v)

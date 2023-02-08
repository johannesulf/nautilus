import numpy as np
import pytest

from scipy.special import gamma

from nautilus import bounds


@pytest.fixture
def neural_network_kwargs():
    return {
        'hidden_layer_sizes': (100, 50, 20), 'alpha': 0,
        'learning_rate_init': 1e-2, 'max_iter': 10000,
        'random_state': 0, 'tol': 1e-4, 'n_iter_no_change': 20}


@pytest.fixture
def points_on_hypersphere_boundary():
    # 2 * n_dim points on the boundary of a unit hypersphere centered on 0.5.

    n_dim = 10
    points = np.zeros((2 * n_dim, n_dim)) + 0.5
    for i in range(n_dim * 2):
        points[i, i // 2] += 1 if i % 2 else -1
    return points


@pytest.fixture
def random_points_from_hypersphere():
    # Random points from a unit hypersphere.

    np.random.seed(0)
    n_dim = 3
    n_points = 1000
    points = np.random.normal(size=(n_points, n_dim))
    points = points / np.sqrt(np.sum(points**2, axis=1))[:, np.newaxis]
    points *= np.random.uniform(size=n_points)[:, np.newaxis]**(1.0 / n_dim)
    return points


@pytest.fixture
def random_points_from_hypercube():
    # Random points from a unit hypersphere.

    np.random.seed(0)
    n_dim = 4
    n_points = 500
    return np.random.random(size=(n_points, n_dim))


def test_unit_cube():
    # Test that UnitCube works as expected.

    n_dim, n_points = 3, 200
    cube = bounds.UnitCube.compute(n_dim)

    point = cube.sample()
    assert point.shape == (n_dim, )
    assert np.all((point >= 0) & (point <= 1))
    assert cube.contains(point)

    points = cube.sample(n_points)
    assert points.shape == (n_points, n_dim)
    assert np.all((points >= 0) & (points < 1))
    assert np.all(cube.contains(points))

    assert cube.volume() == 0


def test_unit_cube_random_state():
    # Test that passing a random state leads to reproducible results.

    n_dim, n_points = 7, 1000
    cube = bounds.UnitCube.compute(
        n_dim, random_state=np.random.RandomState(0))
    cube_same = bounds.UnitCube.compute(
        n_dim, random_state=np.random.RandomState(0))
    cube_diff = bounds.UnitCube.compute(
        n_dim, random_state=np.random.RandomState(1))
    points = cube.sample(n_points)
    assert np.all(points == cube_same.sample(n_points))
    assert not np.all(points == cube_diff.sample(n_points))
    # We've already sampled from the cube. New samples should be different.
    assert not np.all(points == cube.sample(n_points))

    points = np.random.random((n_points, n_dim))
    assert np.all(cube.contains(points))


def test_invert_symmetric_positive_semidefinite_matrix():
    # Test that the fast implementation of matrix inversion works correctly.

    np.random.seed(0)
    points = np.random.normal(size=(1000, 10))
    m = np.cov(points, rowvar=False)
    assert np.allclose(
        bounds.invert_symmetric_positive_semidefinite_matrix(m),
        np.linalg.inv(m))


@pytest.mark.parametrize("tol", [0, 1])
def test_minimum_volume_enclosing_ellipsoid(points_on_hypersphere_boundary,
                                            tol):
    # Test that the MVEE algorithm returns a good approximation to the MVEE.

    c_true = np.median(points_on_hypersphere_boundary, axis=0)
    A_true = np.eye(len(c_true))

    np.random.seed(0)
    # Add a random point. Otherwise, the result is fully correct after the
    # first iteration.
    points = np.concatenate([points_on_hypersphere_boundary,
                             np.atleast_2d(c_true + np.random.random() - 0.5)])
    c, A = bounds.minimum_volume_enclosing_ellipsoid(
        points, tol=tol, max_iterations=1000)
    assert np.allclose(c, c_true, rtol=0, atol=1e-3) or tol > 0
    assert np.allclose(A, A_true, rtol=0, atol=1e-2) or tol > 0


def test_ellipsoid_construction():
    # Test that the ellipsoid construction fails under certain circumstances.

    with pytest.raises(ValueError):
        bounds.Ellipsoid.compute(np.random.random(size=(10, 10)))

    with pytest.raises(ValueError):
        bounds.Ellipsoid.compute(np.random.random(size=(100, 10)), enlarge=0.9)


def test_ellipsoid_sample_and_contains(points_on_hypersphere_boundary):
    # Test that the ellipsoidal sampling and boundary work as expected.

    ell = bounds.Ellipsoid.compute(
        points_on_hypersphere_boundary, enlarge=1.0,
        random_state=np.random.RandomState(0))
    c = np.mean(points_on_hypersphere_boundary)

    point = ell.sample()
    assert point.shape == (points_on_hypersphere_boundary.shape[1], )
    assert np.linalg.norm(point - c) < 1 + 1e-9
    assert ell.contains(point)

    n_points = 100
    points = ell.sample(n_points)
    assert points.shape == (n_points, points_on_hypersphere_boundary.shape[1])
    assert np.all(np.linalg.norm(points - c, axis=1) < 1 + 1e-9)
    assert np.all(ell.contains(points))

    ell = bounds.Ellipsoid.compute(
        points, enlarge=2.0, random_state=np.random.RandomState(0))
    points = ell.sample(n_points)
    assert not np.all(np.linalg.norm(points - c, axis=1) < 1)
    assert np.all(ell.contains(points))


def test_ellipsoid_volume(points_on_hypersphere_boundary):
    # Test that the volume of the ellipsoid is accurate.

    for enlarge in [1.0, 2.0, np.pi]:
        ell = bounds.Ellipsoid.compute(
            points_on_hypersphere_boundary, enlarge=enlarge)
        n_dim = points_on_hypersphere_boundary.shape[1]
        assert np.isclose(
            ell.volume(), np.log(enlarge * np.pi**(n_dim / 2) /
                                 gamma(n_dim / 2 + 1)))


def test_ellipsoid_transform(random_points_from_hypersphere):
    # Test that the Cholesky decomposition works correctly.

    ell = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(0))
    points = ell.sample(100)
    points_t = ell.transform(points)
    assert np.all(np.abs(points_t) < 1 + 1e-9)
    assert np.allclose(points, ell.transform(points_t, inverse=True))


def test_ellipsoid_random_state(random_points_from_hypersphere):
    # Test that passing a random state leads to reproducible results.

    ell = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(0))
    ell_same = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(0))
    ell_diff = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(1))
    n_points = 1000
    points = ell.sample(n_points)
    assert np.all(points == ell_same.sample(n_points))
    assert not np.all(points == ell_diff.sample(n_points))
    assert not np.all(points == ell.sample(n_points))


def test_multi_ellipsoid_construction():
    # Test that the multi-ellipsoid construction fails under certain
    # circumstances.

    with pytest.raises(ValueError):
        bounds.MultiEllipsoid.compute(np.random.random(size=(100, 10)),
                                      n_points_min=5)


def test_multi_ellipsoid_split(random_points_from_hypersphere):
    # Test that adding ellipsoids works correctly.

    points = np.concatenate([random_points_from_hypersphere,
                             random_points_from_hypersphere + 100])

    mell = bounds.MultiEllipsoid.compute(
        points, enlarge=1.0 + 1e-9, random_state=np.random.RandomState(0))

    # When not allowing overlaps, only 2 ellipsoids should be possible.
    while mell.split_ellipsoid(allow_overlap=False):
        pass
    assert len(mell.ells) == 2
    assert np.all(mell.contains(points))

    # It should be possible to add 3 more ellipoids when overlaps are allowed.
    for i in range(3):
        assert mell.split_ellipsoid(allow_overlap=True)
    assert len(mell.ells) == 5
    assert np.all(mell.contains(points))

    points = np.random.random((20, 10))
    mell = bounds.MultiEllipsoid.compute(points)
    # Check that no new ellipsoid can be added.
    assert not mell.split_ellipsoid()

    # Check that every split leads to ellipsoids with the minimum number of
    # points.
    np.random.seed(0)
    n_points_min = 10
    for i in range(10):
        points = np.random.random((2 * n_points_min, 2))
        mell = bounds.MultiEllipsoid.compute(points, n_points_min=n_points_min)
        mell.split_ellipsoid()
        assert len(mell.points) == 2
        assert len(mell.points[0]) == n_points_min
        assert len(mell.points[1]) == n_points_min


def test_multi_ellipsoid_sample_and_contains(random_points_from_hypersphere):
    # Test whether the multi-ellipsoidal sampling and boundary work as
    # expected.

    mell = bounds.MultiEllipsoid.compute(
        random_points_from_hypersphere + 50, enlarge=1.0,
        random_state=np.random.RandomState(0))
    for i in range(4):
        mell.split_ellipsoid()

    point = mell.sample()
    assert point.shape == (random_points_from_hypersphere.shape[1], )
    assert mell.contains(point)

    n_points = 100
    points = mell.sample(n_points)
    assert points.shape == (n_points, random_points_from_hypersphere.shape[1])
    assert np.all(mell.contains(points))

    mell_large = bounds.MultiEllipsoid.compute(
        points, enlarge=2.0, random_state=np.random.RandomState(0))
    points = mell_large.sample(n_points)
    assert not np.all(mell.contains(points))


def test_multi_ellipsoid_random_state(random_points_from_hypersphere):
    # Test that passing a random state leads to reproducible results.

    mell = bounds.MultiEllipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(0))
    mell.split_ellipsoid()
    mell_same = bounds.MultiEllipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(0))
    mell_same.split_ellipsoid()
    mell_diff = bounds.MultiEllipsoid.compute(
        random_points_from_hypersphere, random_state=np.random.RandomState(1))
    mell_diff.split_ellipsoid()
    n_points = 100
    points = mell.sample(n_points)
    assert np.all(points == mell_same.sample(n_points))
    assert not np.all(points == mell_diff.sample(n_points))
    assert not np.all(points == mell.sample(n_points))


def test_neural_bound_contains(random_points_from_hypercube,
                               neural_network_kwargs):
    # Test whether the neural sampling and boundary work as expected.

    points = random_points_from_hypercube
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)
    nbound = bounds.NeuralBound.compute(
        points, log_l, log_l_min, neural_network_kwargs=neural_network_kwargs)

    n_points = 100
    n_dim = random_points_from_hypercube.shape[1]
    points = np.random.random(size=(n_points, n_dim))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    in_bound = nbound.contains(points)
    assert np.mean(log_l[in_bound] > log_l_min) >= 0.9


def test_nautilus_bound_sample_and_contains(random_points_from_hypercube,
                                            neural_network_kwargs):
    # Test whether the nautilus sampling and boundary work as expected.

    points = random_points_from_hypercube
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)
    nbound = bounds.NautilusBound.compute(
        points, log_l, log_l_min, np.log(0.5),
        neural_network_kwargs=neural_network_kwargs)

    n_points = 100
    n_dim = random_points_from_hypercube.shape[1]
    points = np.random.random(size=(n_points, n_dim))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    in_bound = nbound.contains(points)
    assert np.mean(log_l[in_bound] > log_l_min) >= 0.9

    points = nbound.sample(n_points)
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    assert np.mean(log_l > log_l_min) >= 0.9
    assert np.all(nbound.contains(points))


def test_nautilus_bound_gaussian_shell(neural_network_kwargs):
    # Test nautilus sampling on the classic gaussian shell problem. In this
    # problem, the likelihood is high on a thin two-dimensional shell.

    radius = 0.45
    width = 0.01

    np.random.seed(0)
    points = np.random.random((10000, 2))
    log_l = -((np.linalg.norm(points - 0.5, axis=1) - radius) / width)**2
    log_l_min = -1
    points = points[log_l > -100]
    log_l = log_l[log_l > -100]
    log_v_target = np.log(2 * np.pi * radius * width * 2)

    nbound = bounds.NautilusBound.compute(
        points, log_l, log_l_min, log_v_target,
        split_threshold=1,
        neural_network_kwargs=neural_network_kwargs,
        random_state=np.random.RandomState(0))

    points = nbound.sample(10000)
    log_l = -((np.linalg.norm(points - 0.5, axis=1) - radius) / width)**2
    # The volume should be close to the true volume where log_l > log_l_min.
    assert np.isclose(nbound.volume(), log_v_target, rtol=0, atol=np.log(2))
    # Most sampled points should have log_l > log_l_min.
    assert np.mean(log_l > log_l_min) > 0.5
    # We should have only one neural network.
    assert nbound.number_of_networks_and_ellipsoids()[0] == 1


def test_nautilus_bound_small_target(random_points_from_hypercube,
                                     neural_network_kwargs):
    # Test that nothing catastrophic happens if we set the target volume to
    # effectively 0. This should just result in very aggresive ellipsoid
    # splitting and make the boundary miss a small part of the parameter space.

    points = random_points_from_hypercube
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.amin(log_l)
    nbound = bounds.NautilusBound.compute(
        points, log_l, log_l_min, -np.inf,
        neural_network_kwargs=neural_network_kwargs, n_points_min=20)
    assert nbound.volume() > -1
    assert nbound.number_of_networks_and_ellipsoids()[0] == 1
    assert nbound.number_of_networks_and_ellipsoids()[1] > 10


def test_nautilus_bound_two_peaks(neural_network_kwargs):
    # Test that the nautilus bound can identify and sample efficienctly from
    # two peaks with wide separations.

    np.random.seed(0)
    radius = 1e-5
    points = np.vstack([np.random.normal(size=(1000, 2)) * radius + 0.1,
                        np.random.normal(size=(1000, 2)) * radius + 0.9])

    def likelihood(x):
        return - np.minimum(
            np.linalg.norm(x - 0.1, axis=-1),
            np.linalg.norm(x - 0.9, axis=-1)) / radius

    log_l = likelihood(points)
    log_l_min = -1
    log_v_target = np.log(2 * np.pi * radius**2)
    nbound = bounds.NautilusBound.compute(points, log_l, log_l_min,
                                          log_v_target)

    points = nbound.sample(10000)
    log_l = likelihood(points)
    # The volume should be close to the true volume where log_l > log_l_min.
    assert np.isclose(nbound.volume(), log_v_target, rtol=0, atol=0.1)
    # Most sampled points should have log_l > log_l_min.
    assert np.mean(log_l > log_l_min) > 0.9
    # We should have two neural networks.
    assert nbound.number_of_networks_and_ellipsoids()[0] == 2

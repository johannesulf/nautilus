import numpy as np
import pytest

from scipy.special import gamma

from nautilus import bounds
from nautilus.bounds.basic import minimum_volume_enclosing_ellipsoid
from nautilus.bounds.basic import invert_symmetric_positive_semidefinite_matrix
from nautilus.pool import NautilusPool


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

    points = cube.sample(n_points)
    assert points.shape == (n_points, n_dim)
    assert np.all((points >= 0) & (points < 1))
    assert np.all(cube.contains(points))
    assert cube.log_v == 0


def test_unit_cube_rng():
    # Test that passing a random number generator leads to reproducible
    # results.

    n_dim, n_points = 7, 1000
    cube = bounds.UnitCube.compute(n_dim, rng=np.random.default_rng(0))
    cube_same = bounds.UnitCube.compute(n_dim, rng=np.random.default_rng(0))
    cube_diff = bounds.UnitCube.compute(n_dim, rng=np.random.default_rng(1))
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
        invert_symmetric_positive_semidefinite_matrix(m),
        np.linalg.inv(m))


def test_minimum_volume_enclosing_ellipsoid(points_on_hypersphere_boundary):
    # Test that the MVEE algorithm returns a good approximation to the MVEE.

    c_true = np.median(points_on_hypersphere_boundary, axis=0)
    A_true = np.eye(len(c_true))

    np.random.seed(0)
    # Add a random point. Otherwise, the result is fully correct after the
    # first iteration.
    points = np.concatenate([points_on_hypersphere_boundary,
                             np.atleast_2d(c_true + np.random.random() - 0.5)])
    c, A = minimum_volume_enclosing_ellipsoid(points)[:2]
    assert np.allclose(c, c_true, rtol=0, atol=1e-3)
    assert np.allclose(A, A_true, rtol=0, atol=1e-2)


def test_ellipsoid_construction():
    # Test that the ellipsoid construction fails under certain circumstances.

    with pytest.raises(ValueError):
        bounds.Ellipsoid.compute(np.random.random(size=(10, 10)))

    with pytest.raises(ValueError):
        bounds.Ellipsoid.compute(np.random.random(size=(100, 10)),
                                 enlarge_per_dim=0.9)


def test_ellipsoid_sample_and_contains(points_on_hypersphere_boundary):
    # Test that the ellipsoidal sampling and boundary work as expected.

    ell = bounds.Ellipsoid.compute(
        points_on_hypersphere_boundary, enlarge_per_dim=1.0,
        rng=np.random.default_rng(0))
    c = np.mean(points_on_hypersphere_boundary)

    n_points = 100
    points = ell.sample(n_points)
    assert points.shape == (n_points, points_on_hypersphere_boundary.shape[1])
    assert np.all(np.linalg.norm(points - c, axis=1) < 1 + 1e-9)
    assert np.all(ell.contains(points))

    ell = bounds.Ellipsoid.compute(
        points, enlarge_per_dim=1.1, rng=np.random.default_rng(0))
    points = ell.sample(n_points)
    assert not np.all(np.linalg.norm(points - c, axis=1) < 1)
    assert np.all(ell.contains(points))


def test_ellipsoid_volume(points_on_hypersphere_boundary):
    # Test that the volume of the ellipsoid is accurate.

    for enlarge_per_dim in [1.0, 1.1, np.pi / 2.0]:
        ell = bounds.Ellipsoid.compute(
            points_on_hypersphere_boundary, enlarge_per_dim=enlarge_per_dim)
        n_dim = points_on_hypersphere_boundary.shape[1]
        assert np.isclose(
            ell.log_v, np.log(enlarge_per_dim**n_dim * np.pi**(n_dim / 2) /
                              gamma(n_dim / 2 + 1)))


def test_ellipsoid_transform(random_points_from_hypersphere):
    # Test that the Cholesky decomposition works correctly.

    ell = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, rng=np.random.default_rng(0))
    points = ell.sample(100)
    points_t = ell.transform(points)
    assert np.all(np.abs(points_t) < 1 + 1e-9)
    assert np.allclose(points, ell.transform(points_t, inverse=True))


def test_ellipsoid_rng(random_points_from_hypersphere):
    # Test that passing a random number generator leads to reproducible
    # results.

    ell = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, rng=np.random.default_rng(0))
    ell_same = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, rng=np.random.default_rng(0))
    ell_diff = bounds.Ellipsoid.compute(
        random_points_from_hypersphere, rng=np.random.default_rng(1))
    n_points = 1000
    points = ell.sample(n_points)
    assert np.all(points == ell_same.sample(n_points))
    assert not np.all(points == ell_diff.sample(n_points))
    assert not np.all(points == ell.sample(n_points))


def test_union_construction():
    # Test that the union construction fails under certain circumstances.

    with pytest.raises(ValueError):
        bounds.Union.compute(np.random.random(size=(100, 10)),
                             n_points_min=5)


def test_union_split(random_points_from_hypersphere):
    # Test that adding ellipsoids works correctly.

    points = np.concatenate([random_points_from_hypersphere,
                             random_points_from_hypersphere + 100,
                             random_points_from_hypersphere + 101])

    union = bounds.Union.compute(
        points, enlarge_per_dim=1.0 + 1e-9, unit=False,
        rng=np.random.default_rng(0))

    # When not allowing overlaps, only 2 ellipsoids should be possible.
    while union.split(allow_overlap=False):
        pass
    assert len(union.bounds) == 2
    assert np.all(union.contains(points))

    # It should be possible to add one more ellipoids when overlaps are
    # allowed.
    assert union.split()
    assert not union.split()
    assert len(union.bounds) == 3
    assert np.all(union.contains(points))


def test_union_split_stops(random_points_from_hypersphere):
    # Test that splitting ellipsoids correctly stops after some time.

    x = np.linspace(-1, 1, 30)
    y = x**2
    points = np.vstack([x, y]).T

    bound = bounds.Union.compute(points)
    n = 0

    while bound.split():
        n += 1

    assert n > 0
    assert np.all(np.array([len(points) for points in bound.points_bounds]) >=
                  bound.n_points_min)


def test_union_split_and_trim(random_points_from_hypersphere):
    # Test that trimming works as expected.

    points = np.vstack([random_points_from_hypersphere,
                        random_points_from_hypersphere + 10,
                        random_points_from_hypersphere[:30] + 1e7])

    bound = bounds.Union.compute(points, unit=False, n_points_min=50,
                                 rng=np.random.default_rng(0))

    # The volume should be large.
    assert bound.log_v > 15
    # We should be able to split the bound at least twice.
    assert bound.split()
    assert bound.split()
    # In this case, splitting should not help substantially in reducing the
    # volume since one ellipsoid covers points from the sphere at 1e7 and a few
    # points outside it at ~0.
    assert bound.log_v > 15
    # Trimming involves removing the lowest-density ellipsoid and its points.
    assert bound.trim()
    # Now the volume should be reasonable.
    assert bound.log_v < 5
    # Removing more ellipsoid should fail since they all have similar
    # densities.
    assert not bound.trim()


def test_union_sample_and_contains(random_points_from_hypersphere):
    # Test whether the union sampling and boundary work as expected.

    union = bounds.Union.compute(
        random_points_from_hypersphere + 50, enlarge_per_dim=1.0,
        unit=False, rng=np.random.default_rng(0))
    for i in range(4):
        union.split()

    n_points = 100
    points = union.sample(n_points)
    assert points.shape == (n_points, random_points_from_hypersphere.shape[1])
    assert np.all(union.contains(points))

    union_large = bounds.Union.compute(
        points, enlarge_per_dim=1.1, unit=False, rng=np.random.default_rng(0))
    points = union_large.sample(n_points)
    assert not np.all(union.contains(points))


def test_union_rng(random_points_from_hypersphere):
    # Test that passing a random number generator leads to reproducible
    # results.

    union = bounds.Union.compute(
        random_points_from_hypersphere, unit=False,
        rng=np.random.default_rng(0))
    union.split()
    union_same = bounds.Union.compute(
        random_points_from_hypersphere, unit=False,
        rng=np.random.default_rng(0))
    union_same.split()
    union_diff = bounds.Union.compute(
        random_points_from_hypersphere, unit=False,
        rng=np.random.default_rng(1))
    union_diff.split()
    n_points = 100
    points = union.sample(n_points)
    assert np.all(points == union_same.sample(n_points))
    assert not np.all(points == union_diff.sample(n_points))
    assert not np.all(points == union.sample(n_points))


def test_neural_bound_contains(random_points_from_hypercube):
    # Test whether the neural sampling and boundary work as expected.

    points = random_points_from_hypercube
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)
    nbound = bounds.NeuralBound.compute(points, log_l, log_l_min, n_networks=1)

    n_points = 100
    n_dim = random_points_from_hypercube.shape[1]
    points = np.random.random(size=(n_points, n_dim))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    in_bound = nbound.contains(points)
    assert np.mean(log_l[in_bound] > log_l_min) >= 0.9


def test_phase_shift():
    # Test whether the phase-shift correctly centers periodic dimensions.

    np.random.seed(0)
    for i in range(100):
        n_points = 10  # 100
        n_dim = 2  # 10
        points = (np.random.random(size=(n_points, n_dim)) * 0.1 +
                  np.random.random(size=n_dim)) % 1
        shift = bounds.PhaseShift.compute(points, np.arange(n_dim // 2))
        assert np.amin(shift.transform(points)[:, :n_dim // 2]) >= 0.45
        assert np.amax(shift.transform(points)[:, :n_dim // 2]) <= 0.55
        assert np.allclose(points, shift.transform(shift.transform(
            points), inverse=True), rtol=0, atol=1e-12)


def test_nautilus_bound_sample_and_contains(random_points_from_hypercube):
    # Test whether the nautilus sampling and boundary work as expected.

    points = random_points_from_hypercube
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)
    nbound = bounds.NautilusBound.compute(
        points, log_l, log_l_min, np.log(0.5), n_networks=1)

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


def test_nautilus_bound_gaussian_shell():
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
        points, log_l, log_l_min, log_v_target, split_threshold=1,
        n_networks=1, rng=np.random.default_rng(0))

    points = nbound.sample(10000)
    log_l = -((np.linalg.norm(points - 0.5, axis=1) - radius) / width)**2
    # The volume should be close to the true volume where log_l > log_l_min.
    assert np.isclose(nbound.log_v, log_v_target, rtol=0, atol=np.log(2))
    # Most sampled points should have log_l > log_l_min.
    assert np.mean(log_l > log_l_min) > 0.5
    # We should have only one neural network.
    assert nbound.n_net == 1


def test_nautilus_bound_two_peaks():
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
    nbound = bounds.NautilusBound.compute(
        points, log_l, log_l_min, log_v_target, n_networks=1,
        rng=np.random.default_rng(0))

    points = nbound.sample(10000)
    log_l = likelihood(points)
    # The volume should be close to the true volume where log_l > log_l_min.
    assert np.isclose(nbound.log_v, log_v_target, rtol=0, atol=0.1)
    # Most sampled points should have log_l > log_l_min.
    assert np.mean(log_l > log_l_min) > 0.9
    # We should have two neural networks.
    assert nbound.n_net == 2


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_nautilus_bound_reset_and_sample(random_points_from_hypercube, n_jobs):
    # Test that resetting the bound works as expected and that we can sample
    # in parallel.

    points = random_points_from_hypercube
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)

    if n_jobs > 1:
        pool = NautilusPool(n_jobs)
    else:
        pool = None

    nbound = bounds.NautilusBound.compute(
        points, log_l, log_l_min, np.log(0.5), n_networks=1, pool=pool,
        rng=np.random.default_rng(0))

    nbound.reset(np.random.default_rng(0))
    points_1 = nbound.sample(10000, pool=pool)
    volume_1 = nbound.log_v
    nbound.reset(np.random.default_rng(0))
    points_2 = nbound.sample(10000, pool=pool)
    volume_2 = nbound.log_v

    if n_jobs > 1:
        pool.pool.close()

    assert np.all(points_1 == points_2)
    assert volume_1 == volume_2

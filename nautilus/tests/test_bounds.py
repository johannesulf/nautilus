import numpy as np
from nautilus import bounds
from scipy.special import gamma


def get_test_points(n_dim, c):
    points = np.zeros((2 * n_dim, n_dim)) + c
    for i in range(n_dim * 2):
        points[i, i // 2] += 1 if i % 2 else -1
    return points


def test_unit_cube_sample_and_contains():

    n_dim, n_points = 7, 1000
    cube = bounds.UnitCube(n_dim)

    point = cube.sample()
    assert point.shape == (n_dim, )
    assert np.all((point >= 0) & (point <= 1))
    assert cube.contains(point)

    points = cube.sample(n_points)
    assert points.shape == (n_points, n_dim)
    assert np.all((points >= 0) & (points < 1))
    assert np.all(cube.contains(points))


def test_unit_cube_volume():

    n_dim = 7
    cube = bounds.UnitCube(n_dim)
    assert cube.volume() == 0


def test_unit_cube_random_state():
    n_dim, n_points = 7, 1000
    cube = bounds.UnitCube(n_dim, random_state=np.random.RandomState(0))
    cube_same = bounds.UnitCube(n_dim, random_state=np.random.RandomState(0))
    cube_diff = bounds.UnitCube(n_dim, random_state=np.random.RandomState(1))
    points = cube.sample(n_points)
    assert np.all(points == cube_same.sample(n_points))
    assert not np.all(points == cube_diff.sample(n_points))
    # We've already sampled from the cube. New samples should be different.
    assert not np.all(points == cube.sample(n_points))

    points = np.random.random((n_points, n_dim))
    assert np.all(cube.contains(points))


def test_invert_symmetric_positive_semidefinite_matrix():

    np.random.seed(0)
    points = np.random.normal(size=(1000, 10))
    m = np.cov(points, rowvar=False)
    assert np.allclose(
        bounds.invert_symmetric_positive_semidefinite_matrix(m),
        np.linalg.inv(m))


def test_minimum_volume_enclosing_ellipsoid():

    np.random.seed(0)
    n_dim = 7
    c_in = np.random.random(n_dim)
    points = get_test_points(n_dim, c_in)
    c, A = bounds.minimum_volume_enclosing_ellipsoid(points)
    assert np.allclose(c, c_in)
    assert np.allclose(A, np.eye(n_dim))


def test_ellipsoid_sample_and_contains():

    np.random.seed(0)
    n_dim = 7
    c = np.random.random(n_dim)
    points = get_test_points(n_dim, c)
    ell = bounds.Ellipsoid(points, enlarge=1.0,
                           random_state=np.random.RandomState(0))

    point = ell.sample()
    assert point.shape == (n_dim, )
    assert np.linalg.norm(point - c) < 1 + 1e-9
    assert ell.contains(point)

    n_points = 1000
    points = ell.sample(n_points)
    assert points.shape == (n_points, n_dim)
    assert np.all(np.linalg.norm(points - c, axis=1) < 1 + 1e-9)
    assert np.all(ell.contains(points))

    ell = bounds.Ellipsoid(points, enlarge=2.0,
                           random_state=np.random.RandomState(0))
    points = ell.sample(n_points)
    assert not np.all(np.linalg.norm(points - c, axis=1) < 1)
    assert np.all(ell.contains(points))


def test_ellipsoid_volume():

    n_dim = 7
    points = get_test_points(n_dim, np.random.random(n_dim))
    for enlarge in [1.0, 2.0, np.pi]:
        ell = bounds.Ellipsoid(points, enlarge=enlarge)
        assert np.isclose(
            ell.volume(), np.log(enlarge * np.pi**(n_dim / 2) /
                                 gamma(n_dim / 2 + 1)))


def test_ellipsoid_transform():

    np.random.seed(0)
    n_dim, n_points = 7, 1000
    points = np.random.random(size=(n_points, n_dim))
    ell = bounds.Ellipsoid(points, random_state=np.random.RandomState(0))
    points = ell.sample(n_points)
    points_t = ell.transform(points)
    assert np.all(np.abs(points_t) < 1 + 1e-9)
    assert np.allclose(points, ell.transform(points_t, inverse=True))


def test_ellipsoid_random_state():

    np.random.seed(0)
    n_dim, n_points = 7, 1000
    points = np.random.random(size=(n_points, n_dim))

    ell = bounds.Ellipsoid(points, random_state=np.random.RandomState(0))
    ell_same = bounds.Ellipsoid(
        points, random_state=np.random.RandomState(0))
    ell_diff = bounds.Ellipsoid(
        points, random_state=np.random.RandomState(1))
    points = ell.sample(n_points)
    assert np.all(points == ell_same.sample(n_points))
    assert not np.all(points == ell_diff.sample(n_points))
    assert not np.all(points == ell.sample(n_points))


def test_multi_ellipsoid_split():

    n_dim, n_points = 10, 1000
    np.random.seed(0)
    points = np.random.random(size=(n_points, n_dim))
    points[n_points // 2:, 0] += 100

    mell = bounds.MultiEllipsoid(
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

    points = mell.sample(n_points)
    assert np.all(((points[:, 0] >= -3) & (points[:, 0] < 4)) |
                  ((points[:, 0] >= -3 + 100) & (points[:, 0] < 4 + 100)))
    assert np.all((points[:, 1:] >= -3) & (points[:, 1:] < 4))


def test_multi_ellipsoid_sample_and_contains():

    np.random.seed(0)
    n_dim, n_points = 7, 1000
    points = np.random.normal(size=(n_points, n_dim))
    mell = bounds.MultiEllipsoid(points, enlarge=1.0,
                                 random_state=np.random.RandomState(0))
    for i in range(4):
        mell.split_ellipsoid()

    point = mell.sample()
    assert point.shape == (n_dim, )
    assert mell.contains(point)

    n_points = 1000
    points = mell.sample(n_points)
    assert points.shape == (n_points, n_dim)
    assert np.all(mell.contains(points))

    mell_large = bounds.MultiEllipsoid(points, enlarge=2.0,
                                       random_state=np.random.RandomState(0))
    points = mell_large.sample(n_points)
    assert not np.all(mell.contains(points))


def test_multi_ellipsoid_random_state():

    np.random.seed(0)
    n_dim, n_points = 7, 1000
    points = np.random.normal(size=(n_points, n_dim))

    mell = bounds.MultiEllipsoid(points, random_state=np.random.RandomState(0))
    mell.split_ellipsoid()
    mell_same = bounds.MultiEllipsoid(
        points, random_state=np.random.RandomState(0))
    mell_same.split_ellipsoid()
    mell_diff = bounds.MultiEllipsoid(
        points, random_state=np.random.RandomState(1))
    mell_diff.split_ellipsoid()
    points = mell.sample(n_points)
    assert np.all(points == mell_same.sample(n_points))
    assert not np.all(points == mell_diff.sample(n_points))
    assert not np.all(points == mell.sample(n_points))


def test_neural_bound_contains():

    np.random.seed(0)
    n_dim, n_points = 2, 1000
    points = np.random.random(size=(n_points, n_dim))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)
    nell = bounds.NeuralBound(points, log_l, log_l_min)

    points = np.random.random(size=(n_points, n_dim))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    in_bound = nell.contains(points)

    assert np.mean(log_l[in_bound] > log_l_min) >= 0.9


def test_nautilus_bound_sample_and_contains():

    np.random.seed(0)
    n_dim, n_points = 2, 1000
    points = np.random.random(size=(n_points, n_dim))
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    log_l_min = np.median(log_l)
    nell = bounds.NautilusBound(points, log_l, log_l_min, 0)

    points = nell.sample(n_points)
    log_l = -np.linalg.norm(points - 0.5, axis=1)
    assert np.mean(log_l > log_l_min) >= 0.9
    assert np.all(nell.contains(points))

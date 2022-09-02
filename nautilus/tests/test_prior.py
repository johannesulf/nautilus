import pytest
import numpy as np
from scipy.stats import norm
from nautilus.prior import Prior


def test_add_parameter():
    # Test that adding parameters works as expected.

    prior = Prior()
    with pytest.raises(TypeError):
        prior.add_parameter(1.0)

    prior = Prior()
    prior.add_parameter('a')
    prior.add_parameter('b')
    with pytest.raises(ValueError):
        prior.add_parameter('a')

    prior = Prior()
    with pytest.raises(ValueError):
        prior.add_parameter('a', dist='a')

    prior = Prior()
    with pytest.raises(TypeError):
        prior.add_parameter(dist=[0.0])


def test_dimensionality():
    # Test that the dimensionality is correctly reported.

    prior = Prior()
    n_dim = 6
    for i in range(n_dim):
        prior.add_parameter()
    assert prior.dimensionality() == n_dim

    prior = Prior()
    prior.add_parameter()
    prior.add_parameter(dist=5.0)
    prior.add_parameter()
    prior.add_parameter(dist=1)
    assert prior.dimensionality() == 2

    prior = Prior()
    prior.add_parameter(key='a')
    prior.add_parameter(key='b', dist=1)
    prior.add_parameter(key='c', dist='a')
    prior.add_parameter(key='d', dist='b')
    prior.add_parameter(key='e')
    assert prior.dimensionality() == 2


def test_unit_to_physical():
    # Test converting unit hypercube coordinates to physical ones.

    prior = Prior()
    prior.add_parameter(key='a', dist=(-1, +1))
    prior.add_parameter(key='b', dist='a')
    prior.add_parameter(key='c', dist='b')
    dist1 = norm(loc=3.0, scale=2.0)
    dist2 = norm(loc=5.0, scale=1.0)
    prior.add_parameter(key='d', dist=dist1)
    prior.add_parameter(key='e', dist=dist2)

    unit = np.random.random(size=4)
    with pytest.raises(ValueError):
        phys = prior.unit_to_physical(unit)

    unit = np.random.random(size=3)
    phys = prior.unit_to_physical(unit)
    assert unit.shape == phys.shape
    assert np.isclose(phys[0], unit[0] * 2 - 1)
    assert np.isclose(phys[1], dist1.isf(1 - unit[1]))
    assert np.isclose(phys[2], dist2.isf(1 - unit[2]))

    unit = np.random.random(size=(10, 3))
    phys = prior.unit_to_physical(unit)
    assert unit.shape == phys.shape
    assert np.allclose(phys[:, 0], unit[:, 0] * 2 - 1)
    assert np.allclose(phys[:, 1], dist1.isf(1 - unit[:, 1]))
    assert np.allclose(phys[:, 2], dist2.isf(1 - unit[:, 2]))


def test_physical_to_structure():
    # Test converting physical coordinates to dictionaries/structured arrays.

    prior = Prior()
    prior.add_parameter(key='a')
    prior.add_parameter(key='b', dist='a')
    prior.add_parameter(key='c', dist='b')
    prior.add_parameter(key='d')
    prior.add_parameter(key='e')

    phys = np.random.random(size=4)
    with pytest.raises(ValueError):
        struct = prior.physical_to_structure(phys)

    phys = np.random.random(size=3)
    struct = prior.physical_to_structure(phys)
    assert isinstance(struct, dict)
    assert len(struct) == 5
    assert struct['a'] == phys[0]
    assert struct['d'] == phys[1]
    assert struct['e'] == phys[2]
    assert struct['b'] == struct['a']
    assert struct['c'] == struct['b']

    phys = np.random.random(size=(10, 3))
    struct = prior.physical_to_structure(phys)
    assert isinstance(struct, np.ndarray)
    assert struct.shape[0] == phys.shape[0]
    assert len(struct.dtype.names) == 5
    assert np.all(struct['a'] == phys[:, 0])
    assert np.all(struct['d'] == phys[:, 1])
    assert np.all(struct['e'] == phys[:, 2])
    assert np.all(struct['b'] == struct['a'])
    assert np.all(struct['c'] == struct['b'])

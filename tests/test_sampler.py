import numpy as np
import pytest
import warnings

from functools import partial
from scipy.stats import multivariate_normal, norm

from nautilus import Prior, Sampler


def likelihood_basic(x, pass_dict, n_blobs):
    # Likelihood used for the test below. It's defined at the module scope
    # so it's pickleable with 'spawn' and 'forkserver' multiprocessing methods.
    if pass_dict:
        x = np.squeeze(np.column_stack([x['a'], x['b']]))
    log_l = -np.linalg.norm(x - 0.5, axis=-1) * 0.001
    if n_blobs == 0:
        return log_l
    elif n_blobs == 1:
        return log_l, x[..., 0]
    else:
        return log_l, x[..., 0], x[..., 1]


@pytest.mark.parametrize("n_networks", [0, 1, 2])
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("pass_dict", [True, False])
@pytest.mark.parametrize("pool", [None, 2])
@pytest.mark.parametrize("n_blobs", [0, 1, 2])
def test_sampler_basic(n_networks, vectorized, pass_dict, pool, n_blobs):
    # Test that the sampler does not crash.

    if pass_dict:
        prior = Prior()
        prior.add_parameter('a')
        prior.add_parameter('b')
    else:
        def prior(x):
            return x

    likelihood = partial(likelihood_basic, pass_dict=pass_dict,
                         n_blobs=n_blobs)

    sampler = Sampler(
        prior, likelihood, n_dim=2, n_networks=n_networks,
        vectorized=vectorized, pass_dict=pass_dict, n_live=200, pool=pool)
    sampler.run(n_like_max=600, verbose=True)
    sampler.posterior()
    sampler.posterior(equal_weight=True)
    points, log_w, log_l = sampler.posterior(return_as_dict=pass_dict)
    if pass_dict:
        assert isinstance(points, dict)
        points = np.column_stack([points[key] for key in points])
    # All points should be unique (unless equal_weight_boost > 1).
    assert len(np.unique(points, axis=0)) == len(points)
    assert sampler.n_eff > 0
    sampler.log_z
    assert sampler.eta > 0
    assert sampler.eta < 1
    if n_blobs == 1:
        blobs = sampler.posterior(return_blobs=True)[-1]
        assert np.allclose(points[:, 0], blobs)
    elif n_blobs == 2:
        blobs = sampler.posterior(return_blobs=True)[-1]
        assert np.allclose(points[:, 0], blobs['blob_0'])
        assert np.allclose(points[:, 1], blobs['blob_1'])


def test_sampler_errors_and_warnings():
    # Test that the sampler correctly raises errors and warnings.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5, axis=-1) * 0.001

    with pytest.raises(ValueError):
        Sampler(prior, likelihood)

    with pytest.raises(ValueError):
        Sampler(prior, likelihood, 1)

    sampler = Sampler(prior, likelihood, 2, n_live=300)
    sampler.run(n_like_max=1000, verbose=True)

    with pytest.warns(DeprecationWarning):
        sampler.evidence()
    with pytest.warns(DeprecationWarning):
        sampler.effective_sample_size()
    with pytest.warns(DeprecationWarning):
        sampler.asymptotic_sampling_efficiency()
    with pytest.raises(ValueError):
        sampler.posterior(return_blobs=True)


@pytest.mark.parametrize("discard_exploration_start", [True, False])
@pytest.mark.parametrize("discard_exploration_end", [True, False, 1])
def test_sampler_switch_exploration(
        discard_exploration_start, discard_exploration_end):
    # Test that we can switch later whether to discard the exploration phase.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5, axis=-1) * 0.001

    sampler = Sampler(
        prior, likelihood, n_dim=2, n_networks=1, vectorized=True, n_live=500)
    sampler.run(f_live=0.45, n_eff=10000, verbose=True,
                discard_exploration=discard_exploration_start)
    assert sampler.discard_exploration == discard_exploration_start
    points, log_w, log_l = sampler.posterior()
    n_start = len(points)
    log_z_start = sampler.log_z

    if not isinstance(discard_exploration_end, bool):
        with pytest.raises(ValueError):
            sampler.discard_exploration = discard_exploration_end
        return

    sampler.discard_exploration = discard_exploration_end
    points, log_w, log_l = sampler.posterior()
    n_end = len(points)
    log_z_end = sampler.log_z

    assert ((discard_exploration_start == discard_exploration_end) ==
            (n_start == n_end))
    assert ((discard_exploration_start == discard_exploration_end) ==
            (log_z_start == log_z_end))


@pytest.mark.parametrize("custom_prior", [True, False])
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("pass_dict", [True, False])
def test_sampler_prior(custom_prior, vectorized, pass_dict):
    # Test that the sampler can handle all prior defintions.

    if custom_prior:
        if pass_dict:
            def prior(x):
                return dict(a=x[..., 0], b=x[..., 1])
        else:
            def prior(x):
                return x
    else:
        prior = Prior()
        prior.add_parameter('a')
        prior.add_parameter('b')

    def likelihood(x):
        if isinstance(x, dict):
            x = np.squeeze(np.column_stack([x['a'], x['b']]))
        return -np.linalg.norm(x - 0.5, axis=-1) * 0.001

    sampler = Sampler(
        prior, likelihood, n_dim=2, n_networks=1, vectorized=vectorized,
        pass_dict=pass_dict, n_live=500)
    sampler.run(f_live=0.45, n_eff=0, verbose=True)
    points, log_w, log_l = sampler.posterior(return_as_dict=pass_dict)
    if custom_prior and pass_dict:
        with pytest.raises(ValueError):
            points, log_w, log_l = sampler.posterior(return_as_dict=False)


@pytest.mark.parametrize("n_networks", [0, 4])
@pytest.mark.parametrize("discard_exploration", [True, False])
def test_sampler_accuracy(n_networks, discard_exploration):
    # Do a basic tests of the sampler accuracy.

    n_dim = 2
    mean = np.repeat(0.5, n_dim)
    cov = np.eye(n_dim) * 0.01

    def prior(x):
        return x

    def likelihood(x):
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)

    sampler = Sampler(prior, likelihood, n_dim=n_dim, n_live=500,
                      n_networks=n_networks, seed=0)
    sampler.run(discard_exploration=discard_exploration, f_live=0.1,
                verbose=True)

    assert np.abs(sampler.log_z) < 0.05

    for equal_weight in [True, False]:
        for equal_weight_boost in [1, 3, 10]:
            points, log_w, log_l = sampler.posterior(
                equal_weight=equal_weight,
                equal_weight_boost=equal_weight_boost)
            weights = np.exp(log_w)

            mean_sampler = np.average(points, weights=weights, axis=0)
            cov_sampler = np.cov(points, aweights=weights, rowvar=False)
            assert np.all(np.isclose(mean_sampler, mean, atol=0.01, rtol=0))
            assert np.all(np.isclose(cov_sampler, cov, atol=0.001, rtol=0))

    equal_weight_boost = 3
    n_eq = len(sampler.posterior(
        equal_weight=True, equal_weight_boost=1)[0])
    n_eq_err = np.sqrt(n_eq)  # This is actually on overestimate.
    n_eqw = len(sampler.posterior(
        equal_weight=True, equal_weight_boost=equal_weight_boost)[0])
    n_eqw_err = np.sqrt(n_eqw)  # This is actually on overestimate.
    assert np.abs(n_eq * equal_weight_boost - n_eqw) < 5 * np.sqrt(
        equal_weight_boost**2 * n_eq_err**2 + n_eqw_err**2)

    # This is a simple problem so the bounds should be perfectly nested, i.e.
    # bound i+1 is fully contained within bound i.
    shell_bound_occupation = sampler.shell_bound_occupation()
    assert np.all(shell_bound_occupation ==
                  np.tril(np.ones_like(shell_bound_occupation)))


def test_sampler_enlarge_per_dim():
    # Test that the enlarge_per_dim keyword is passed correctly. In this
    # example, we choose an enlargment factor so big that every bound should be
    # the same if we don't use neural networks. Thus, after the exploration
    # phase, all but one bound should be left.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5)**2 * 0.001

    sampler = Sampler(prior, likelihood, n_dim=2, enlarge_per_dim=100,
                      n_networks=0, seed=0)
    sampler.run(f_live=0.1, n_eff=0, verbose=True)

    # The effective sample size should be very close to the number of calls
    # since the likelihood is extremely flat.
    assert np.isclose(sampler.n_like, sampler.n_eff, rtol=0, atol=1)
    # Only one bound should be left.
    assert len(sampler.bounds) == 1
    # Check evidence accuracy.
    assert np.isclose(sampler.log_z, -4 * 0.5**3 / 3 * 0.001, rtol=0,
                      atol=1e-4)


def test_sampler_empty_shells():
    # Test that the sampler can correctly deal with shells that are empty at
    # the end of the run. This is a somewhat contrived example that should
    # result in empty shells since each one should only have one sample, on
    # average.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5)**2 * 0.001

    sampler = Sampler(prior, likelihood, n_dim=2,
                      n_networks=0, seed=0, n_update=1, n_live=10, n_batch=1)
    sampler.run(f_live=1e-3, n_eff=0, verbose=True)


def test_sampler_n_like_max():
    # Test that the sampler correctly stops when hitting the maximum number of
    # likelihood calls and can resume afterwards.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5)**2 * 0.001

    sampler_a = Sampler(prior, likelihood, n_dim=2, n_networks=0, seed=0)
    sampler_b = Sampler(prior, likelihood, n_dim=2, n_networks=0, seed=0)

    sampler_a.run(verbose=True)
    for n_like_max in range(sampler_a.n_like + 1):
        success = sampler_b.run(n_like_max=n_like_max, verbose=True)
        assert sampler_b.n_like <= n_like_max + sampler_b.n_batch
        assert not success if sampler_a.n_like != sampler_b.n_like else success

    assert sampler_a.log_z == sampler_b.log_z
    assert sampler_a.n_eff == sampler_b.n_eff


def test_sampler_timeout():
    # Test that the sampler correctly stops when reaching the timeout limits.

    def prior(x):
        return x

    def likelihood(x):
        return -np.linalg.norm(x - 0.5)**2 * 0.001

    sampler = Sampler(prior, likelihood, n_dim=10, n_networks=0, seed=0)
    # The sampler shouldn't finish within 1 second.
    success = sampler.run(verbose=True, timeout=1)
    assert not success

    # We should be able to continue afterwards.
    success = sampler.run(verbose=True, timeout=5)


def test_sampler_funnel():
    # Test the sampler on a funnel distribution. This is a great, challenging
    # distribution. Also, the nature of the likelihood leads to nautilus
    # boundaries that are not strictly nested, unlike for simpler
    # distributions.

    def prior(x):
        return x

    def likelihood(x):
        return (norm.logpdf(x[0], loc=0.5, scale=0.1) +
                norm.logpdf(x[1], loc=0.5, scale=np.exp(20 * (x[0] - 0.5)) /
                            100))

    # Determine the true evidence. The likelihood is normalized so it should be
    # close to 1 minus a correction for the fraction of points falling outside
    # the unit cube.
    np.random.seed(0)
    x_0 = np.random.normal(loc=0.5, scale=0.1, size=1000000)
    x_1 = np.random.normal(loc=0.5, scale=np.exp(20 * (x_0 - 0.5)) / 100)
    log_z_true = np.log(np.mean((x_0 > 0) & (x_0 < 1) & (x_1 > 0) & (x_1 < 1)))

    sampler = Sampler(prior, likelihood, n_dim=2, n_networks=1, seed=0)
    sampler.run(verbose=True)
    assert np.isclose(log_z_true, sampler.log_z, rtol=0, atol=0.1)
    # Check whether the boundaries nautilus drew are strictly nested.
    shell_bound_occupation = sampler.shell_bound_occupation()
    if np.all(shell_bound_occupation ==
              np.tril(np.ones_like(shell_bound_occupation))):
        warnings.warn('The funnel distribution was too easy.', RuntimeWarning)


def test_sampler_constant():
    # Test that the sampler handles a single constant likelihood gracefully.

    def prior(x):
        return x

    def likelihood(x):
        return 0

    sampler = Sampler(prior, likelihood, 2, n_live=500, seed=0)
    sampler.run(verbose=True, f_live=0.1, n_eff=0)

    assert np.isclose(sampler.log_z, 0)
    # The sampler should not have build any boundaries.
    assert len(sampler.bounds) == 1


def test_sampler_plateau_1():
    # Test that the sampler can deal with a pleateau.

    def prior(x):
        return x

    def likelihood(x):
        if x[0] < 0.9:
            return -np.inf
        else:
            return np.log(x[0] - 0.9)

    log_z_true = np.log(0.5 * 0.1**2)

    for i in range(10):
        sampler = Sampler(prior, likelihood, 2, n_live=1000, n_networks=1,
                          seed=i)
        sampler.run(verbose=True, f_live=0.1)
        assert np.isclose(sampler.log_z, log_z_true, rtol=0, atol=0.1)


def test_sampler_plateau_2():
    # Test that the sampler can deal with a pleateau.

    def prior(x):
        return x

    def likelihood(x):
        return np.ceil(-np.log10(1 - x[0]))

    log_z_true = np.log(
        np.sum(0.9 * 0.1**np.arange(100) * np.exp(1 + np.arange(100))))

    for i in range(10):
        sampler = Sampler(prior, likelihood, 2,
                          n_live=2000, n_networks=1, seed=0)
        sampler.run(verbose=True, f_live=1e-6)
        assert np.isclose(sampler.log_z, log_z_true, atol=0.1)
        # The minimum likelihood values should correspond to the different
        # plateaus.
        assert np.all(np.isclose(sampler.shell_log_l_min[1:],
                      np.arange(len(sampler.bounds) - 1) + 2, rtol=0))


@pytest.mark.parametrize("periodic", [False, True])
def test_sampler_periodic(periodic):
    # Test that the periodic boundary conditions work correctly. In particular,
    # the sampler shouldn't split modes extending over a boundary.

    n_dim = 2

    def prior(x):
        return x

    def likelihood(x):
        return multivariate_normal.logpdf(
            np.abs(x - 0.5), mean=[0.5] * n_dim, cov=0.1)

    sampler = Sampler(prior, likelihood, n_dim,
                      periodic=np.arange(n_dim) if periodic else None,
                      n_networks=0, seed=0)
    sampler.run(verbose=True)
    points, log_w, log_l = sampler.posterior()

    for bound in sampler.bounds[1:]:
        assert len(bound.neural_bounds) == 1 if periodic else 4

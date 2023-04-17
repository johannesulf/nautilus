import numpy as np
import pytest
import warnings

from scipy.stats import multivariate_normal, norm

from nautilus import Prior, Sampler


@pytest.mark.parametrize("n_networks", [0, 1, 2])
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("pass_dict", [True, False])
def test_sampler_basic(n_networks, vectorized, pass_dict):
    # Test that the sampler does not crash.

    if pass_dict:
        prior = Prior()
        prior.add_parameter('a')
        prior.add_parameter('b')
    else:
        def prior(x):
            return x

    def likelihood(x):
        if pass_dict:
            x = np.squeeze(np.column_stack([x['a'], x['b']]))
        return -np.linalg.norm(x - 0.5, axis=-1) * 0.001

    sampler = Sampler(
        prior, likelihood, n_dim=2, n_networks=n_networks,
        vectorized=vectorized, pass_dict=pass_dict, n_live=500, n_jobs=1)
    sampler.run(f_live=0.45, n_eff=0, verbose=False)
    points, log_w, log_l = sampler.posterior(return_as_dict=pass_dict)


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
        pass_dict=pass_dict, n_live=500, n_jobs=1)
    sampler.run(f_live=0.45, n_eff=0, verbose=False)
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
                      n_networks=n_networks, n_jobs=1, seed=0)
    sampler.run(discard_exploration=discard_exploration, f_live=0.1,
                verbose=False)

    assert np.abs(sampler.evidence()) < 0.05

    for equal_weight in [True, False]:
        points, log_w, log_l = sampler.posterior(equal_weight=equal_weight)
        weights = np.exp(log_w)

        mean_sampler = np.average(points, weights=weights, axis=0)
        cov_sampler = np.cov(points, aweights=weights, rowvar=False)
        assert np.all(np.isclose(mean_sampler, mean, atol=0.005, rtol=0))
        assert np.all(np.isclose(cov_sampler, cov, atol=0.001, rtol=0))

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
                      n_networks=0, n_jobs=1, seed=0)
    sampler.run(f_live=0.1, n_eff=0)

    # The effective sample size should be very close to the number of calls
    # since the likelihood is extremely flat.
    assert np.isclose(sampler.n_like, sampler.effective_sample_size(), rtol=0,
                      atol=1)
    # Only one bound should be left.
    assert len(sampler.bounds) == 1
    # Check evidence accuracy.
    assert np.isclose(sampler.evidence(), -4 * 0.5**3 / 3 * 0.001, rtol=0,
                      atol=1e-4)


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

    sampler = Sampler(prior, likelihood, n_dim=2, n_networks=1, n_jobs=1,
                      seed=0)
    sampler.run()
    assert np.isclose(log_z_true, sampler.evidence(), rtol=0, atol=0.1)
    # Check whether the boundaries nautilus drew are strictly nested.
    shell_bound_occupation = sampler.shell_bound_occupation()
    if np.all(shell_bound_occupation ==
              np.tril(np.ones_like(shell_bound_occupation))):
        warnings.warn('The funnel distribution was too easy.', RuntimeWarning)

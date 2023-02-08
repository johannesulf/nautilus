import numpy as np
import pytest

from scipy.stats import multivariate_normal

from nautilus import Prior, Sampler


@pytest.mark.parametrize("use_neural_networks", [True, False])
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize("pass_dict", [True, False])
def test_sampler_basic(use_neural_networks, vectorized, pass_dict):
    # Test that the sampler does not crash.

    if pass_dict:
        prior = Prior()
        prior.add_parameter('a')
        prior.add_parameter('b')
    else:
        def prior(x):
            return x

    def likelihood(x):
        if isinstance(x, dict):
            x = np.squeeze(np.column_stack([x['a'], x['b']]))
        return -np.linalg.norm(x - 0.5, axis=-1) * 0.001

    sampler = Sampler(
        prior, likelihood, n_dim=2, use_neural_networks=use_neural_networks,
        vectorized=vectorized, pass_dict=pass_dict, n_live=500)
    sampler.run(f_live=0.45, n_eff=0, verbose=True)
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
        prior, likelihood, n_dim=2, use_neural_networks=False,
        vectorized=vectorized, pass_dict=pass_dict, n_live=500)
    sampler.run(f_live=0.45, n_eff=0, verbose=True)
    points, log_w, log_l = sampler.posterior(return_as_dict=pass_dict)
    if custom_prior and pass_dict:
        with pytest.raises(ValueError):
            points, log_w, log_l = sampler.posterior(return_as_dict=False)


@pytest.mark.parametrize("use_neural_networks", [True, False])
@pytest.mark.parametrize("discard_exploration", [True, False])
def test_sampler_accuracy(use_neural_networks, discard_exploration):
    # Do a basic tests of the sampler accuracy.

    n_dim = 2
    mean = np.repeat(0.5, n_dim)
    cov = np.eye(n_dim) * 0.01

    def prior(x):
        return x

    def likelihood(x):
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)

    sampler = Sampler(prior, likelihood, n_dim=n_dim, n_live=500,
                      use_neural_networks=use_neural_networks,
                      random_state=0)
    sampler.run(discard_exploration=discard_exploration, f_live=0.1,
                verbose=True)

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

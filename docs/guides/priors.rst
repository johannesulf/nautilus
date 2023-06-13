Prior Probability
=================

In Bayesian statistics, one needs to specify a prior probability :math:`p(\theta)` of model parameters :math:`\theta` where :math:`\theta = (\theta_1, \theta_2, ..., \theta_n)`. However, nested sampling algorithms generally assume that model parameters :math:`u` are uniformly distributed in the unit hypercube. Thus, one needs to perform a variable transformation :math:`\theta = f(u)`. A general recipe for separable priors, i.e. :math:`p(\theta) = p_1(\theta_1) \times ... \times p_n(\theta_n)`, is to use

.. math::

    u_i = \int\limits_{-\infty}^{\theta_i} p(x) dx \, ,

i.e. the sampler uses the cumulative distribution function of :math:`\theta_i` as the variable. In the general case, including for non-separable priors, the transformation must fulfill

.. math::

    p(\theta) = | J^{-1} (\theta) | \, ,

where :math:`|J^{-1}|` is the determinant of the Jacobian of :math:`f^{-1}`.

Standard Priors
---------------

In most cases, the prior will be separable, and the one-dimensional priors will follow well-known distributions such as uniform or normal distributions. In this case, one can use the :py:meth:`nautilus.Prior` class. For each model parameter, we can pass a ``scipy.stats`` `distribution <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. For example, let's assume our model has three parameters :math:`a`, :math:`b` and :math:`c`. :math:`a` follows a uniform distribution within :math:`[1, 3]`, :math:`b` a normal distribution :math:`\mathcal{N}(2, 0.5)` and :math:`c` a gamma distribution :math:`\Gamma(1.0, 2.0)`. We can specify this with the following code.

.. code-block:: python

    from scipy.stats import norm, gamma
    from nautilus import Prior

    prior = Prior()
    prior.add_parameter('a', dist=(1, 3))
    prior.add_parameter('b', dist=norm(loc=2.0, scale=0.5))
    prior.add_parameter('c', dist=gamma(1.0, scale=1.0 / 2.0))

The advantage of :py:meth:`nautilus.Prior` is also that it is straightforward to implement nested models. For example, let's assume the parameters :math:`a`, :math:`b` and :math:`c` describe model B whereas model A is the same as B, just that :math:`c` is fixed to 0. In this case, model A is nested within B, or, in other words, the parameter space of A is a subset of the more general model B. To explore the parameter posterior of A, we only need to change the prior and, if we pass a dictionary to it, can leave the likelihood function untouched.

.. code-block:: python

    from scipy.stats import norm, gamma
    from nautilus import Prior

    prior = Prior()
    prior.add_parameter('a', dist=(1, 3))
    prior.add_parameter('b', dist=norm(loc=2.0, scale=0.5))
    prior.add_parameter('c', dist=0)


Custom Priors
-------------

If one wants to use a distribution function not available in SciPy, one can manually specify the inverse transformation :math:`f`. This will also be the case if the prior is not separable. For example, let's assume we have two parameters. We want the first model parameter to follow a uniform distribution in the range :math:`[0, 1]` and the second parameter to follow a uniform distribution in the range :math:`[0, x]`. This can be achieved with the following code.

.. code-block:: python

    def prior(x):
        theta = np.copy(x)
        theta[..., -1] *= theta[..., 0]
        return theta

Note that this is the same way that dynesty, UltraNest and PyMultiNest pass the prior to the sampler.


Without Transformations
-----------------------

If one has a prior :math:`p(x)` and does not want to or cannot express this as a transformation from the unit cube, one can run ``nautilus`` by absorbing the prior into the likelihood, i.e. :math:`\mathcal{L}(x) \rightarrow \mathcal{L}(x) p(x)`. However, one still needs to define integration ranges :math:`x_{\rm min}` and :math:`x_{\rm max}`, and make sure that they cover (practically) all of the probability mass of the prior. While posterior samples are already accurate using this method, the estimate of the evidence will be offset by :math:`\int_{x_{\rm min}}^{x_{\rm max}} dx / \int_{x_{\rm min}}^{x_{\rm max}} p(x) dx`. Fortunately, this factor can be estimated with ``nautilus`` by analyzing the prior as if it was the likelihood. Here's an example application of a two-dimensional model where we can compute the evidence using the recommended way to express the prior and the method described here.

.. code-block:: python

    import matplotlib.pyplot as plt
    import corner
    import numpy as np
    
    from nautilus import Prior, Sampler
    from scipy.stats import norm, gamma, multivariate_normal
    
    # First, let's do it the "right" way.
    prior = Prior()
    prior.add_parameter('a', dist=(1, 3))
    prior.add_parameter('b', dist=norm(loc=2.0, scale=0.5))
    prior.add_parameter('c', dist=gamma(1.0, scale=1.0 / 2.0))


    def likelihood(param_dict):
        x = [param_dict[key] for key in 'abc']
        return multivariate_normal.logpdf(x, mean=[1.5, 0.5, 1.5], cov=0.01)


    sampler = Sampler(prior, likelihood)
    sampler.run(verbose=True)
    log_z = sampler.evidence()

    # Now, let's use the trick above. First, we need to choose the integration ranges.
    prior_flat = Prior()
    prior_flat.add_parameter('a', dist=(1, 3))
    prior_flat.add_parameter('b', dist=(0, 4))
    prior_flat.add_parameter('c', dist=(0, 4))


    def prior_probablity(param_dict):
        return np.sum([prior.dists[prior.keys.index(key)].logpdf(param_dict[key])
                       for key in 'abc'])
    
    
    def likelihood_plus_prior(param_dict):
        return likelihood(param_dict) + prior_probablity(param_dict)


    sampler = Sampler(prior_flat, prior_probablity)
    sampler.run(verbose=True)
    log_z_prior = sampler.evidence()

    sampler = Sampler(prior_flat, likelihood_plus_prior)
    sampler.run(verbose=True)
    log_z_likelihood_plus_prior = sampler.evidence()

    # Let's compare the two evidence estimates.
    print('log Z estimates: {:.2f} vs. {:.2f}'.format(
        log_z, log_z_likelihood_plus_prior - log_z_prior))

Output::

    log Z estimates: -7.55 vs -7.55

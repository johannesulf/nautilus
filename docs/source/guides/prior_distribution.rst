Prior Distribution
==================

In Bayesian statistics, one needs to specify a prior probability :math:`p(\theta)` of model parameters :math:`\theta` where :math:`\theta = (\theta_1, \theta_2, ..., \theta_n)`. However, nested sampling algorithms generally assume that model parameters :math:`u` are uniformly distributed in the unit hypercube. Thus, one needs to perform a variable transformation :math:`\theta = f(u)`. A general recipe for separable priors, i.e. :math:`p(\theta) = p_1(\theta_1) \times ... \times p_n(\theta_n)`, is to use

.. math::

    u_i = \int\limits_{-\infty}^{\theta_i} p(x) dx \, ,

i.e. the sampler uses the cumulative distribution function of :math:`\theta_i` as the variable. In the general case, including for non-separable priors, the transformation must fulfill

.. math::

    p(\theta) = | J^{-1} (\theta) | \, ,

where :math:`|J^{-1}|` is the determinant of the Jacobian of :math:`f^{-1}`.

Standard Priors
---------------

In most cases, the prior will be separable, and the one-dimensional priors will follow well-known distributions such as uniform or normal distributions. In this case, one can use the ``prior`` module provided by ``nautilus``. For each model parameter, we can pass a ``scipy.stats`` `distribution <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. For example, let's assume our model has three parameters :math:`a`, :math:`b` and :math:`c`. :math:`a` follows a uniform distribution within :math:`[1, 3]`, :math:`b` a normal distribution :math:`\mathcal{N}(2, 0.5)` and :math:`c` a gamma distribution :math:`\Gamma(1.0, 2.0)`. We can specify this with the following code.

.. code-block:: python

    from scipy.stats import norm, gamma
    from nautilus import Prior

    prior = Prior()
    prior.add_parameter('a', dist=(1, 3))
    prior.add_parameter('b', dist=norm(loc=2.0, scale=0.5))
    prior.add_parameter('c', dist=gamma(1.0, scale=1.0 / 2.0))

Custom Priors
-------------

If one wants to use a distribution function not available in ``scipy``, one can manually specify the inverse transformation :math:`f`. This will also be the case if the prior is not separable. For example, let's assume we have two parameters. We want the first model parameter to follow a uniform distribution in the range :math:`[0, 1]` and the second parameter to follow a uniform distribution in the range :math:`[0, x]`. This can be achieved with the following code.

.. code-block:: python

    def prior(x):
        theta = np.copy(x)
        theta[..., -1] *= theta[..., 0]
        return theta

Note that this is the same way that ``dynesty``, ``ultranest`` and ``pymultinest`` pass the prior to the sampler.

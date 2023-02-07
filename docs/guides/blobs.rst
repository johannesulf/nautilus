Blobs
=====

Nautilus supports saving ancillary information returned from the likelihood function. This feature is called "blobs" and closely follows the implementation in emcee.

Here's a demonstration of how we can use this feature to track the amount of time each likelihood evaluation takes.

.. code-block:: python

    import time
    import numpy as np
    from nautilus import Prior, Sampler
    from scipy.stats import multivariate_normal

    prior = Prior()
    for key in 'abc':
        prior.add_parameter(key)

    def likelihood(param_dict):
        t_start = time.time()
        x = [param_dict[key] for key in 'abc']
        t_end = time.time()
        return (multivariate_normal.logpdf(x, mean=[0.4, 0.5, 0.6], cov=0.01),
                t_end - t_start)

    sampler = Sampler(prior, likelihood)
    sampler.run(verbose=True)

    points, log_w, log_l, t = sampler.posterior(return_blobs=True)

One can also keep track of several variables. In this case, it is a good idea to name them and specify the data type. This can be done as follows.

.. code-block:: python

    import time
    import numpy as np
    from nautilus import Prior, Sampler
    from scipy.stats import multivariate_normal

    prior = Prior()
    for key in 'abc':
        prior.add_parameter(key)

    def likelihood(param_dict):
        t_start = time.time()
        x = [param_dict[key] for key in 'abc']
        t_end = time.time()
        return (multivariate_normal.logpdf(x, mean=[0.4, 0.5, 0.6], cov=0.01),
                t_end - t_start, np.median(x))

    sampler = Sampler(prior, likelihood,
                      blobs_dtype=[('time', float), ('median', '|S10')])
    sampler.run(verbose=True)

    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    print(blobs['time'][:3], blobs['median'][:3])

This will print the following output.

.. code-block:: python

    [5.00679016e-06 2.52723694e-05 6.43730164e-06] [b'0.38478858' b'0.15641548' b'0.16970572']

If more than one blob is specified, the ``blobs`` object will be a structured NumPy array with different columns. If no names were specified, the different columns are called "blob_0", "blob_1" etc.

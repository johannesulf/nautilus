FAQs
====

The following page collects commonly asked questions about nautilus. Feel free to reach out, for example, by opening a `GitHub <https://github.com/johannesulf/nautilus/issues>`_ issue if you don't find your question answered.

Does ``nautilus`` scale well to high dimensions?
------------------------------------------------

``nautilus`` has been tested successfully for problems with up to 60 dimensions. While the scaling is very favorable, up to ~50 dimensions, it will eventually break down. In this case, ``nautilus`` may need many likelihood evaluations and have a large computational overhead. For problems with hundreds of parameters, other samplers based on Hamiltonian Monte-Carlo or slice sampling may be better suited.

Does ``nautilus`` use GPUs for neural networks?
-----------------------------------------------

No, ``nautilus`` uses CPUs for all computations. There are two primary motivations for this. First, one overarching philosophy is to make ``nautilus`` easy to use with few dependencies. Packages like ``pytorch`` or ``tensorflow`` that implement neural network calculations on GPUs are typically more challenging to install. Instead, ``nautilus`` depends on ``scikit-learn``, a widely-used and easy-to-install machine learning package. Second, ``nautilus`` uses fairly shallow networks and small training sizes. In these scenarios, unlike for very deep networks and large training sizes, the speed-up offered by GPUs may not be that large.

Is ``nautilus`` a likelihood emulator?
--------------------------------------

The sampler uses an emulator for the likelihood to determine which parts of parameter space to sample. However, the actual posterior samples and evidence estimates derive from the true likelihood, not an emulation of the likelihood. If the likelihood emulator is inaccurate, this will lower the sampling efficiency, i.e., more likelihood evaluations are needed to achieve a certain accuracy of the results. However, the accuracy of the posterior and evidence estimates should be unchanged. Overall, ``nautilus`` can solve a variety of challenging problems with percent-level accuracy.

How many networks should I use?
-------------------------------

``nautilus`` uses multiple neural networks to determine which part of parameter space to sample from. By default, ``nautilus`` averages the results from 4 independent networks. This value can be adjusted by the user through the ``n_networks`` keyword argument of :py:class:`nautilus.Sampler`. Averaging the results from 4 networks gives ``nautilus`` a noticeably better performance than one network in various problems. Increasing the number further may lead to further improvements, i.e., fewer likelihood evaluations needed. At the same time, this may increase the overall runtime due to the increased cost of training networks. Note that when ``nautilus`` trains networks, it trains in parallel with each network using a single CPU core. Thus, if ``nautilus`` uses multiple CPU cores, increasing the number of networks up to the CPU core limit may not even increase the time to train the networks. Conversely, if the number of CPU cores is lower than the number of networks, reducing the number of networks may reduce overall runtime.

Why is the posterior noisy despite having a large effective sample size?
------------------------------------------------------------------------

This commonly happens if users are working with equal-weight posteriors by setting ``equal_weight=True``. However, ``nautilus`` naturally produces an unequal-weight posterior. The equal-weight posterior is effectively a noisy estimate of the unequal-weight posterior. It is not recommended to work with the equal-weight posterior if one wants the most precise posterior estimate.

By default, the equal-weight posterior is drawn under the condition that no posterior point appears twice and can be very noisy. However, by setting ``equal_weight_boost`` to a value larger than unity, the equal-weight posterior is allowed to have duplicates. The higher the value of ``equal_weight_boost``, the better the equal-weight posterior approximates the unequal-weight posterior, increasing its precision. This is the recommended solution if the downstream analysis cannot handle unequal-weight posteriors. This feature was introduced in ``nautilus`` version 1.0.5.

What is the uncertainty in the evidence :math:`\log \mathcal{Z}`?
-----------------------------------------------------------------

In general, there are two sources of uncertainty for :math:`\log \mathcal{Z}`: statistical and systematic. "Statistical" here refers to the scatter in :math:`\log \mathcal{Z}` between repeated runs with the same settings while "systematic" denotes any potential bias in :math:`\log \mathcal{Z}` from the true result over repeated runs. The latter would be non-zero in case nautilus was run using settings that aren't sufficiently "converged."

In practice, the statistical uncertainty on :math:`\log \mathcal{Z}` can be very well approximated by :math:`1 / \sqrt{N_\mathrm{eff}}`, where :math:`N_\mathrm{eff}` is the effective sample size. That means, by default (:math:`N_\mathrm{eff} = 10,000`), nautilus will determine :math:`\log \mathcal{Z}` with a statistical uncertainty of around :math:`\Delta \log \mathcal{Z} \approx 0.01`. This turns out to be substantially smaller than typical uncertainties from traditional nested samplers.

On the other hand, quantitatively assessing the systematic uncertainty is very hard and requires repeated runs with different settings. Fortunately, given the very small statistical uncertainty, even small systematic biases are easy to detect. One recommendation is to vary the number of live points. Additionally, it is always recommended  to set ``discard_exploration=True`` for publication results.

`nautilus` hangs when using parallelization in Jupyter notebooks. What can I do?
--------------------------------------------------------------------------------

If this happens, the easiest solutions is to create a pool using the `multiprocess` module as opposed to the standard `multiprocessing` library. Here's a simple example.

.. code-block:: python

    import multiprocess as mp
    import numpy as np
    from nautilus import Prior, Sampler
    from scipy.stats import multivariate_normal

    prior = Prior()
    for key in 'abc':
        prior.add_parameter(key)

    def likelihood(param_dict):
        x = [param_dict[key] for key in 'abc']
        return multivariate_normal.logpdf(x, mean=[0.4, 0.5, 0.6], cov=0.01)

        with mp.Pool(50) as pool:
            sampler = Sampler(prior, likelihood, pool=pool)
            sampler.run(verbose=True)

Thanks to `Paul Shah <https://github.com/johannesulf/nautilus/issues/64>`_ for this suggestion!

When using checkpointing, I occasionally get `BlockingIOError`. What can I do?
------------------------------------------------------------------------------

`nautilus` uses HDF5 for writing checkpoint files via the `h5py` library. Occasionally, `h5py` may encounter a `BlockingIOError`. This is typically caused by file system delays in the release of the lock file, making it appear as if another process is still writing to the checkpoint file. However, `nautilus` writes to the checkpoint file sequentially. Thus, disabling file locking is generally a safe and effective workaround for this issue. To disable file locking, you can set an environment variable before calling the python script or starting a jupyter notebook via the following shell command.

.. code-block:: bash

    export HDF5_USE_FILE_LOCKING=FALSE

Alternatively, you may use the following Python code.

.. code-block:: python

    import os
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

**Important**: To make sure this works, you should call this Python command before importing `nautilus`, `h5py` or any other package making use of HDF5.

Thanks to `Ho-Hin Leung <https://github.com/johannesulf/nautilus/issues/47>`_ for help investigating this issue!

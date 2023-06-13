Checkpointing
=============

``nautilus`` supports application checkpointing. This means that sampling runs can be interrupted and continued later. Checkpointing is an excellent way to guard against extensive data loss from interruptions and compute node failures. To use checkpointing, h5py must be installed. To enable checkpointing, specify a file path via the ``filepath`` keyword argument when initializing :py:meth:`nautilus.Sampler`.

.. code-block:: python

    import numpy as np
    from nautilus import Prior, Sampler
    from scipy.stats import multivariate_normal

    prior = Prior()
    for key in 'abc':
        prior.add_parameter(key)

    def likelihood(param_dict):
        x = [param_dict[key] for key in 'abc']
        return multivariate_normal.logpdf(x, mean=[0.4, 0.5, 0.6], cov=0.01)

    sampler = Sampler(prior, likelihood, filepath='checkpoint.hdf5')
    sampler.run(verbose=True)

By default, ``nautilus`` will resume a previous run if it finds a file under this path. After a crash, one can call the same script again and continues where one left off. However, be careful not to initialize a sampler for one likelihood problem with the checkpoint of a different likelihood problem. Finally, to overwrite any file under that file path and ignore previous calculations, use ``resume=False``.

.. code-block:: python

    sampler = Sampler(prior, likelihood, filepath='checkpoint.hdf5', resume=False)
    sampler.run(verbose=True)

When checkpointing is activated, ``nautilus`` will save the progress after each shell is filled during the exploration phase and after each batch in the sampling phase.

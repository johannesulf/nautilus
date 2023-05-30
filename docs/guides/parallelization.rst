Parallelization
===============

Nautilus supports distributing computations over multiple CPU cores and nodes on a high-performance computing (HPC) cluster. The parallelization includes likelihood calculations and the most computationally demanding aspects of the sampler, i.e., neural network training and sampling new points. Let's take the following simple problem. We added ``time.sleep(0.01)`` so that every likelihood evaluation takes a non-negligible amount of time, around ten milliseconds. This simulates a computationally more expensive likelihood problem.

.. code-block:: python

    import numpy as np
    import time
    
    from nautilus import Prior, Sampler
    
    prior = Prior()
    for i in range(3):
        prior.add_parameter()
    
    
    def likelihood(x):
        time.sleep(0.01)
        return np.sum(-(x - 0.5)**2 / (2 * 0.1**2))
    
    
    sampler = Sampler(prior, likelihood, pass_dict=False)
    t_start = time.time()
    sampler.run(verbose=True)
    t_end = time.time()
    print('Total time: {:.1f}s'.format(t_end - t_start))

For this example, the computation took around 330 seconds. Let's see how to speed this up with parallelization.

SMP Parallelization
-------------------

If you're running nautilus on a computer or laptop, shared-memory multiprocessing (SMP) is likely the most straightforward way to distribute computations. All you need to do is give a number to the `pool` keyword argument. In this example, let's run the computation on 4 CPU cores.

.. code-block:: python

    sampler = Sampler(prior, likelihood, pass_dict=False, pool=4)
    t_start = time.time()
    sampler.run(verbose=True)
    t_end = time.time()
    print('Total time: {:.1f}s'.format(t_end - t_start))

Using 4 CPU cores, the computation took around 100 seconds, i.e., it was about 3.3 times faster than without parallelization.

MPI Parallelization
-------------------

On an HPC cluster with multiple distinct CPUs and nodes without shared memory, SMP parallelization will not work. In this case, use Message Passing Interface (MPI) parallelization. Here, we use that ``pool`` can also be any pool, i.e., an instance of ``mpi4py.futures.MPIPoolExecutor``.

.. code-block:: python

    if __name__ == '__main__':
        sampler = Sampler(prior, likelihood, pass_dict=False, pool=MPIPoolExecutor(4))
        t_start = time.time()
        sampler.run(verbose=True)
        t_end = time.time()
        print('Total time: {:.1f}s'.format(t_end - t_start))

The above script should be executed via ``mpiexec -n 1 python -m mpi4py.futures script.py``. Using 4 workers, the computation took around 110 seconds, i.e., it was about 3.0 times faster than without parallelization.

Notes
-----

Here are a few additional notes about parallelization. First, note that some functions, i.e., those used in ``numpy``, may distribute operations internally via OpenMP. This may interfere with the top-level parallelization performed by nautilus. Thus, it may be beneficial to deactivate OpenMP parallelization manually.

.. code-block:: python

    import os
    
    os.environ["OMP_NUM_THREADS"] = "1"

Finally, in some situations, it may be beneficial to use different parallelization schemes for the likelihood evaluations and the sampler calculations. The keyword argument ``pool`` also be a tuple defining two pools. In this case, the first is used for likelihood calculations and the second for sampler calculations. For example, to parallelize likelihood evaluations but not sampler calculations, use ``pool=(4, None)``.

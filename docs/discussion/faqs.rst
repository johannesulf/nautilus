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
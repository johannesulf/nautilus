FAQs
====

The following page collects commonly asked questions about nautilus. Feel free to reach out, for example, by opening a `GitHub <https://github.com/johannesulf/nautilus/issues>`_ issue if you don't find your question answered.

**Does the sampler scale well to high dimensions?**

``nautilus`` has been tested successfully for problems with up to 60 dimensions. While the scaling is very favorable, up to ~50 dimensions, it will eventually break down. In this case, ``nautilus`` may need many likelihood evaluations and have a large computational overhead. For problems with hundreds of parameters, other samplers based on Hamiltonian Monte-Carlo or slice sampling may be better suited.

**Since the sampler uses neural networks, does it utilize GPUs?**

No, ``nautilus`` uses the CPU for all computations. There are two primary motivations for this. First, one overarching philosophy is to make ``nautilus`` easy to use with few dependencies. Packages like ``pytorch`` or ``tensorflow`` that implement neural network calculations on GPUs are typically more challenging to install. Instead, ``nautilus`` depends on ``scikit-learn``, a widely-used and easy-to-install machine learning package. Second, ``nautilus`` uses fairly shallow networks and small training sizes. In these scenarios, unlike for very deep networks and large training sizes, the speed-up offered by GPUs may not be that large.

**Is the sampler a likelihood emulator? How accurate are the results?**

The sampler uses an emulator for the likelihood to determine which parts of parameter space to sample. However, the actual posterior samples and evidence estimates derive from the true likelihood, not an emulation of the likelihood. If the likelihood emulator is inaccurate, this will lower the sampling efficiency, i.e., more likelihood evaluations are needed to achieve a certain accuracy of the results. However, the accuracy of the posterior and evidence estimates should be unchanged. Overall, ``nautilus`` can solve a variety of challenging problems with percent-level accuracy.

.. image:: nautilus_text.png
   :width: 100 %
   :align: center

|

.. image:: https://img.shields.io/github/actions/workflow/status/johannesulf/nautilus/tests.yml?branch=main&label=tests
.. image:: https://img.shields.io/coverallsCoverage/github/johannesulf/nautilus
.. image:: https://img.shields.io/readthedocs/nautilus-sampler
.. image:: https://img.shields.io/pypi/v/nautilus-sampler?color=blue
.. image:: https://img.shields.io/conda/vn/conda-forge/nautilus-sampler?color=blue
.. image:: https://img.shields.io/github/license/johannesulf/nautilus?color=blue
.. image:: https://img.shields.io/github/languages/top/johannesulf/nautilus

|

Overview
--------

Nautilus is an MIT-licensed pure-Python package for Bayesian posterior and evidence estimation. It utilizes importance sampling and efficient space exploration using neural networks. Compared to traditional MCMC and Nested Sampling codes, it needs fewer likelihood calls and produces much larger posterior samples. Additionally, nautilus is highly accurate and produces Bayesian evidence estimates with percent precision.

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    guides/installation
    guides/crash_course
    guides/priors
    guides/parallelization
    guides/checkpointing
    guides/blobs


.. toctree::
    :maxdepth: 1
    :caption: Discussion

    discussion/nested_sampling
    discussion/convergence
    discussion/faqs

.. toctree::
    :maxdepth: 1
    :caption: API Documentation

    api

Support
-------

If you are encountering issues with nautilus, please raise an issue on the nautilus `GitHub <https://github.com/johannesulf/nautilus/issues>`_ page. If you have suggestions to improve this tutorial or would like to request features, you can use the same procedure or reach out to the authors.

Attribution
-----------

A paper describing nautilus's underlying methods and performance has been submitted for publication. A draft of the paper will be uploaded to arXiv around the end of June 2023. Please cite the paper, once available, if you found nautilus helpful in your research.


License
-------

The project is licensed under the MIT license. The logo uses an image from the Illustris Collaboration.

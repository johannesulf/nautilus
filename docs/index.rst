.. image:: nautilus_text_black.png
   :width: 100 %
   :align: center
   :class: only-light

.. image:: nautilus_text_white.png
   :width: 100 %
   :align: center
   :class: only-dark

|

.. image:: https://img.shields.io/github/actions/workflow/status/johannesulf/nautilus/tests.yml?branch=main&label=tests
.. image:: https://img.shields.io/coverallsCoverage/github/johannesulf/nautilus
.. image:: https://img.shields.io/readthedocs/nautilus-sampler
.. image:: https://img.shields.io/pypi/v/nautilus-sampler
.. image:: https://img.shields.io/conda/vn/conda-forge/nautilus-sampler
.. image:: https://img.shields.io/github/license/johannesulf/nautilus
.. image:: https://img.shields.io/github/languages/top/johannesulf/nautilus

|

Overview
========

Nautilus is an MIT-licensed pure-Python package for Bayesian posterior and evidence estimation. It utilizes importance sampling and efficient space exploration using neural networks. Compared to traditional MCMC and Nested Sampling codes, it often needs fewer likelihood calls and produces much larger posterior samples. Additionally, nautilus is highly accurate and produces Bayesian evidence estimates with percent precision. It is widely used in many areas of astrophysical research.

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    guides/installation
    guides/quickstart
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

    api_high
    api_full

Support
-------

If you are encountering issues with nautilus, please raise an issue on the nautilus `GitHub <https://github.com/johannesulf/nautilus/issues>`_ page. If you have suggestions to improve this tutorial or would like to request features, you can use the same procedure or reach out to the authors.

Attribution
-----------

A paper describing nautilus's underlying methods and performance has been published in the `Monthly Notices of the Royal Astronomical Society <https://academic.oup.com/mnras/article/525/2/3181/7243406>`_. A draft of the paper is also available on `arXiv <https://arxiv.org/abs/2306.16923>`_. Please cite the paper if you find nautilus helpful in your research.

.. code-block::

    @article{nautilus,
        author = {Lange, Johannes U},
        title = "{nautilus: boosting Bayesian importance nested sampling with deep learning}",
        journal = {Monthly Notices of the Royal Astronomical Society},
        volume = {525},
        number = {2},
        pages = {3181-3194},
        year = {2023},
        month = {08},
        doi = {10.1093/mnras/stad2441},
        url = {https://doi.org/10.1093/mnras/stad2441},
        eprint = {https://academic.oup.com/mnras/article-pdf/525/2/3181/51331635/stad2441.pdf},
    }

License
-------

The project is licensed under the MIT license. The logo uses an image from the Illustris Collaboration.

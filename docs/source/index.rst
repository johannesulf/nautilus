.. image:: nautilus_text.png
   :width: 100 %
   :align: center

|

Overview
--------

``nautilus`` is an MIT-licensed pure-Python package for Bayesian posterior and evidence estimation. It utilizes importance sampling and efficient space tessellation using neural networks. Compared to traditional MCMC and Nested Sampling codes, it needs fewer likelihood calls and produces much larger posterior samples. Additionally, ``nautilus`` is very accurate and produces accurate Bayesian evidence estimates with percent precision.

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    guides/installation
    guides/crash_course
    guides/prior_distribution
    guides/nested_sampling

.. toctree::
    :maxdepth: 1
    :caption: API Documentation

    api

Support
-------

If you are encountering issues with ``nautilus``, please raise an issue on the ``nautilus`` `GitHub <https://github.com/johannesulf/nautilus/issues>`_ page. If you have suggestions to improve this tutorial or would like to request features, you can use the same procedure or reach out to the authors.

License
-------

The project is licensed under the MIT license. The logo uses an image from the Illustris Collaboration.

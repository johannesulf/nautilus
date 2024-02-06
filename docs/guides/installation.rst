Installation
============

The most recent stable version of ``nautilus`` is listed in the Python Package Index (PyPI) and can be installed via ``pip``.

.. code-block:: bash

    pip install nautilus-sampler

Additionally, ``nautilus`` is also on conda-forge. To install via ``conda`` use the following command.

.. code-block:: bash

    conda install -c conda-forge nautilus-sampler

Finally, you can install the latest development version from GitHub.

.. code-block:: bash

    git clone https://github.com/johannesulf/nautilus.git
    cd nautilus
    pip install .

``nautilus`` depends on a small number of fairly standard packages: ``NumPy``, ``ScipPy``, ``scikit-learn``, and ``threadpoolctl``. Optionally, to use :ref:`checkpointing <Checkpointing>` you need to have ``h5py`` installed.

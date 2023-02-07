Installation
============

The most recent stable version of nautilus is listed in the Python Package Index (PyPI) and be installed via `pip`.

.. code-block:: bash

    pip install nautilus-sampler

Alternatively, you can install the latest development version from GitHub.

.. code-block:: bash

    git clone https://github.com/johannesulf/nautilus.git
    cd nautilus
    pip install .

Dependencies
------------

nautilus depends on a small number of fairly standard packages: NumPy, ScipPy, scikit-learn, tqdm and threadpoolctl. Optionally, to use :ref:`checkpointing <Checkpointing>` you need to have h5py installed.

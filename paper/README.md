This directory contains all the code needed to reproduce the results and figures for the upcoming paper describing `nautilus`. To run a test, execute the `compute.py` script.

```console
python compute.py likelihood
```

The likelihood can be "loggamma", "funnel", "rosenbrock", "cosmology", "galaxy" or "exoplanet". For the former three cases, you must specify the dimensionality, i.e., "funnel-10" for the 10-dimensional funnel likelihood. The script has additional options. To see those, type the following command.

```console
python compute.py --help
```

Once tests have been run, the script `analyze.py` script is used to produce figures and tables.

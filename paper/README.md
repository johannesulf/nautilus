This directory contains all the code needed to reproduce the results and figures for the upcoming paper describing `nautilus`. To run a test, execute the `compute.py` script.

```console
python compute.py likelihood sampler1 sampler2 ...
```

The likelihood can be "loggamma", "funnel", "rosenbrock", "cosmology", "galaxy" or "exoplanet". For the former three cases, you must specify the dimensionality, i.e., "funnel-10" for the 10-dimensional funnel likelihood. The sampler can be "nautilus", "dynesty-u" (dynesty with uniform sampling), "dynesty-r" (random walk sampling), "dynesty-s" (slice sampling), "UltraNest", "pocoMC", or "emcee" (1000 walkers run until 1000 auto-correlation times). The script has additional options. To see those, type the following command.

```console
python compute.py --help
```

Once tests have been run, the script `analyze.py` script is used to produce figures and tables.

Here is a list of the sub-directories and their contents.

* draft: The paper draft together with all the necessary figures and tables.
* figures: All the figures produced by `analyze.py`.
* likelihoods: All the likelihoods are defined here.
* lux: Scripts to run computations on the UCSC lux cluster.
* pipes: Empty directory created by `bagpipes` if the galaxy likelihood is run.
* results: Results of all compuations are saved here.

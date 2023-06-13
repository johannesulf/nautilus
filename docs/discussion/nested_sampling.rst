Nested Sampling
===============

Nested Sampling (NS) describes a class of algorithms used to solve Bayesian inference problems. ``nautilus`` originated from NS, and, as such, it shares similarities with other NS codes such as ``dynesty``, ``MultiNest``, and ``UltraNest``. Here, we describe the key differences and similarities between NS codes and ``nautilus``.

Algorithm Differences
---------------------

The primary objectives of NS codes and ``nautilus`` are identical: identify the parts of parameter space with the highest likelihood, sample points from the parameter posterior, and estimate the Bayesian evidence. Additionally, NS codes and ``nautilus`` work similarly during initial parameter space exploration. Both randomly sample the prior space, identify the points with the highest likelihood, the so-called live set, and then estimate a boundary around the live set. Afterward, points are sampled from within the live set boundary, and a new live set with a higher minimum likelihood is identified. This procedure, sampling from within a live set boundary and identifying a new live set, is repeated until a convergence criterion is reached and leads to a live set that rapidly shrinks to the maximum likelihood value.

A key difference between NS codes and ``nautilus`` is how the live set boundary is drawn. In particular, unlike the NS codes mentioned above, ``nautilus`` uses a neural network-based algorithm to determine efficient boundaries. Furthermore, NS codes and ``nautilus`` differ significantly in how the Bayesian posterior and evidence are estimated after this initial exploration phase. In particular, ``nautilus`` uses an Importance Nested Sampling (INS) algorithm, not NS. Another big difference between ``nautilus`` and dynamic NS codes such as ``dynesty`` is how additional posterior samples can be added and the evidence refined after the exploration phase. Dynamic NS codes use separate NS runs that are later combined for that. On the other hand, ``nautilus`` uses the information on the likelihood obtained during the exploration phase to more or less directly sample from the posterior.

General Advice
--------------

``nautilus`` profits from more live points! We recommend running ``nautilus`` with at least 1000 live points. Although the runtime of the algorithm is, in principle, proportional to the number of live points, the increased sampling efficiency with more points more than makes up for that. Often, ``nautilus`` with 3000 live points runs faster than NS codes with 500.

# Changelog

All notable changes to nautilus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2023-04-22

### Changed
- The code now uses the more modern `numpy.random.Generator` framework instead of `numpy.random.RandomState`.
- Added the keyword arguments `n_points_min` and `split_threshold` to the sampler. Previously, they were not accessible.
- The default value for `n_points_min` is now the number of dimensions plus 50. Previously, it was hard-coded to be the number of dimensions plus 1.
- The multi-ellipsoidal decomposition has been tweaked with the goal of reducing computational overhead for high-dimensional problems.
- The default number of parallel processes has been changed to one. By default, the sampler will not use parallelization.
- Multi-ellipsoidal decomposition now uses Gaussian mixture modeling instead of K-Means. The former typically results in better performance, i.e., smaller boundaries with fewer ellipsoids.
- Sampling new points can now be parallelized using the `n_jobs` keyword argument.

### Fixed
- The sampler now correctly writes the random number generator in the sampling phase.
- The keyword argument `n_jobs` is now being correctly passed when training networks. Previously, all cores were used regardless of `n_jobs`.
- The sampler doesn't crash when setting `verbose=True`and `n_networks=0`.

### Deprecated
- The `random_state` keyword argument for the sampler has been deprecated in favor of the new keyword argument `seed`.

## [0.5.0] - 2023-04-02

### Changed
- Introduced neural network ensembles. Instead of relying on a single network, the sampler now uses 4 networks by default. This should lead to better sampling performance. The training of the networks is done in parallel, by default. Which means that on multi-core systems, time spent on neural network training shouldn't increase dramatically. The number of networks and training parallelization can be adjusted by the user.
- Introduced new bounds that lead to lower overhead for high-dimensional problems. The number of likelihood calls should be close to unaffected by this change.
- Increased the default number of live points from 1500 to 2000.
- Limited number of threads to 1 when calculating minimum volume enclosing ellipsoid and doing K-means clustering. More often than not, thread parallelization would slow things down while increasing CPU usage.
- When setting the filepath to "filename.hdf5" and discarding points in the exploration stage, the sampler now creates a backup of the end of the exploration stage under "filename_exp.hdf5".

### Fixed
- Previously, the sampler would sometimes reject new bounds because their volume was estimated to be larger than the previous bound. However, this was based on very noisy volume estimates in certain situations. The sampler now uses more precise volume estimates, which can increase sampling performance for high-dimensional problems since bounds are updated more often.
- The sampler now doesn't perform unnecessary prior transformations when calculating the fraction of the evidence in the live set. This helps lower the computational overhead when using the sampler in CosmoSIS.
- When discarding points in the exploration phase, blobs are now also correctly removed.
- Fixed a crash when `blobs_dtype` is a single elementary dtype and multiple blobs are returned.

### Deprecated
- The `enlarge` keyword argument for the sampler has been deprecated in favor of the new keyword `enlarge_per_dim`. Specifying `enlarge` will be ignored but not raise an error.
- The `use_neural_network` keyword argument for the sampler has been deprecated. To not use neural networks, set the new keyword argument `n_networks` to 0.

## [0.4.4] - 2023-03-14

### Fixed
- Fixed a potential crash when more than one blob is returned per likelihood call.

## [0.4.3] - 2023-03-03

### Changed
- By default, the neural network is now trained longer. This led to better sampling performance in all problems tested.

### Fixed
- Fixed warnings when sampling from likelihoods with so many negative infinities that certain shells have 0 probability.

## [0.4.2] - 2023-02-24

### Changed
- Checkpointing file sizes are much smaller in the sampling phase.

### Fixed
- When sampling from likelihoods with negative infinities, no warnings will appear.

## [0.4.1] - 2023-02-13

### Fixed
- Likelihood function now receives array with `pass_dict=False` when using `nautilus.Prior`.

## [0.4.0] - 2023-02-12

### Changed
- Likelihood functions will now never be passed structured numpy arrays. In case they previously received structured numpy arrays, they will now receive dictionaries. Similarly, `nautilus.Prior.physical_to_structure` has been renamed to `nautilus.Prior.physical_to_dictionary` and the keyword argument of `nautilus.Sampler` from `pass_struct` to `pass_dict`.
- The `n_like_update` keyword argument for `nautilus.Sampler` has been renamed to `n_like_new_bound`.
- The number of points for which the likelihood is evaluated is now always the batch size.
- Renamed `return_dict` keyword argument in `Sampler.posterior` to `return_as_dict`.

### Added
- Checkpointing and writing the sampler to disk.
- Support for so-called blobs, following the implementation in `emcee`.

## [0.3.3] - 2023-01-23

### Changed
- Renamed "tessellation phase" to "exploration_phase". Accordingly, changed the keyword argument `discard_tessellation` to `discard_exploration` for `run` function.
- Increased performance for sampling new points from shells.

## [0.3.2] - 2022-12-20

### Added
- Support for `scikit-learn` 1.2.

## [0.3.1] - 2022-12-08

### Fixed
- Changed `scikit-learn` requirement to < 1.2. Currently, `nautilus` is incompatible with the newly-released `scikit-learn` version.

## [0.3.0] - 2022-11-08

### Added
- Made several parameters such as those of the neural networks available to the user.

### Changed
- Changed the default number of live points and neural network structure.
- Now using `threadpoolctl` to limit the number of cores used by the neural network.
- The keyword argument `threads` passed to the `Sampler` class has been removed and absorbed into the `pool` keyword argument.

## [0.2.1] - 2022-09-02

### Fixed
- Fixed crash in `Prior` class.

## [0.2.0] - 2022-07-19

### Changed
- Improved sampling strategy after tessellation. Sampling each shell now depends on the evidence, the number of likelihood calls and the effective sample size in each shell.
- Changed the way points are assigned to different neural network emulators. Effectively, the number of neural network emulators for each bound has been reduced, often to only one.
- Changed the way neural network emulators are trained.

### Removed
- Removed TensorFlow backend in favor of only using scikit-learn.

### Fixed
- The sampler now raises an error if sampling with less than 2 parameters. (#2)
- The sampler now correctly passes a dictionary as input to the likelihood function if `pass_struct` is True and `vectorized` is False.

## [0.1.0] - 2022-06-28

### Added
- Initial beta release.

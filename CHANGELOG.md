# Changelog

All notable changes to nautilus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.6] - 2025-12-29

### Added
- One can now specify periodic parameters via the `periodic` keyword argument. This should increase sampling efficiency and reduce computational overhead in cases where certain parameters are periodic, such as angles.

### Fixed
- The combination of vectorization and parallelization is now properly implemented and does not crash. (#69)

## [1.0.5] - 2024-10-18

### Added
- The equal-weighted posterior can now be made larger and more precise using the `equal_weight_boost` keyword argument of `sampler.posterior`.

## [1.0.4] - 2024-07-27

### Added
- Nautilus can now use Dask clusters for parallelization. (#49)

## [1.0.3] - 2024-04-29

### Added
- The user can now specify a timeout interval for the sampler. If that time is exceeded, the sampler will not start new calculations. (#46)
- The sampler now returns whether it finished normally or stopped because the timeout or maximum number of likelihood computations was reached. (#46)

## [1.0.2] - 2024-02-15

### Changed
- Further improved the way the sampler handles plateaus. For example, if the lowest-likelihood point in the live set is part of a plateau, the sampler will jump past the plateau if enough live points with higher likelihood exist. The sampler also behaves reasonably if most or all points are part of a plateau. This leads to less freezes and better performance.

## [1.0.1] - 2024-02-12

### Fixed
- Fixed a crash when multiple blobs per likelihood call are returned as a single array.

## [1.0.0] - 2024-02-12

### Added
- The user can now specify a maximum number of likelihood calls. If that number is exceeded, the sampler will automatically stop. (#23)

### Changed
- Updated the terminal output to be more compact and more friendly for log files. This also removes the dependency on `tqdm`. (#36)
- By default, the batch size is now dynamically determined at the start based on the pool size. This should prevent issues for new users parallelizing over a large number of CPUs.

### Fixed
- Fixed a crash when returning the posterior as a dictionary and with equal weight.
- Fixed a potential crash when `n_update` is extremely low.

### Deprecated
- The `evidence`, `asymptotic_sampling_efficiency`, and `effective_sample_size` sampler functions are deprecated and replaced by the `log_z`, `eta`, and `n_eff` properties, respectively.

### Removed
- The `n_jobs` parameter when initializing the sampler has been removed. Use `pool`, instead.

## [0.7.4] - 2023-08-23

### Fixed
- Increased numerical stability when finding the minimum-volume enclosing ellipsoid. Instances of `np.linalg.LinAlgError` should be reduced or eliminated. (#34)

## [0.7.3] - 2023-08-15

### Changed
- When passing a number to the `pool` keyword argument, the sampler automatically caches the likelihood function in the subprocesses of the multiprocessing pool. This reduces communication between processes and can substantially improve performance.
- Changed the way ellipsoids are split, preventing rare instances where ellipsoid splitting stops prematurely. (#28)
- Improved the performance of the algorithm for finding the minimum-volume enclosing ellipsoid.

### Fixed
- Fixed a rare freeze when the neural networks predict the same score for all input values. (#27)
- Prevented a rare freeze if the likelihood is multi-modal and one mode has much less volume than the other. (#29)
- Fixed an occasional crash when resuming a calculation without blobs from a checkpoint file. (#31)

## [0.7.2] - 2023-07-11

### Changed
- The sampler should now deal better with likelihood plateaus.

### Fixed
- The prior function now always receives a copy of the points, preventing buggy behavior if the user's prior function does the prior transform on the original array.

## [0.7.1] - 2023-07-07

### Fixed
- Parallelization with MPIPoolExecutor should now work correctly.

## [0.7.0] - 2023-06-20

### Added
- Added the function `sampler.asymptotic_sampling_efficiency` which returns an estimate of the sampling efficiency in the sampling phase.

### Changed
- One can now change whether to discard points in the exploration phase after calling `run` by changing the `discard_exploration` argument of the sampler. To achieve this, information about points in the exploration phase is never dropped. Consequently, the sampler does not create a backup of the end of the exploration stage under "filename_exp.hdf5", anymore.
- The computational overhead was slightly reduced when new bounds are rejected. Also, the output now specifically mentions when adding a new bound failed because the volume increased.

### Fixed
- Likelihoods with large plateaus shouldn't crash, anymore.
- Information updates about bounds are now written during the sampling phase. Previously, if computations were interrupted and restarted during the sampling phase, the volume estimates were noisier than necessary, and some points may have been proposed twice.

### Removed
- The `n_jobs` keyword argument for the sampler has been removed. The pool used for likelihood calls is now also used for sampler parallelization, by default. To use independent pools for likelihood calls and sampler calculations, pass a tuple to `pool`.
- The `random_state` keyword argument for the sampler has been removed. Use `seed`, instead.
- The `enlarge` keyword argument has been removed. Use `enlarge_per_dim`, instead.

## [0.6.0] - 2023-04-22

### Added
- Added the keyword arguments `n_points_min` and `split_threshold` to the sampler. Previously, they were not accessible.
- Sampling new points can now be parallelized using the `n_jobs` keyword argument.

### Changed
- The code now uses the more modern `numpy.random.Generator` framework instead of `numpy.random.RandomState`.
- The default value for `n_points_min` is now the number of dimensions plus 50. Previously, it was hard-coded to be the number of dimensions plus 1.
- The multi-ellipsoidal decomposition has been tweaked with the goal of reducing computational overhead for high-dimensional problems.
- The default number of parallel processes has been changed to one. By default, the sampler will not use parallelization.
- Multi-ellipsoidal decomposition now uses Gaussian mixture modeling instead of K-Means. The former typically results in better performance, i.e., smaller boundaries with fewer ellipsoids.

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

### Removed
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

# Changelog

All notable changes to nautilus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `nautilus` now raises an error if sampling with less than 2 parameters. (#2)
- `nautilus` now correctly passes a dictionary as input to the likelihood function if `pass_struct` is True and `vectorized` is False.

## [0.1.0] - 2022-06-28

### Added
- Initial beta release.

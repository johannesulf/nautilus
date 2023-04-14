"""Module implementing the Nautilus sampler."""

try:
    import h5py
except ImportError:
    pass
import numpy as np
import warnings

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .bounds import UnitCube, NautilusBound


class Sampler():
    """
    A dynamic sampler built upon the framework of importance nested sampling.

    Attributes
    ----------
    prior : function or nautilus.Prior
        Prior describing the mapping of the unit hypercube to physical
        parameters.
    likelihood : function
        Function returning the natural logarithm of the likelihood.
    n_dim : int
        Number of dimensions of the likelihood function.
    n_live : int
        Number of live points.
    n_update : int
        The maximum number of additions to the live set before a new bound is
        created.
    n_like_new_bound : int
        The maximum number of likelihood calls before a new bounds is created.
    enlarge_per_dim : float
        Along each dimension, outer ellipsoidal bounds are enlarged by this
        factor.
    n_points_min : int or None, optional
        The minimum number of points each ellipsoid should have. Effectively,
        ellipsoids with less than twice that number will not be split further.
    split_threshold: float, optional
        Threshold used for splitting the multi-ellipsoidal bound used for
        sampling. If the volume of the bound prior enlarging is larger than
        `split_threshold` times the target volume, the multi-ellipsiodal
        bound is split further, if possible.
    n_networks : int
        Number of networks used in the estimator.
    neural_network_kwargs : dict
        Keyword arguments passed to the constructor of
        `sklearn.neural_network.MLPRegressor`.
    n_batch : int
        Number of likelihood evaluations that are performed at each step.
    vectorized : bool
        If True, the likelihood function can receive multiple input sets at
        once.
    pass_dict : bool
        If True, the likelihood function expects model parameters as
        dictionaries.
    pool : object
        Pool used to parallelize likelihood calls.
    n_jobs : int or string
        Number of parallel jobs to use for neural network training and sampling
        new points.
    rng : np.random.Generator
        Random number generator of the sampler.
    n_like : int
        Total number of likelihood evaluations.
    explored : bool, optional
        Whether the space has been explored and the shells have been
        constructed.
    bounds : list
        List of all the constructed bounds.
    points : list
        List of arrays where each array at position i lists the points
        belonging to the i-th bound/shell.
    log_l : list
        Log likelihood values of each point. Same ordering as `points`.
    blobs : list
        Blobs associated with each point. Same ordering as `points`.
    blobs_dtype : numpy.dtype
        Data type of the blobs.
    shell_n : numpy.ndarray
        Number of points for each bound/shell.
    shell_n_sample_shell : numpy.ndarray
        Number of points sampled in each bound that fall into the shell.
    shell_n_sample_bound : numpy.ndarray
        Number of points sampled in each bound.
    shell_n_eff : numpy.ndarray
        Effective sample size for each bound/shell.
    shell_log_l_min : numpy.ndarray
        Minimum logarithm of the likelihood required for an update in each
        bound/shell.
    shell_log_l : numpy.ndarray
        Logarithm of the mean likelihood of points in each bound/shell.
    shell_log_v : numpy.ndarray
        Logarithm of the volume of each bound/shell.

    """

    def __init__(self, prior, likelihood, n_dim=None, n_live=2000,
                 n_update=None, enlarge=None, enlarge_per_dim=1.1,
                 n_points_min=None, split_threshold=100,
                 n_networks=4, neural_network_kwargs=dict(), prior_args=[],
                 prior_kwargs=dict(), likelihood_args=[],
                 likelihood_kwargs=dict(), n_batch=100,
                 use_neural_networks=None, n_like_new_bound=None,
                 vectorized=False, pass_dict=None, pool=None, n_jobs=1,
                 random_state=None, seed=None,
                 blobs_dtype=None, filepath=None, resume=True):
        r"""
        Initialize the sampler.

        Parameters
        ----------
        prior : function or nautilus.Prior
            Prior describing the mapping of the unit hypercube to the
            parameters.
        likelihood : function
            Function returning the natural logarithm of the likelihood.
        n_dim : int, optional
            Number of dimensions of the likelihood function. If not specified,
            it will be inferred from the `prior` argument. But this requires
            `prior` to be an instance of `nautilus.Prior`.
        n_live : int, optional
            Number of so-called live points. New bounds are constructed so that
            they encompass the live points. Default is 3000.
        n_update : None or int, optional
            The maximum number of additions to the live set before a new bound
            is created. If None, use `n_live`. Default is None.
        enlarge : float, optional
            Deprecated.
        enlarge_per_dim : float, optional
            Along each dimension, outer ellipsoidal bounds are enlarged by this
            factor. Default is 1.1.
        n_points_min : int or None, optional
            The minimum number of points each ellipsoid should have.
            Effectively, ellipsoids with less than twice that number will not
            be split further. If None, uses `n_points_min = n_dim + 50`.
            Default is None.
        split_threshold: float, optional
            Threshold used for splitting the multi-ellipsoidal bound used for
            sampling. If the volume of the bound prior enlarging is larger than
            `split_threshold` times the target volume, the multi-ellipsiodal
            bound is split further, if possible. Default is 100.
        n_networks : int, optional
            Number of networks used in the estimator. Default is 4.
        neural_network_kwargs : dict, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor.
        prior_args : list, optional
            List of extra positional arguments for `prior`. Only used if
            `prior` is a function.
        prior_kwargs : dict, optional
            Dictionary of extra keyword arguments for `prior`. Only used if
            `prior` is a function.
        likelihood_args : list, optional
            List of extra positional arguments for `likelihood`.
        likelihood_kwargs : dict, optional
            Dictionary of extra keyword arguments for `likelihood`.
        n_batch : int, optional
            Number of likelihood evaluations that are performed at each step.
            If likelihood evaluations are parallelized, should be multiple
            of the number of parallel processes. Very large numbers can
            lead to new bounds being created long after `n_update` additions to
            the live set have been achieved. This will not cause any bias but
            could reduce efficiency. Default is 100.
        use_neural_networks : bool, optional
            Deprecated.
        n_like_new_bound : None or int, optional
            The maximum number of likelihood calls before a new bounds is
            created. If None, use 10 times `n_live`. Default is None.
        vectorized : bool, optional
            If True, the likelihood function can receive multiple input sets
            at once. For example, if the likelihood function receives arrays,
            it should be able to take an array with shape (n_points, n_dim)
            and return an array with shape (n_points). Similarly, if the
            likelihood function accepts dictionaries, it should be able to
            process dictionaries where each value is an array with shape
            (n_points). Default is False.
        pass_dict : bool or None, optional
            If True, the likelihood function expects model parameters as
            dictionaries. If False, it expects regular numpy arrays. Default is
            to set it to True if prior was a nautilus.Prior instance and False
            otherwise.
        pool : object or int, optional
            Object with a `map` function used for parallelization of likelihood
            calls, e.g. a multiprocessing.Pool object, or a positive integer.
            If it is an integer, it determines the number of workers in the
            Pool. Default is None.
        n_jobs : int or string, optional
            Number of parallel jobs to use for neural network training and
            sampling new points. If the string 'max' is passed, all available
            cores are used. Default is 'max'.
        random_state : int or np.random.RandomState, optional
            Deprecated.
        seed : int, optional
            Seed for random number generation used for reproducible results
            accross different runs. Default is None.
        blobs_dtype : object or None, optional
            Object that can be converted to a data type object describing the
            blobs. If None, this will be inferred from the first blob. Default
            is None.
        filepath : string, pathlib.Path or None, optional
            Path to the file where results are saved. Must have a '.h5' or
            '.hdf5' extension. If None, no results are written. Default is
            None.
        resume : bool, optional
            If True, resume from previous run if `filepath` exists. If False,
            start from scratch and overwrite any previous file. Default is
            True.

        Raises
        ------
        ValueError
            If `prior` is a function and `n_dim` is not given or `pass_struct`
            is set to True. If the dimensionality of the problem is less than
            2.

        """
        if callable(prior):
            self.prior = partial(prior, *prior_args, **prior_kwargs)
        else:
            self.prior = prior
        self.likelihood = partial(
            likelihood, *likelihood_args, **likelihood_kwargs)

        if callable(prior):
            if n_dim is None:
                raise ValueError("When passing a function as the 'prior' " +
                                 "argument, 'n_dim' cannot be None.")
            self.n_dim = n_dim
            if pass_dict is None:
                pass_dict = False
        else:
            self.n_dim = prior.dimensionality()
            if pass_dict is None:
                pass_dict = True

        if self.n_dim <= 1:
            raise ValueError(
                'Cannot run Nautilus with less than 2 parameters.')

        self.n_live = n_live

        if n_update is None:
            n_update = n_live
        self.n_update = n_update

        if n_like_new_bound is None:
            n_like_new_bound = 10 * n_live
        self.n_like_new_bound = n_like_new_bound

        if enlarge is not None:
            warnings.warn("The 'enlarge' keyword argument has been " +
                          "deprecated. Use 'enlarge_per_dim', instead.",
                          DeprecationWarning, stacklevel=2)

        self.enlarge_per_dim = enlarge_per_dim

        if n_points_min is None:
            n_points_min = self.n_dim + 50
        self.n_points_min = n_points_min

        self.split_threshold = split_threshold

        if use_neural_networks is not None:
            warnings.warn("The 'use_neural_networks' keyword argument has " +
                          "been deprecated. Set 'n_networks', instead.",
                          DeprecationWarning, stacklevel=2)

        self.n_networks = n_networks

        self.neural_network_kwargs = neural_network_kwargs
        self.n_batch = n_batch
        self.vectorized = vectorized
        self.pass_dict = pass_dict

        if isinstance(pool, int):
            self.pool = Pool(pool)
        elif pool is not None:
            self.pool = pool
        else:
            self.pool = None

        if n_jobs == 'max':
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

        if random_state is not None:
            warnings.warn("The 'random_state' keyword argument has been " +
                          "deprecated. Use 'seed' instead.",
                          DeprecationWarning, stacklevel=2)

        self.rng = np.random.default_rng(seed)

        # The following variables carry the information about the run.
        self.n_like = 0
        self.explored = False
        self.bounds = []
        self.points = []
        self.log_l = []
        self.blobs = []
        self.blobs_dtype = blobs_dtype
        self.shell_n = np.zeros(0, dtype=int)
        self.shell_n_sample_shell = np.zeros(0, dtype=int)
        self.shell_n_sample_bound = np.zeros(0, dtype=int)
        self.shell_n_eff = np.zeros(0, dtype=float)
        self.shell_log_l_min = np.zeros(0, dtype=float)
        self.shell_log_l = np.zeros(0, dtype=float)
        self.shell_log_v = np.zeros(0, dtype=float)

        self.filepath = filepath
        if resume and filepath is not None and Path(filepath).exists():
            with h5py.File(filepath, 'r') as fstream:

                group = fstream['sampler']

                self.rng.bit_generator.state = dict(
                    bit_generator='PCG64',
                    state=dict(
                        state=int(group.attrs['rng_state']),
                        inc=int(group.attrs['rng_inc'])),
                    has_uint32=group.attrs['rng_has_uint32'],
                    uinteger=group.attrs['rng_uinteger'])

                for key in ['n_like', 'explored', 'shell_n',
                            'shell_n_sample_shell', 'shell_n_sample_bound',
                            'shell_n_eff', 'shell_log_l_min', 'shell_log_l',
                            'shell_log_v']:
                    setattr(self, key, group.attrs[key])

                for shell in range(len(self.shell_n)):
                    self.points.append(
                        np.array(group['points_{}'.format(shell)]))
                    self.log_l.append(
                        np.array(group['log_l_{}'.format(shell)]))
                    try:
                        self.blobs.append(
                            np.array(group['blobs_{}'.format(shell)]))
                        self.blobs_dtype = self.blobs[-1].dtype
                    except KeyError:
                        pass

                self.bounds = [
                    UnitCube.read(fstream['bound_0'], rng=self.rng), ]
                for i in range(1, len(self.shell_n)):
                    self.bounds.append(NautilusBound.read(
                        fstream['bound_{}'.format(i)], rng=self.rng))

    def run(self, f_live=0.01, n_shell=None, n_eff=10000,
            discard_exploration=False, verbose=False):
        """Run the sampler until convergence.

        Parameters
        ----------
        f_live : float, optional
            Maximum fraction of the evidence contained in the live set before
            building the initial shells terminates. Default is 0.01.
        n_shell : int, optional
            Minimum number of points in each shell. The algorithm will sample
            from the shells until this is reached. Default is the batch size of
            the sampler which is 100 unless otherwise specified.
        n_eff : float, optional
            Minimum effective sample size. The algorithm will sample from the
            shells until this is reached. Default is 10000.
        discard_exploration : bool, optional
            Whether to discard points drawn in the exploration phase. This is
            required for a fully unbiased posterior and evidence estimate.
            Default is False.
        verbose : bool, optional
            If True, print additional information. Default is False.

        """
        self._pool = Pool(self.n_jobs) if self.n_jobs > 1 else None

        if not self.explored:

            if verbose:
                print('#########################')
                print('### Exploration Phase ###')
                print('#########################')
                print()

            while (self.live_evidence_fraction() > f_live or
                   len(self.bounds) == 0):
                self.add_bound(verbose=verbose)
                self.fill_bound(verbose=verbose)
                if self.filepath is not None:
                    self.write(self.filepath, overwrite=True)

            # If some shells are unoccupied in the end, remove them. They will
            # contain close to 0 volume and may never yield a point when
            # trying to sample from them.
            for shell in reversed(range(len(self.bounds))):
                if self.shell_n[shell] == 0:
                    self.bounds.pop(shell)
                    self.points.pop(shell)
                    self.log_l.pop(shell)
                    if self.blobs_dtype is not None:
                        self.blobs.pop(shell)
                    for key in ['shell_n', 'shell_n_sample_shell',
                                'shell_n_sample_bound', 'shell_n_eff',
                                'shell_log_l_min', 'shell_log_l',
                                'shell_log_v']:
                        setattr(self, key, np.delete(
                            getattr(self, key), shell))

            self.explored = True
            if self.filepath is not None:
                self.write(self.filepath, overwrite=True)

            if discard_exploration:
                self.discard_points()
                if self.filepath is not None:
                    # Rename the old checkpoint file containing points in the
                    # exploration phase and start a new one.
                    path = Path(self.filepath)
                    path.rename(Path(path.parent, path.stem + '_exp' +
                                     path.suffix))
                    self.write(self.filepath)

        if n_shell is None:
            n_shell = self.n_batch

        if (np.any(self.shell_n < n_shell) or
                self.effective_sample_size() < n_eff):

            if verbose:
                print('#########################')
                print('##### Sampling Phase ####')
                print('#########################')
                print()

            self.add_points(n_shell=n_shell, n_eff=n_eff, verbose=verbose)

        if self.n_jobs > 1:
            self._pool.close()

    def posterior(self, return_as_dict=None, equal_weight=False,
                  return_blobs=False):
        """Return the posterior sample estimate.

        Parameters
        ----------
        return_as_dict : bool or None, optional
            If True, return `points` as a dictionary. If None, will default to
            False unless one uses custom prior that only returns dictionaries.
            Default is None.
        equal_weight : bool, optional
            If True, return an equal weighted posterior. Default is False.
        return_blobs : bool, optional
            If True, return the blobs. Default is False.

        Returns
        -------
        points : numpy.ndarray or dict
            Points of the posterior.
        log_w : numpy.ndarray
            Weights of each point of the posterior.
        log_l : numpy.ndarray
            Logarithm of the likelihood at each point of the posterior.
        blobs : numpy.ndarray, optional
            Blobs for each point of the posterior. Only returned if
            `return_blobs` is True.

        Raises
        ------
        ValueError
            If `return_as_dict` or `return_blobs` are True but the sampler has
            not been run in a way that that's possible.

        """
        if return_as_dict is None:
            if callable(self.prior) and self.pass_dict:
                return_as_dict = True
            else:
                return_as_dict = False

        points = np.concatenate(self.points)
        log_v = np.repeat(self.shell_log_v -
                          np.log(np.maximum(self.shell_n, 1)), self.shell_n)
        log_l = np.concatenate(self.log_l)
        log_w = log_v + log_l
        if return_blobs:
            if self.blobs_dtype is None:
                raise ValueError('No blobs have been calculated.')
            blobs = np.concatenate(self.blobs)

        if callable(self.prior):
            transform = self.prior
        else:
            if return_as_dict:
                transform = self.prior.unit_to_dictionary
            else:
                transform = self.prior.unit_to_physical

        if not self.vectorized and callable(self.prior):
            points = np.array(list(map(transform, points)))
        else:
            points = transform(points)

        if not return_as_dict and callable(self.prior) and self.pass_dict:
            raise ValueError(
                'Cannot return points as numpy array. The prior function' +
                ' only returns dictionaries.')

        if equal_weight:
            select = (self.rng.random(len(log_w)) <
                      np.exp(log_w - np.amax(log_w)))
            points = points[select]
            log_w = np.ones(len(points)) * np.log(1.0 / np.sum(select))
            log_l = log_l[select]
            if return_blobs:
                blobs = blobs[select]

        # Normalize weights.
        log_w = log_w - logsumexp(log_w)

        if return_blobs:
            return points, log_w, log_l, blobs
        else:
            return points, log_w, log_l

    def effective_sample_size(self):
        r"""Estimate the total effective sample size :math:`N_{\rm eff}`.

        Returns
        -------
        n_eff : float
            Estimate of the total effective sample size :math:`N_{\rm eff}`.

        """
        select = self.shell_n_eff > 0
        sum_w = np.exp(self.shell_log_l + self.shell_log_v -
                       np.nanmax(self.shell_log_l + self.shell_log_v))[select]
        sum_w_sq = sum_w**2 / self.shell_n_eff[select]
        return np.sum(sum_w)**2 / np.sum(sum_w_sq)

    def evidence(self):
        r"""Estimate the global evidence :math:`\log \mathcal{Z}`.

        Returns
        -------
        log_z : float
            Estimate the global evidence :math:`\log \mathcal{Z}`.

        """
        select = ~np.isnan(self.shell_log_l)
        return logsumexp(self.shell_log_l[select] + self.shell_log_v[select])

    def sample_shell(self, index, shell_t=None):
        """Sample a batch of points uniformly from a shell.

        The shell at index :math:`i` is defined as the volume enclosed by the
        bound of index :math:`i` and enclosed by not other bound of index
        :math:`k` with :math:`k > i`.

        Parameters
        ----------
        index : int
            Index of the shell.
        shell_t : np.ndarray or None, optional
            If not None, an array of shell associations of possible transfer
            points.

        Returns
        -------
        points : numpy.ndarray
            Array of shape (n_shell, n_dim) containing points sampled uniformly
            from the shell.
        n_bound : int
            Number of points drawn within the bound at index :math:`i`. Will
            be different from `n_shell` if there are bounds with index
            :math:`k` with :math:`k > i`.
        idx_t : np.ndarray, optional
            Indeces of the transfer candidates that should be transferred. Only
            returned if `shell_t` is not None.

        """
        if shell_t is not None and index not in [-1, len(self.bounds) - 1]:
            raise ValueError("'shell_t' must be empty list if not sampling " +
                             "from the last bound/shell.")

        n_bound = 0
        n_sample = 0
        idx_t = np.zeros(0, dtype=int)
        points_all = []

        with threadpool_limits(limits=1):
            while n_sample < self.n_batch:
                points = self.bounds[index].sample(
                    self.n_batch - n_sample, pool=self._pool)
                n_bound += self.n_batch - n_sample

                # Remove points that are actually in another shell.
                in_shell = np.ones(len(points), dtype=bool)
                for bound in self.bounds[index:][1:]:
                    in_shell = in_shell & ~bound.contains(points)
                    if np.all(~in_shell):
                        continue
                points = points[in_shell]

                # Replace points for which we can use transfer points.
                replace = np.zeros(len(points), dtype=bool)
                if shell_t is not None and len(shell_t) > 0:
                    shell_p = self.shell_association(
                        points, n_max=len(self.bounds) - 1)
                    for shell in range(len(self.bounds) - 1):
                        idx_1 = np.flatnonzero(shell_t == shell)
                        idx_2 = np.flatnonzero(shell_p == shell)
                        n = min(len(idx_1), len(idx_2))
                        if n > 0:
                            idx_t = np.append(idx_t, self.rng.choice(
                                idx_1, size=n, replace=False))
                            shell_t[idx_t] = -1
                            replace[self.rng.choice(
                                idx_2, size=n, replace=False)] = True

                points = points[~replace]

                if len(points) > 0:
                    points_all.append(points)
                    n_sample += len(points)

        points = np.concatenate(points_all)

        if shell_t is None:
            return points, n_bound
        else:
            return points, n_bound, idx_t

    def evaluate_likelihood(self, points):
        """Evaluate the likelihood for a given set of points.

        Parameters
        ----------
        points : numpy.ndarray
            Points at which to evaluate the likelihood.

        Returns
        -------
        log_l : numpy.ndarray
            Natural log of the likelihood of each point.
        blobs : list, dict or None
            Blobs associated with the points, if any.

        Raises
        ------
        ValueError
            If `self.blobs_dtype` is not None but the likelihood function does
            not return blobs.

        """
        # Transform points from the unit cube to the input arguments of the
        # likelihood function.
        if callable(self.prior):
            transform = self.prior
        else:
            if self.pass_dict:
                transform = self.prior.unit_to_dictionary
            else:
                transform = self.prior.unit_to_physical

        if not self.vectorized:
            args = list(map(transform, points))
        else:
            args = transform(points)

        # Evaluate the likelihood.
        if self.vectorized:
            result = self.likelihood(args)
            if isinstance(result, tuple):
                result = list(zip(*result))
        elif self.pool is not None:
            result = list(self.pool.map(self.likelihood, args))
        else:
            result = list(map(self.likelihood, args))

        if isinstance(result[0], tuple):
            log_l = np.array([r[0] for r in result])
            blobs = [r[1:] for r in result]
            if self.blobs_dtype is None:
                if len(blobs[0]) > 1:
                    self.blobs_dtype = [
                        ('blob_{}'.format(i), np.array([b]).dtype) for i, b in
                        enumerate(blobs[0])]
                else:
                    self.blobs_dtype = np.array([blobs[0][0]]).dtype
            blobs = np.squeeze(np.array(blobs, dtype=self.blobs_dtype))
        else:
            log_l = np.array(result)
            blobs = None

        if blobs is None and self.blobs_dtype is not None:
            raise ValueError("'blobs_dtype' was specified but the likelihood" +
                             " function does not return blobs.")

        self.n_like += len(log_l)

        return log_l, blobs

    def update_shell_info(self, index):
        """Update the shell information for calculation of summary statistics.

        Parameters
        ----------
        index: int
            Index of the shell.

        """
        log_l = self.log_l[index]
        self.shell_n[index] = len(log_l)

        if self.shell_n[index] > 0:
            self.shell_log_v[index] = (
                self.bounds[index].volume() +
                np.log(self.shell_n_sample_shell[index] /
                       self.shell_n_sample_bound[index]))
            self.shell_log_l[index] = logsumexp(log_l) - np.log(len(log_l))
            if not np.all(log_l == -np.inf):
                self.shell_n_eff[index] = np.exp(2 * logsumexp(log_l) -
                                                 logsumexp(2 * log_l))
            else:
                self.shell_n_eff[index] = len(log_l)
        else:
            self.shell_log_v[index] = -np.inf
            self.shell_log_l[index] = np.nan
            self.shell_n_eff[index] = 0

    def print_status(self):
        """Print current summary statistics."""
        print('N_like: {:>17}'.format(self.n_like))
        print('N_eff: {:>18.0f}'.format(self.effective_sample_size()))
        print('log Z: {:>18.3f}'.format(self.evidence()))
        if not self.explored:
            print('log V: {:>18.3f}'.format(self.shell_log_v[-1]))
            print('f_live: {:>17.3f}'.format(self.live_evidence_fraction()))

    def add_bound(self, verbose=False):
        """Build a new bound from existing points.

        Parameters
        ----------
        verbose : bool, optional
            If True, print additional information. Default is False.

        """
        self.shell_n = np.append(self.shell_n, 0)
        self.shell_n_sample_shell = np.append(self.shell_n_sample_shell, 0)
        self.shell_n_sample_bound = np.append(self.shell_n_sample_bound, 0)
        self.shell_n_eff = np.append(self.shell_n_eff, 0)
        self.shell_log_l = np.append(self.shell_log_l, np.nan)
        self.shell_log_v = np.append(self.shell_log_v, np.nan)

        # If this is the first bound, use the UnitCube bound.
        if len(self.bounds) == 0:
            log_l_min = -np.inf
            self.bounds.append(UnitCube.compute(self.n_dim, rng=self.rng))
        else:
            if verbose:
                print('Adding bound {}'.format(len(self.bounds) + 1), end='\r')
            log_l = np.concatenate(self.log_l)
            points = np.concatenate(self.points)[np.argsort(log_l)]
            log_l = np.sort(log_l)
            log_l_min = 0.5 * (log_l[-self.n_live] + log_l[-self.n_live - 1])
            with threadpool_limits(limits=1):
                bound = NautilusBound.compute(
                    points, log_l, log_l_min, self.live_volume(),
                    enlarge_per_dim=self.enlarge_per_dim,
                    n_points_min=self.n_points_min,
                    split_threshold=self.split_threshold,
                    n_networks=self.n_networks,
                    neural_network_kwargs=self.neural_network_kwargs,
                    pool=self._pool, rng=self.rng)
                bound.sample(1000, return_points=False, pool=self._pool)
                if bound.volume() > self.bounds[-1].volume():
                    bound = self.bounds[-1]
            self.bounds.append(bound)

        self.points.append([])
        self.log_l.append([])
        self.blobs.append([])
        self.shell_log_l_min = np.append(self.shell_log_l_min, log_l_min)

        if verbose:
            print('Adding bound {:<7} done'.format(
                str(len(self.bounds)) + ':'))
            if isinstance(self.bounds[-1], NautilusBound):
                n_neural, n_sample =\
                    self.bounds[-1].number_of_networks_and_ellipsoids()
            else:
                n_neural, n_sample = 0, 0
            print("Neural nets: {:>12}".format(n_neural))
            print("Ellipsoids: {:>13}".format(n_sample))

    def fill_bound(self, verbose=False):
        """Fill a new bound with points until a new bound should be created.

        Parameters
        ----------
        verbose : bool, optional
            If True, print additional information. Default is False.

        """
        shell_t = []
        points_t = []
        log_l_t = []
        blobs_t = []

        # Check which points points from previous shells could be transferred
        # to the new bound.
        if len(self.bounds) > 1:
            for shell in range(len(self.bounds) - 1):

                in_bound = self.bounds[-1].contains(self.points[shell])
                shell_t.append(np.repeat(shell, np.sum(in_bound)))

                points_t.append(self.points[shell][in_bound])
                self.points[shell] = self.points[shell][~in_bound]

                log_l_t.append(self.log_l[shell][in_bound])
                self.log_l[shell] = self.log_l[shell][~in_bound]

                if self.blobs_dtype is not None:
                    blobs_t.append(self.blobs[shell][in_bound])
                    self.blobs[shell] = self.blobs[shell][~in_bound]

                self.shell_n[shell] -= np.sum(in_bound)
                self.shell_n_sample_shell[shell] -= np.sum(in_bound)
                self.update_shell_info(shell)

            shell_t = np.concatenate(shell_t)
            points_t = np.concatenate(points_t)
            log_l_t = np.concatenate(log_l_t)
            if self.blobs_dtype is not None:
                blobs_t = np.concatenate(blobs_t)

        log_l_min = self.shell_log_l_min[-1]
        n_update = 0
        n_like = 0
        n_update_max = self.n_update
        n_like_max = self.n_like_new_bound
        if len(self.bounds) == 1:
            n_update_max += self.n_live
            n_like_max = np.inf

        if verbose:
            pbar = tqdm(desc='Filling bound {}'.format(len(self.bounds)),
                        total=n_update_max, leave=False)

        while n_update < n_update_max and n_like < n_like_max:
            points, n_bound, idx_t = self.sample_shell(-1, shell_t)
            assert len(points) + len(idx_t) == n_bound
            log_l, blobs = self.evaluate_likelihood(points)
            self.points[-1].append(points)
            self.log_l[-1].append(log_l)
            if self.blobs_dtype is not None:
                self.blobs[-1].append(blobs)

            if len(idx_t) > 0:
                self.points[-1].append(points_t[idx_t])
                points_t = np.delete(points_t, idx_t, axis=0)
                self.log_l[-1].append(log_l_t[idx_t])
                log_l_t = np.delete(log_l_t, idx_t)
                if self.blobs_dtype is not None:
                    self.blobs[-1].append(blobs_t[idx_t])
                    blobs_t = np.delete(blobs_t, idx_t, axis=0)
                shell_t = np.delete(shell_t, idx_t)

            self.shell_n[-1] += n_bound
            self.shell_n_sample_shell[-1] += n_bound
            self.shell_n_sample_bound[-1] += n_bound
            n_update += np.sum(log_l >= log_l_min)
            n_like += len(points)

            if verbose:
                pbar.update(np.sum(log_l >= log_l_min))

        self.points[-1] = np.concatenate(self.points[-1])
        self.log_l[-1] = np.concatenate(self.log_l[-1])
        if self.blobs_dtype is not None:
            self.blobs[-1] = np.concatenate(self.blobs[-1])
        self.update_shell_info(-1)

        if verbose:
            pbar.close()
            print('Filling bound {:<6} done'.format(
                str(len(self.bounds)) + ':'))
            self.print_status()
            print('')

    def live_evidence_fraction(self):
        """Estimate the fraction of the evidence contained in the live set.

        This estimate can be used as a stopping criterion.

        Returns
        -------
        log_z : float
            Estimate of the fraction of the evidence in the live set.

        """
        if len(self.bounds) == 0:
            return 1.0
        else:
            log_v = np.repeat(
                self.shell_log_v - np.log(np.maximum(self.shell_n, 1)),
                self.shell_n)
            log_l = np.concatenate(self.log_l)
            log_w = log_v + log_l
            log_w_live = log_w[np.argsort(log_l)][-self.n_live:]
            return np.exp(logsumexp(log_w_live) - logsumexp(log_w))

    def live_volume(self):
        """Estimate the volume that is currently contained in the live set.

        Returns
        -------
        log_v : float
            Estimate of the volume in the live set.

        """
        if len(self.bounds) == 0:
            return 1.0
        else:
            log_l = np.concatenate(self.log_l)
            log_v = np.repeat(
                self.shell_log_v - np.log(np.maximum(self.shell_n, 1)),
                self.shell_n)
            log_v_live = log_v[np.argsort(log_l)][-self.n_live:]

            return logsumexp(log_v_live)

    def add_samples_to_shell(self, shell):
        """Add samples to a shell.

        The number of points added is always equal to the batch size.

        Parameters
        ----------
        shell : int
            The index of the shell for which to add points.

        """
        points, n_bound = self.sample_shell(shell)
        self.shell_n_sample_shell[shell] += len(points)
        self.shell_n_sample_bound[shell] += n_bound
        log_l, blobs = self.evaluate_likelihood(points)
        self.points[shell] = np.concatenate([self.points[shell], points])
        self.log_l[shell] = np.concatenate([self.log_l[shell], log_l])
        if self.blobs_dtype is not None:
            self.blobs[shell] = np.concatenate([self.blobs[shell], blobs])
        self.update_shell_info(shell)
        if self.filepath is not None:
            self.write_shell_update(self.filepath, shell)

    def discard_points(self):
        """Discard all points drawn."""
        for i in range(len(self.bounds)):
            self.shell_n_sample_shell[i] = 0
            self.shell_n_sample_bound[i] = 0
            self.points[i] = np.zeros((0, self.n_dim))
            self.log_l[i] = np.zeros(0)
            if self.blobs_dtype is not None:
                self.blobs[i] = self.blobs[i][:0]
            self.update_shell_info(i)

    def add_points(self, n_eff=0, n_shell=0, verbose=False):
        """Add samples to shells.

        This function add samples to the shell until very shell has a minimum
        number of points and a minimum effective sample size is achieved.

        Parameters
        ----------
        n_eff : float, optional
            Minimum effective sample size. The algorithm will sample from the
            shells until this is reached. Default is 0.
        n_shell : int, optional
            Minimum number of points in each shell. The algorithm will sample
            from the shells until this is reached. Default is 0.
        verbose : bool, optional
            If True, print additional information. Default is False.

        """
        idx = np.flatnonzero(self.shell_n < n_shell)
        if verbose and len(idx) > 0:
            pbar = tqdm(desc='Sampling shells ', total=len(idx),
                        leave=False)

        for index in idx:
            while self.shell_n[index] < n_shell:
                self.add_samples_to_shell(index)
            if verbose:
                pbar.update(1)

        if verbose and len(idx) > 0:
            pbar.close()
            print('Sampling shells:     done')
            self.print_status()
            print('')

        d_n_eff = n_eff - self.effective_sample_size()

        if verbose and d_n_eff > 0:
            pbar = tqdm(desc='Sampling posterior', total=n_eff,
                        leave=False, initial=self.effective_sample_size(),
                        bar_format="{l_bar}{bar}{n:.0f}/{total_fmt} " +
                        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        while self.effective_sample_size() < n_eff:
            index = np.argmax(
                self.shell_log_l + self.shell_log_v -
                0.5 * np.log(self.shell_n) - 0.5 * np.log(self.shell_n_eff))
            n_eff_old = self.effective_sample_size()
            self.add_samples_to_shell(index)
            n_eff_new = self.effective_sample_size()
            if verbose:
                pbar.update(n_eff_new - n_eff_old)

        if verbose and d_n_eff > 0:
            pbar.close()
            print('Sampling posterior:  done')
            self.print_status()
            print('')

    def shell_association(self, points, n_max=None):
        """Determine the shells each point belongs to.

        Parameters
        ----------
        points : numpy.ndarray
            points for which to determine shell association.
        n_max : int, optional
            The maximum number of shells to consider. Effectively, this
            determines the shell association at step `n_max` in the exploration
            phase. Default is to consider all shells.

        Returns
        -------
        shell: int
            Shell association for each point.

        """
        if n_max is None:
            n_max = len(self.bounds)

        shell = np.repeat(-1, len(points))
        for i, bound in reversed(list(enumerate(self.bounds[:n_max]))):
            mask = shell >= 0
            if np.all(mask):
                break
            mask[~mask] = ~bound.contains(points[~mask])
            shell[~mask] = i

        return shell

    def shell_bound_occupation(self, fractional=True):
        """Determine how many points of each shell are also part of each bound.

        Parameters
        ----------
        fractional : bool, optional
            Whether to return the absolute or fractional dependence. Default
            is True.

        Returns
        -------
        m : numpy.ndarray
            Two-dimensional array with occupation numbers. The element at index
            :math:`(i, j)` corresponds to the occupation of points in shell
            shell :math:`i` that also belong to bound :math:`j`. If
            `fractional` is True, this is the fraction of all points in shell
            :math:`i` and otherwise it is the absolute number.

        """
        m = np.zeros((len(self.bounds), len(self.bounds)), dtype=int)

        for i, points in enumerate(self.points):
            for k, bound in enumerate(self.bounds):
                m[i, k] = np.sum(bound.contains(points))

        if fractional:
            m = m / np.diag(m)[:, np.newaxis]

        return m

    def write(self, filepath, overwrite=False):
        """Write the sampler to disk.

        Parameters
        ----------
        filepath : string or pathlib.Path
            Path to the file. Must have a '.h5' or '.hdf5' extension.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.

        Raises
        ------
        ValueError
            If file extension is not '.h5' or '.hdf5'.
        RuntimeError
            If file exists and `overwrite` is False.

        """
        filepath = Path(filepath)

        if filepath.suffix not in ['.h5', '.hdf5']:
            raise ValueError("File ending must '.h5' or '.hdf5'.")

        if filepath.exists():
            if not overwrite:
                raise RuntimeError(
                    "File {} already exists.".format(str(filepath)))
            else:
                filepath.unlink()

        filepath.parent.mkdir(parents=True, exist_ok=True)

        fstream = h5py.File(filepath, 'x')
        group = fstream.create_group('sampler')

        for key in ['n_dim', 'n_live', 'n_update', 'n_like_new_bound',
                    'enlarge_per_dim', 'n_points_min', 'split_threshold',
                    'n_networks', 'n_batch', 'vectorized', 'pass_dict',
                    'n_like', 'explored', 'shell_n', 'shell_n_sample_shell',
                    'shell_n_sample_bound', 'shell_n_eff', 'shell_log_l_min',
                    'shell_log_l', 'shell_log_v']:
            group.attrs[key] = getattr(self, key)

        for key in self.neural_network_kwargs.keys():
            group.attrs['neural_network_{}'.format(key)] =\
                self.neural_network_kwargs[key]

        for shell in range(len(self.bounds)):
            group.create_dataset(
                'points_{}'.format(shell), data=self.points[shell],
                maxshape=(None, self.n_dim))
            group.create_dataset(
                'log_l_{}'.format(shell), data=self.log_l[shell],
                maxshape=(None, ))
            if self.blobs_dtype is not None:
                maxshape = list(self.blobs[shell].shape)
                maxshape[0] = None
                group.create_dataset(
                    'blobs_{}'.format(shell), data=self.blobs[shell],
                    maxshape=tuple(maxshape))

        for i, bound in enumerate(self.bounds):
            bound.write(fstream.create_group('bound_{}'.format(i)))

        rng_state = self.rng.bit_generator.state
        group.attrs['rng_state'] = str(rng_state['state']['state'])
        group.attrs['rng_inc'] = str(rng_state['state']['inc'])
        group.attrs['rng_has_uint32'] = rng_state['has_uint32']
        group.attrs['rng_uinteger'] = rng_state['uinteger']

        fstream.close()

    def write_shell_update(self, filepath, shell):
        """Update the sampler data for a single shell.

        Parameters
        ----------
        filepath : string or pathlib.Path
            Path to the file. Must have a '.h5' or '.hdf5' extension.
        shell : int
            Shell index for which to write the upate.

        Raises
        ------
        RuntimeError
            If file does not exist.

        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise RuntimeError(
                "File {} does not exist.".format(str(filepath)))

        fstream = h5py.File(filepath, 'r+')
        group = fstream['sampler']

        for key in ['n_like', 'shell_n', 'shell_n_sample_shell',
                    'shell_n_sample_bound', 'shell_n_eff', 'shell_log_l_min',
                    'shell_log_l', 'shell_log_v']:
            group.attrs[key] = getattr(self, key)

        for key in self.neural_network_kwargs.keys():
            group.attrs['neural_network_{}'.format(key)] =\
                self.neural_network_kwargs[key]

        group['points_{}'.format(shell)].resize(self.points[shell].shape)
        group['points_{}'.format(shell)][...] = self.points[shell]
        group['log_l_{}'.format(shell)].resize(self.log_l[shell].shape)
        group['log_l_{}'.format(shell)][...] = self.log_l[shell]
        if self.blobs_dtype is not None:
            group['blobs_{}'.format(shell)].resize(self.blobs[shell].shape)
            group['blobs_{}'.format(shell)][...] = self.blobs[shell]

        rng_state = self.rng.bit_generator.state
        group.attrs['rng_state'] = str(rng_state['state']['state'])
        group.attrs['rng_inc'] = str(rng_state['state']['inc'])
        group.attrs['rng_has_uint32'] = rng_state['has_uint32']
        group.attrs['rng_uinteger'] = rng_state['uinteger']

        fstream.close()

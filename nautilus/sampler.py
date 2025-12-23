"""Module implementing the Nautilus sampler."""

try:
    import h5py
except ImportError:
    pass
import numpy as np

from functools import partial
from pathlib import Path
from scipy.special import logsumexp
from shutil import get_terminal_size
from threadpoolctl import threadpool_limits
from time import time
from warnings import warn

from .bounds import UnitCube, NautilusBound
from .pool import likelihood_worker, NautilusPool


class Sampler():
    """A dynamic sampler built upon the framework of importance nested sampling.

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
    n_points_min : int or None
        The minimum number of points each ellipsoid should have. Effectively,
        ellipsoids with less than twice that number will not be split further.
    split_threshold: float
        Threshold used for splitting the multi-ellipsoidal bound used for
        sampling. If the volume of the bound prior enlarging is larger than
        `split_threshold` times the target volume, the multi-ellipsiodal
        bound is split further, if possible.
    periodic : numpy.ndarray or None
        Indices of the parameters that are periodic.
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
    pool_l : object
        Pool used to parallelize likelihood calls.
    pool_s : object
        Pool used to parallelize sampler calculations.
    rng : np.random.Generator
        Random number generator of the sampler.
    n_like : int
        Total number of likelihood evaluations.
    explored : bool
        Whether the space has been explored and the shells have been
        constructed.
    bounds : list
        List of all the constructed bounds.
    points : list
        List of arrays where each array at position i lists the points
        belonging to the i-th bound/shell.
    log_l : list
        Log likelihood values of each point. Same ordering as `points`.
    blobs : list or None
        Blobs associated with each point, if avaiable. Same ordering as
        `points`.
    blobs_dtype : numpy.dtype or None
        Data type of the blobs.
    _discard_exploration : bool
        Whether to exclude points in the exploration phase.
    shell_n : numpy.ndarray
        Number of points for each bound/shell.
    shell_n_sample : numpy.ndarray
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
    shell_n_sample_exp : numpy.ndarray
        Number of points sampled in each bound during the exploration phase.
    shell_end_exp : numpy.ndarray
        Index at which points are coming from the sampling phase instead of the
        exploration phase.
    points_t : numpy.ndarray
        Set of points that may be transferred to the latest bound if
        exploration stage isn't finished, yet.
    shell_t : numpy.ndarray
        Shell indices from which the transfer points came from.
    log_l_t : numpy.ndarray
        Likelihood values of the points that may be transferred.
    blobs_t : numpy.ndarray
        Blobs of the points that may be transferred.

    """

    def __init__(self, prior, likelihood, n_dim=None, n_live=2000,
                 n_update=None, enlarge_per_dim=1.1, n_points_min=None,
                 split_threshold=100, periodic=None, n_networks=4,
                 neural_network_kwargs=dict(), prior_args=[],
                 prior_kwargs=dict(), likelihood_args=[],
                 likelihood_kwargs=dict(), n_batch=None,
                 n_like_new_bound=None, vectorized=False, pass_dict=None,
                 pool=None, seed=None, blobs_dtype=None, filepath=None,
                 resume=True):
        r"""Initialize the sampler.

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
        periodic : numpy.ndarray or None, optional
            Indices of the parameters that are periodic. Default is None.
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
        n_batch : int or None, optional
            Number of likelihood evaluations that are performed at each step.
            If likelihood evaluations are parallelized, should be multiple
            of the number of parallel processes. If None, will be the smallest
            multiple of the pool size used for likelihood calls that is at
            least 100. Default is None.
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
        pool : None, object, int or tuple, optional
            Pool used for parallelization of likelihood calls and sampler
            calculations. If None, no parallelization is performed. If an
            integer, the sampler will use a multiprocessing.Pool object with
            the specified number of processes. Finally, if specifying a tuple,
            the first one specifies the pool used for likelihood calls and the
            second one the pool for sampler calculations. Supported pools
            include instances of `multiprocessing.Pool` and
            `dask.distributed.client.Client`. Default is None.
        seed : None or int, optional
            Seed for random number generation used for reproducible results
            accross different runs. If None, results are not reproducible.
            Default is None.
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

        self.enlarge_per_dim = enlarge_per_dim

        if n_points_min is None:
            n_points_min = self.n_dim + 50
        self.n_points_min = n_points_min

        self.split_threshold = split_threshold
        self.periodic = periodic

        self.n_networks = n_networks

        self.neural_network_kwargs = neural_network_kwargs
        self.vectorized = vectorized
        self.pass_dict = pass_dict

        try:
            pool = list(pool)
        except TypeError:
            pool = [pool]

        for i in range(len(pool)):
            if pool[i] in [None, 1]:
                pool[i] = None
            elif i == 0 and isinstance(pool[i], int):
                pool[i] = NautilusPool(pool[i], likelihood=self.likelihood)
                self.likelihood = likelihood_worker
            else:
                pool[i] = NautilusPool(pool[i])

        self.pool_l = pool[0]
        self.pool_s = pool[-1]

        if n_batch is None:
            s = 1 if self.pool_l is None else self.pool_l.size
            n_batch = (100 // s + (100 % s != 0)) * s
        self.n_batch = n_batch

        self.rng = np.random.default_rng(seed)

        # The following variables carry the information about the run.
        self.n_like = 0
        self.explored = False
        self.bounds = []
        self.points = []
        self.log_l = []
        self.blobs = None
        self.blobs_dtype = blobs_dtype
        self._discard_exploration = False
        self.shell_n = np.zeros(0, dtype=int)
        self.shell_n_sample = np.zeros(0, dtype=int)
        self.shell_n_eff = np.zeros(0, dtype=float)
        self.shell_log_l_min = np.zeros(0, dtype=float)
        self.shell_log_l = np.zeros(0, dtype=float)
        self.shell_log_v = np.zeros(0, dtype=float)
        self.shell_n_sample_exp = np.zeros(0, dtype=int)
        self.shell_end_exp = np.zeros(0, dtype=int)
        self.points_t = np.zeros((0, self.n_dim))
        self.shell_t = np.zeros(0, dtype=int)
        self.log_l_t = np.zeros(0)
        self.blobs_t = None

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

                for key in ['n_like', 'explored', '_discard_exploration',
                            'shell_n', 'shell_n_sample', 'shell_n_eff',
                            'shell_log_l_min', 'shell_log_l', 'shell_log_v',
                            'shell_n_sample_exp', 'shell_end_exp',
                            'n_update_iter', 'n_like_iter']:
                    setattr(self, key, group.attrs[key])

                for shell in range(len(self.shell_n)):
                    self.points.append(
                        np.array(group['points_{}'.format(shell)]))
                    self.log_l.append(
                        np.array(group['log_l_{}'.format(shell)]))
                    if 'blobs_{}'.format(shell) in group:
                        if shell == 0:
                            self.blobs = []
                        self.blobs.append(
                            np.array(group['blobs_{}'.format(shell)]))
                        if shell == 0:
                            self.blobs_dtype = self.blobs[-1].dtype

                for key in ['shell_t', 'points_t', 'log_l_t', 'blobs_t']:
                    if key in group:
                        setattr(self, key, np.array(group[key]))

                self.bounds = [
                    UnitCube.read(fstream['bound_0'], rng=self.rng), ]
                for i in range(1, len(self.shell_n)):
                    self.bounds.append(NautilusBound.read(
                        fstream['bound_{}'.format(i)], rng=self.rng))

    def run(self, f_live=0.01, n_shell=1, n_eff=10000, n_like_max=np.inf,
            discard_exploration=False, timeout=np.inf, verbose=False):
        """Run the sampler until convergence.

        Parameters
        ----------
        f_live : float, optional
            Maximum fraction of the evidence contained in the live set before
            building the initial shells terminates. Default is 0.01.
        n_shell : int, optional
            Minimum number of points in each shell. The algorithm will sample
            from the shells until this is reached. Default is 1.
        n_eff : float, optional
            Minimum effective sample size. The algorithm will sample from the
            shells until this is reached. Default is 10000.
        n_like_max : int, optional
            Maximum total (accross multiple runs) number of likelihood
            evaluations. Regardless of progress, the sampler will not start new
            likelihood computations if this value is reached. Note that this
            value includes likelihood calls from previous runs, if applicable.
            Default is infinity.
        discard_exploration : bool, optional
            Whether to discard points drawn in the exploration phase. This is
            required for a fully unbiased posterior and evidence estimate.
            Default is False.
        timeout : float, optional
            Timeout interval in seconds. The sampler will not start new
            likelihood computations if this limit is reached. Unlike for
            `n_like_max`, this maximum only refers to the current function
            call. Default is infinity.
        verbose : bool, optional
            If True, print information about sampler progress. Default is
            False.

        Returns
        -------
        success : bool
            Whether the run finished successfully without stopping prematurely.
            False if the run finished because the `n_like_max` or `timeout`
            limits were reached and True otherwise.

        """
        t_start = time()

        if verbose:
            if self.n_like == 0:
                print('Starting the nautilus sampler...')
            else:
                print('Resuming nautilus run...')
            print('Please report issues at github.com/johannesulf/nautilus.')
            self.print_status(header=True)

        if len(self.bounds) == 0:
            self.add_bound()
            self.n_update_iter = -self.n_live
            self.n_like_iter = 0

        success = (self.explored and np.all(self.shell_n >= n_shell) and
                   self.n_eff >= n_eff)

        while ((self.n_like < n_like_max) and (time() - t_start < timeout) and
               not success):

            if not self.explored:

                if ((self.n_update_iter >= self.n_update or
                     self.n_like_iter >= self.n_like_new_bound) and
                        np.sum(self.shell_n) > self.n_live):
                    self.add_bound(verbose=verbose)
                    self.n_update_iter = 0
                    self.n_like_iter = 0
                    if self.filepath is not None:
                        self.write(self.filepath, overwrite=True)

                self.n_update_iter += self.add_samples(-1, verbose=verbose)
                self.n_like_iter += self.n_batch
                if self.filepath is not None:
                    # Write the complete file if this is the first batch.
                    if self.n_like == self.n_batch:
                        self.write(self.filepath, overwrite=True)
                    self.write_shell_update(self.filepath, -1)

                if self.f_live <= f_live:

                    # If some shells are unoccupied in the end, remove them.
                    # They will contain close to 0 volume and may never yield a
                    # point when trying to sample from them.
                    if np.any(self.shell_n == 0):
                        for shell in np.flatnonzero(self.shell_n == 0)[::-1]:
                            self.bounds.pop(shell)
                            self.points.pop(shell)
                            self.log_l.pop(shell)
                            if self.blobs is not None:
                                self.blobs.pop(shell)
                            for key in ['shell_n', 'shell_n_sample',
                                        'shell_n_eff', 'shell_log_l_min',
                                        'shell_log_l', 'shell_log_v']:
                                setattr(self, key, np.delete(
                                    getattr(self, key), shell))

                    self.shell_n_sample_exp = np.copy(self.shell_n_sample)
                    self.shell_end_exp = np.array(
                        [len(p) for p in self.points])

                    self.explored = True
                    self.discard_exploration = discard_exploration
                    if self.filepath is not None:
                        self.write(self.filepath, overwrite=True)

            elif np.any(self.shell_n < n_shell):
                shell = np.flatnonzero(self.shell_n < n_shell)[0]
                self.add_samples(shell, verbose=verbose)
                if self.filepath is not None:
                    self.write_shell_update(self.filepath, shell)

            elif self.n_eff < n_eff:
                shell = np.argmax(self.shell_log_l + self.shell_log_v -
                                  0.5 * np.log(self.shell_n) -
                                  0.5 * np.log(self.shell_n_eff))
                self.add_samples(shell, verbose=verbose)
                if self.filepath is not None:
                    self.write_shell_update(self.filepath, shell)

            success = (self.explored and np.all(self.shell_n >= n_shell) and
                       self.n_eff >= n_eff)

        if verbose:
            if success:
                self.print_status('Finished')
            else:
                self.print_status('Stopped')

        return success

    @property
    def discard_exploration(self):
        """Return whether the exploration phase is discarded.

        Returns
        -------
        discard_exploration : bool
            Whether the exploration phase is discarded.

        """
        return self._discard_exploration

    @discard_exploration.setter
    def discard_exploration(self, discard_exploration):
        """Set whether exploration phase should be discarded.

        Parameters
        ----------
        discard_exploration : bool
            Whether the exploration phase is discarded.

        Raises
        ------
        ValueError
            If `discard_exploration` is not a bool.

        """
        if not isinstance(discard_exploration, bool):
            raise ValueError("'discard_exploration' must be a bool.")

        self._discard_exploration = discard_exploration
        for index in range(len(self.log_l)):
            self.update_shell_info(index)

    def posterior(self, return_as_dict=None, equal_weight=False,
                  equal_weight_boost=1.0, return_blobs=False):
        """Return the posterior sample estimate.

        Parameters
        ----------
        return_as_dict : bool or None, optional
            If True, return `points` as a dictionary. If None, will default to
            False unless one uses custom prior that only returns dictionaries.
            Default is None.
        equal_weight : bool, optional
            If True, return an equal-weighted posterior. This is done by
            randomly sampling each point from the unequal-weighted posterior
            proportional to its weight. Note that this effectively downgrades
            the posterior. For high-precision estimates of the posterior,
            use the unequal-weighted posterior. Default is False.
        equal_weight_boost : float, optional
            For the equal-weighted posterior, each point is sampled n times,
            where n is drawn from a nearest-integer distribution with
            mean value w / max(w) * `equal_weight_boost`. Here, max(w) is the
            maximum weight across all points in the posterior. For
            `equal_weight_boost`=1, this means that each point is at most
            sampled once, i.e., the posterior estimate contains no
            duplicates. For `equal_weight_boost` > 1, duplicates are possible
            but the equal-weighted posterior is a better approximation to the
            unequal-weight posterior. Note that the number of points returned
            is, on average, proportional to `equal_weight_boost`. Default is
            1.0.
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

        if self._discard_exploration and self.explored:
            start = self.shell_end_exp
        else:
            start = np.zeros(len(self.points), dtype=int)

        points = np.concatenate([p[s:] for p, s in zip(self.points, start)])
        log_v = np.repeat(self.shell_log_v -
                          np.log(np.maximum(self.shell_n, 1)), self.shell_n)
        log_l = np.concatenate([ll[s:] for ll, s in zip(self.log_l, start)])
        log_w = log_v + log_l
        if return_blobs:
            if self.blobs is None:
                raise ValueError('No blobs have been calculated.')
            blobs = np.concatenate([b[s:] for b, s in zip(self.blobs, start)])

        if equal_weight:
            repeats = np.exp(log_w - np.amax(log_w)) * equal_weight_boost
            repeats = np.floor(repeats).astype(int) + (
                self.rng.random(len(repeats)) < repeats - np.floor(repeats)
            ).astype(int)
            points = np.repeat(points, repeats, axis=0)
            log_w = np.zeros(np.sum(repeats))
            log_l = np.repeat(log_l, repeats, axis=0)
            if return_blobs:
                blobs = np.repeat(blobs, repeats, axis=0)

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

        # Normalize weights.
        log_w = log_w - logsumexp(log_w)

        if return_blobs:
            return points, log_w, log_l, blobs
        else:
            return points, log_w, log_l

    @property
    def n_eff(self):
        r"""Estimate the total effective sample size :math:`N_{\rm eff}`.

        Returns
        -------
        n_eff : float
            Estimate of the total effective sample size :math:`N_{\rm eff}`.

        """
        if np.all(self.shell_n_eff == 0):
            return 0
        select = self.shell_n_eff > 0
        sum_w = np.exp(self.shell_log_l + self.shell_log_v -
                       np.nanmax(self.shell_log_l + self.shell_log_v))[select]
        sum_w_sq = sum_w**2 / self.shell_n_eff[select]
        return np.sum(sum_w)**2 / np.sum(sum_w_sq)

    def effective_sample_size(self):
        r"""Estimate the total effective sample size :math:`N_{\rm eff}`.

        Returns
        -------
        n_eff : float
            Estimate of the total effective sample size :math:`N_{\rm eff}`.

        """
        warn("The function 'effective_sample_size' is deprecated. " +
             "Please use the property 'n_eff', instead.",
             DeprecationWarning, stacklevel=2)
        return self.n_eff

    @property
    def log_z(self):
        r"""Estimate the global evidence :math:`\log \mathcal{Z}`.

        Returns
        -------
        log_z : float or None
            Estimate of the global evidence :math:`\log \mathcal{Z}`.

        """
        if np.sum(self.shell_n) == 0:
            return None
        select = ~np.isnan(self.shell_log_l)
        return logsumexp(self.shell_log_l[select] + self.shell_log_v[select])

    def evidence(self):
        r"""Estimate the global evidence :math:`\log \mathcal{Z}`.

        Returns
        -------
        log_z : float
            Estimate of the global evidence :math:`\log \mathcal{Z}`.

        """
        warn("The function 'evidence' is deprecated. Please use the " +
             "property 'log_z', instead.", DeprecationWarning, stacklevel=2)
        return self.log_z

    @property
    def eta(self):
        r"""Estimate the asymptotic sampling efficiency :math:`\eta`.

        The asymptotic sampling efficiency is defined as
        :math:`\eta = \lim_{N_{\rm like} \to \infty} N_{\rm eff} / N_{\rm like}`.
        This is set after the exploration phase. However, the estimate will be
        updated based on what is found in the sampling phase.

        Returns
        -------
        eta : float
            Estimate of the asymptotic sampling efficiency.

        """
        shell_log_z = self.shell_log_l + self.shell_log_v
        shell_eta = self.shell_n_eff / self.shell_n
        select = ~np.isnan(self.shell_log_l)
        shell_log_z = shell_log_z[select]
        shell_eta = shell_eta[select]
        return np.exp(2 * logsumexp(shell_log_z) - 2 * logsumexp(
            shell_log_z - 0.5 * np.log(shell_eta)))

    def asymptotic_sampling_efficiency(self):
        r"""Estimate the asymptotic sampling efficiency :math:`\eta`.

        The asymptotic sampling efficiency is defined as
        :math:`\eta = \lim_{N_{\rm like} \to \infty} N_{\rm eff} / N_{\rm like}`.
        This is set after the exploration phase. However, the estimate will be
        updated based on what is found in the sampling phase.

        Returns
        -------
        eta : float
            Estimate of the asymptotic sampling efficiency.

        """
        warn("The function 'asymptotic_sampling_efficiency' is deprecated. " +
             "Please use the property 'log_z', instead.", DeprecationWarning,
             stacklevel=2)
        return self.eta

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
                    self.n_batch - n_sample, pool=self.pool_s)
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
        elif self.pass_dict:
            transform = self.prior.unit_to_dictionary
        else:
            transform = self.prior.unit_to_physical

        if not self.vectorized:
            args = list(map(transform, np.copy(points)))
        else:
            args = list(map(transform, np.array_split(
                points, 1 if self.pool_l is None else self.pool_l.size)))

        # Evaluate the likelihood.
        if self.pool_l is not None:
            result = list(self.pool_l.map(self.likelihood, args))
        else:
            result = list(map(self.likelihood, args))

        # Extracts the blobs, if present.
        if isinstance(result[0], tuple):
            log_l = [result[i][0] for i in range(len(result))]
            blobs = [result[i][1:] for i in range(len(result))]
        else:
            log_l = result
            blobs = None

        if self.vectorized:
            log_l = np.concatenate(log_l)
        else:
            log_l = np.array(log_l)

        if blobs is not None:
            if self.vectorized:
                blobs = [np.concatenate([
                    blobs[row][col] for row in range(len(blobs))])
                    for col in range(len(blobs[0]))]
            else:
                blobs = [np.array([
                    blobs[row][col] for row in range(len(blobs))])
                    for col in range(len(blobs[0]))]
            if self.blobs_dtype is None:
                if len(blobs) > 1:
                    self.blobs_dtype = [('blob_{}'.format(i), b.dtype) for
                                        i, b in enumerate(blobs)]
                else:
                    self.blobs_dtype = blobs[0].dtype
            blobs = np.squeeze(
                np.array(list(zip(*blobs)), dtype=self.blobs_dtype))

        self.n_like += len(log_l)

        return log_l, blobs

    def update_shell_info(self, index):
        """Update the shell information for calculation of summary statistics.

        Parameters
        ----------
        index: int
            Index of the shell.

        """
        shell_n_sample = self.shell_n_sample[index]

        if self._discard_exploration and self.explored:
            start = self.shell_end_exp[index]
            shell_n_sample -= self.shell_n_sample_exp[index]
        else:
            start = 0

        log_l = self.log_l[index][start:]
        shell_n = len(log_l)
        self.shell_n[index] = shell_n

        if self.shell_n[index] > 0:
            self.shell_log_v[index] = (self.bounds[index].log_v +
                                       np.log(shell_n / shell_n_sample))
            self.shell_log_l[index] = logsumexp(log_l) - np.log(shell_n)
            if not np.all(log_l == -np.inf):
                self.shell_n_eff[index] = np.exp(2 * logsumexp(log_l) -
                                                 logsumexp(2 * log_l))
            else:
                self.shell_n_eff[index] = len(log_l)
        else:
            self.shell_log_v[index] = -np.inf
            self.shell_log_l[index] = np.nan
            self.shell_n_eff[index] = 0

    def print_status(self, status='', header=False, end='\n'):
        """Print current summary statistics.

        Parameters
        ----------
        status: string, optional
            Status of the sampler to be printed. Default is ''.
        header : bool, optional
            If True, print a static header. Default is False.
        end : str, optional
            String printed at the end. Default is newline.

        """
        if header:
            data = ['Status', 'Bounds', 'Ellipses', 'Networks', 'Calls',
                    'f_live', 'N_eff', 'log Z']
        else:
            data = [status, len(self.bounds)]
            if len(self.bounds) > 1:
                data.extend([self.bounds[-1].n_ell, self.bounds[-1].n_net])
            else:
                data.extend([0, 0])
            data.extend([self.n_like, self.f_live, self.n_eff, self.log_z])

            fmt = ['{}', '{:d}', '{:d}', '{:d}', '{:d}', '{:.4f}', '{:.0f}',
                   '{:+.2f}']
            for i in range(len(data)):
                data[i] = 'N/A' if data[i] is None else fmt[i].format(data[i])

        for i, length in enumerate([9, 6, 8, 8, 8, 6, 5, 7]):
            data[i] = '{:<{}}'.format(data[i], length)

        output = ' | '.join(data)
        width = get_terminal_size((80, 24)).columns
        output = output.ljust(width)[:width]
        print(output, end=end, flush=True)

    def add_bound(self, verbose=False):
        """Try building a new bound from existing points.

        If the new bound would be larger than the previous bound, reject the
        new bound.

        Parameters
        ----------
        verbose : bool, optional
            If True, print additional information. Default is False.

        Returns
        -------
        success : boolean
            Whether a new bound has been added.

        """
        # If this is the first bound, use the UnitCube bound.
        if len(self.bounds) == 0:
            log_l_min = -np.inf
            self.bounds.append(UnitCube.compute(self.n_dim, rng=self.rng))
            success = True
        else:
            if verbose:
                self.print_status('Bounding', end='\r')
            log_l = np.concatenate(self.log_l)
            points = np.concatenate(self.points)[np.argsort(log_l)]
            log_l = np.sort(log_l)
            log_l_min = log_l[-self.n_live]

            # If log_l_min is part of a likelihood plateau and there exist
            # enough points above the plateau, skip the plateau.
            if (np.sum(log_l == log_l_min) > 1 and
                    np.sum(log_l > log_l_min) >= self.n_points_min):
                log_l_min = np.amin(log_l[log_l > log_l_min])

            # If there are no points below the plateau, don't zoom in.
            if np.all(log_l >= log_l_min):
                success = False
            else:
                with threadpool_limits(limits=1):
                    bound = NautilusBound.compute(
                        points, log_l, log_l_min, self.log_v_live,
                        enlarge_per_dim=self.enlarge_per_dim,
                        n_points_min=self.n_points_min,
                        split_threshold=self.split_threshold,
                        periodic=self.periodic,
                        n_networks=self.n_networks,
                        neural_network_kwargs=self.neural_network_kwargs,
                        pool=self.pool_s, rng=self.rng)
                    bound.sample(1000, return_points=False, pool=self.pool_s)

                # Only accept a new bound if it's smaller.
                if bound.log_v < self.bounds[-1].log_v:
                    self.bounds.append(bound)
                    success = True
                else:
                    success = False

        if success:
            self.shell_n = np.append(self.shell_n, 0)
            self.shell_n_sample = np.append(self.shell_n_sample, 0)
            self.shell_n_eff = np.append(self.shell_n_eff, 0)
            self.shell_log_l = np.append(self.shell_log_l, np.nan)
            self.shell_log_v = np.append(self.shell_log_v, np.nan)
            self.shell_log_l_min = np.append(self.shell_log_l_min, log_l_min)
            self.points.append(np.zeros((0, self.n_dim)))
            self.log_l.append(np.zeros(0))
            if self.blobs is not None:
                self.blobs.append(
                    np.zeros(self.blobs[-1][:0].shape, dtype=self.blobs_dtype))
        else:
            self.shell_log_l_min[-1] = log_l_min
            return False

        # Check which points points from previous shells could be transferred
        # to the new bound.
        if len(self.bounds) > 1:

            self.shell_t = []
            self.points_t = []
            self.log_l_t = []
            if self.blobs is not None:
                self.blobs_t = []

            for shell in range(len(self.bounds) - 1):

                in_bound = self.bounds[-1].contains(self.points[shell])
                self.shell_t.append(np.repeat(shell, np.sum(in_bound)))

                self.points_t.append(self.points[shell][in_bound])
                self.points[shell] = self.points[shell][~in_bound]

                self.log_l_t.append(self.log_l[shell][in_bound])
                self.log_l[shell] = self.log_l[shell][~in_bound]

                if self.blobs is not None:
                    self.blobs_t.append(self.blobs[shell][in_bound])
                    self.blobs[shell] = self.blobs[shell][~in_bound]

                self.shell_n[shell] -= np.sum(in_bound)
                self.update_shell_info(shell)

            self.shell_t = np.concatenate(self.shell_t)
            self.points_t = np.concatenate(self.points_t)
            self.log_l_t = np.concatenate(self.log_l_t)
            if self.blobs is not None:
                self.blobs_t = np.concatenate(self.blobs_t)

        return True

    def add_samples(self, shell, verbose=False):
        """Add samples to a shell.

        The number of new points added is always equal to the batch size.

        Parameters
        ----------
        shell : int
            The index of the shell for which to add points.
        verbose : bool, optional
            If True, print additional information. Default is False.

        Returns
        -------
        n_update : int
            Number of new samples with likelihood equal or higher than the
            likelihood threshold of the bound.

        """
        if verbose:
            self.print_status('Sampling', end='\r')

        if shell == -1 and len(self.shell_t) > 0:
            points, n_bound, idx_t = self.sample_shell(-1, self.shell_t)
            assert len(points) + len(idx_t) == n_bound
            if verbose:
                self.print_status('Computing', end='\r')
            if len(idx_t) > 0:
                self.points[-1] = np.concatenate((
                    self.points[-1], self.points_t[idx_t]))
                self.log_l[-1] = np.concatenate((
                    self.log_l[-1], self.log_l_t[idx_t]))
                if self.blobs is not None:
                    self.blobs[-1] = np.concatenate((
                        self.blobs[-1], self.blobs_t[idx_t]))
        else:
            points, n_bound = self.sample_shell(shell)
            if verbose:
                self.print_status('Computing', end='\r')

        self.shell_n_sample[shell] += n_bound
        log_l, blobs = self.evaluate_likelihood(points)
        self.points[shell] = np.append(self.points[shell], points, axis=0)
        self.log_l[shell] = np.append(self.log_l[shell], log_l, axis=0)
        if blobs is not None:
            if self.blobs is None:
                self.blobs = [blobs]
            else:
                self.blobs[shell] = np.append(self.blobs[shell], blobs, axis=0)
        self.update_shell_info(shell)

        return np.sum(log_l >= self.shell_log_l_min[shell])

    @property
    def f_live(self):
        """Estimate the fraction of the evidence contained in the live set.

        This estimate can be used as a stopping criterion.

        Returns
        -------
        f_live : float
            Estimate of the fraction of the evidence in the live set.

        """
        if self.explored:
            return None
        elif np.sum(self.shell_n) == 0:
            return 1.0
        else:
            log_v = np.repeat(
                self.shell_log_v - np.log(np.maximum(self.shell_n, 1)),
                self.shell_n)
            log_l = np.concatenate(self.log_l)
            log_w = log_v + log_l
            log_w_live = log_w[np.argsort(log_l)][-self.n_live:]
            return np.exp(logsumexp(log_w_live) - logsumexp(log_w))

    @property
    def log_v_live(self):
        """Estimate the volume that is currently contained in the live set.

        Returns
        -------
        log_v_live : float
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
                    'n_like', 'explored', '_discard_exploration', 'shell_n',
                    'shell_n_sample', 'shell_n_eff', 'shell_log_l_min',
                    'shell_log_l', 'shell_log_v', 'shell_n_sample_exp',
                    'shell_end_exp', 'n_update_iter', 'n_like_iter']:
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
            if self.blobs is not None:
                maxshape = list(self.blobs[shell].shape)
                maxshape[0] = None
                group.create_dataset(
                    'blobs_{}'.format(shell), data=self.blobs[shell],
                    maxshape=tuple(maxshape))

        group.create_dataset('points_t', data=self.points_t,
                             maxshape=(None, self.n_dim))
        group.create_dataset('shell_t', data=self.shell_t, maxshape=(None, ))
        group.create_dataset('log_l_t', data=self.log_l_t, maxshape=(None, ))
        if self.blobs_t is not None:
            group.create_dataset('blobs_t', data=self.blobs_t,
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

        """
        if shell < 0:
            shell = len(self.bounds) + shell
        fstream = h5py.File(Path(filepath), 'r+')
        group = fstream['sampler']

        for key in ['n_like', 'shell_n', 'shell_n_sample', 'shell_n_eff',
                    'shell_log_l_min', 'shell_log_l', 'shell_log_v',
                    'n_update_iter', 'n_like_iter']:
            group.attrs[key] = getattr(self, key)

        group['points_{}'.format(shell)].resize(self.points[shell].shape)
        group['points_{}'.format(shell)][...] = self.points[shell]
        group['log_l_{}'.format(shell)].resize(self.log_l[shell].shape)
        group['log_l_{}'.format(shell)][...] = self.log_l[shell]
        if self.blobs is not None:
            group['blobs_{}'.format(shell)].resize(self.blobs[shell].shape)
            group['blobs_{}'.format(shell)][...] = self.blobs[shell]

        for key in ['points_t', 'shell_t', 'log_l_t', 'blobs_t']:
            if getattr(self, key) is not None:
                group[key].resize(getattr(self, key).shape)
                group[key][...] = getattr(self, key)

        if isinstance(self.bounds[shell], NautilusBound):
            self.bounds[shell].update(fstream['bound_{}'.format(shell)])

        rng_state = self.rng.bit_generator.state
        group.attrs['rng_state'] = str(rng_state['state']['state'])
        group.attrs['rng_inc'] = str(rng_state['state']['inc'])
        group.attrs['rng_has_uint32'] = rng_state['has_uint32']
        group.attrs['rng_uinteger'] = rng_state['uinteger']

        fstream.close()

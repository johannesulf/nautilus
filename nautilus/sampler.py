"""Module implementing the Nautilus sampler."""

import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from scipy.special import logsumexp

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
    enlarge : float
        Factor by which the volume of ellipsoidal bounds is increased.
    use_neural_networks : bool
        Whether to use neural network emulators in the construction of the
        bounds.
    neural_network_kwargs : dict
        Keyword arguments passed to the constructor of
        `sklearn.neural_network.MLPRegressor`.
    n_batch : int
        Number of likelihood evaluations that are performed at each step.
    vectorized : bool
        If true, the likelihood function can receive multiple input sets at
        once.
    pass_dict : bool
        If true, the likelihood function expects model parameters as
        dictionaries.
    neural_network_thread_limit : int or None
        Maximum number of threads used by `sklearn`.
    pool : object
        Pool used to parallelize likelihood calls.
    random_state : np.random.RandomState
        Random state of the sampler used for random number generation.
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
        List of arrays where each array at position i list the likelihood of
        points belonging to the i-th bound/shell.
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
    def __init__(self, prior, likelihood, n_dim=None, n_live=1500,
                 n_update=None, enlarge=None, neural_network_kwargs={
                     'hidden_layer_sizes': (100, 50, 20), 'alpha': 0,
                     'learning_rate_init': 1e-2, 'max_iter': 10000,
                     'random_state': 0, 'tol': 1e-4, 'n_iter_no_change': 20},
                 prior_args=[], prior_kwargs={}, likelihood_args=[],
                 likelihood_kwargs={}, n_batch=100, use_neural_networks=True,
                 n_like_new_bound=None, vectorized=False, pass_dict=None,
                 pool=None, neural_network_thread_limit=1, random_state=None):
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
            Factor by which the volume of ellipsoidal bounds is increased.
            Default is 1.1 to the power of `n_dim`, i.e. the ellipsoidal bounds
            are increased by 10% in every dimension.
        neural_network_kwargs : dict, optional
            Keyword arguments passed to the constructor of
            `sklearn.neural_network.MLPRegressor`.
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
            Whether to use neural network emulators in the construction of the
            bounds. Default is True.
        n_like_new_bound : None or int, optional
            The maximum number of likelihood calls before a new bounds is
            created. If None, use 10 times `n_live`. Default is None.
        vectorized : bool, optional
            If true, the likelihood function can receive multiple input sets
            at once. For example, if the likelihood function receives arrays,
            it should be able to take an array with shape (n_points, n_dim)
            and return an array with shape (n_points). Similarly, if the
            likelihood function accepts dictionaries, it should be able to
            process dictionaries where each value is an array with shape
            (n_points). Default is False.
        pass_dict : bool or None, optional
            If true, the likelihood function expects model parameters as
            dictionaries. If false, it expects regular numpy arrays. Default is
            to set it to True if prior was a nautilus.Prior instance and False
            otherwise.
        pool : object or int, optional
            Object with a `map` function used for parallelization of likelihood
            calls, e.g. a multiprocessing.Pool object, or a positive integer.
            If it is an integer, it determines the number of workers in the
            Pool. Default is None.
        neural_network_thread_limit : int or None, optional
            Maximum number of threads used by `sklearn`. If None, no limits
            are applied. Default is 1.
        random_state : int or np.random.RandomState, optional
            Determines random number generation. Pass an int for reproducible
            results accross different runs. Default is None.

        Raises
        ------
        ValueError
            If `prior` is a function and `n_dim` is not given or `pass_struct`
            is set to True. Also, if the dimensionality of the problem is less
            than 2.

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
            elif pass_dict:
                raise ValueError("When passing a function as the 'prior' " +
                                 "argument, 'pass_dict' cannot be True.")
        else:
            self.n_dim = prior.dimensionality()
            if pass_dict is None:
                pass_dict = True

        if self.n_dim <= 1:
            raise ValueError(
                'Cannot run Nautilus with less than 2 parameters.')

        self.n_live = n_live

        if n_update is None:
            self.n_update = n_live
        else:
            self.n_update = n_update

        if n_like_new_bound is None:
            self.n_like_new_bound = 10 * n_live
        else:
            self.n_like_new_bound = n_like_new_bound

        if enlarge is None:
            self.enlarge = 1.1**self.n_dim
        else:
            self.enlarge = enlarge

        self.use_neural_networks = use_neural_networks
        self.neural_network_kwargs = neural_network_kwargs
        self.n_batch = n_batch
        self.vectorized = vectorized
        self.pass_dict = pass_dict
        self.neural_network_thread_limit = neural_network_thread_limit

        if isinstance(pool, int):
            self.pool = Pool(pool)
        elif pool is not None:
            self.pool = pool
        else:
            self.pool = None

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        # The following variables carry the information about the run.
        self.n_like = 0
        self.explored = False
        self.bounds = []
        self.points = []
        self.log_l = []
        self.shell_n = np.zeros(0, dtype=int)
        self.shell_n_sample_shell = np.zeros(0, dtype=int)
        self.shell_n_sample_bound = np.zeros(0, dtype=int)
        self.shell_n_eff = np.zeros(0, dtype=float)
        self.shell_log_l_min = np.zeros(0, dtype=float)
        self.shell_log_l = np.zeros(0, dtype=float)
        self.shell_log_v = np.zeros(0, dtype=float)

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
            Default is true.
        verbose : bool, optional
            If true, print additional information. Default is false.

        """
        if not self.explored:

            if verbose:
                print('#########################')
                print('### Exploration Phase ###')
                print('#########################')
                print()

            while (self.live_evidence_fraction() > f_live):
                self.add_bound(verbose=verbose)
                self.fill_bound(verbose=verbose)

            # If some shells are unoccupied in the end, remove them. They will
            # contain close to 0 volume and may never yield a point when
            # trying to sample from them.
            for shell in reversed(range(len(self.bounds))):
                if self.shell_n[shell] == 0:
                    self.bounds.pop(shell)
                    self.points.pop(shell)
                    self.log_l.pop(shell)
                    self.shell_n = np.delete(self.shell_n, shell)
                    self.shell_log_l = np.delete(self.shell_log_l, shell)
                    self.shell_log_v = np.delete(self.shell_log_v, shell)

            self.explored = True

            if discard_exploration:
                self.discard_points()

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

    def posterior(self, return_dict=False, equal_weight=False):
        """Return the posterior sample estimate.

        Parameters
        ----------
        return_dict : bool, optional
            If true, return `points` as a dictionary.
        equal_weight : bool, optional
            If true, return an equal weighted posterior.

        Returns
        -------
        points : numpy.ndarray or dict
            Points of the posterior.
        log_w : numpy.ndarray
            Weights of each point of the posterior.
        log_l : numpy.ndarray
            Logarithm of the likelihood at each point of the posterior.

        """
        points = np.vstack(self.points)
        log_v = np.repeat(self.shell_log_v -
                          np.log(np.maximum(self.shell_n, 1)), self.shell_n)
        log_l = np.concatenate(self.log_l)
        log_w = log_v + log_l

        if callable(self.prior):
            unit_to_physical = self.prior
        else:
            unit_to_physical = self.prior.unit_to_physical

        if not self.vectorized and callable(self.prior):
            points = np.array(list(map(unit_to_physical, points)))
        else:
            points = unit_to_physical(points)

        if return_dict:
            if callable(self.prior):
                raise ValueError(
                    'Cannot return points as dictionary. The prior passed ' +
                    'to the sampler must have been a nautilus.Prior object.')
            points = self.prior.physical_to_dictionary(points)

        if equal_weight:
            select = (self.random_state.random(len(log_w)) <
                      np.exp(log_w - np.amax(log_w)))
            points = points[select]
            log_w = np.ones(len(points)) * np.log(1.0 / np.sum(select))
            log_l = log_l[select]

        # Normalize weights.
        log_w = log_w - logsumexp(log_w)

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
        return logsumexp(self.shell_log_l + self.shell_log_v)

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

        while n_sample < self.n_batch:
            points = self.bounds[index].sample(self.n_batch - n_sample)
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
                        idx_t = np.append(idx_t, self.random_state.choice(
                            idx_1, size=n, replace=False))
                        shell_t[idx_t] = -1
                        replace[self.random_state.choice(
                            idx_2, size=n, replace=False)] = True

            points = points[~replace]

            if len(points) > 0:
                points_all.append(points)
                n_sample += len(points)

        points = np.vstack(points_all)

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

        """
        if callable(self.prior):
            unit_to_physical = self.prior
        else:
            unit_to_physical = self.prior.unit_to_physical

        if not self.vectorized and callable(self.prior):
            args = np.array(list(map(unit_to_physical, points)))
        else:
            args = unit_to_physical(points)

        if self.pass_dict:
            if self.vectorized:
                args = self.prior.physical_to_dictionary(args)
            else:
                args = [self.prior.physical_to_dictionary(arg) for arg in args]

        if self.pool is not None:
            log_l = self.pool.map(self.likelihood, args)
        elif self.vectorized:
            log_l = self.likelihood(args)
        else:
            log_l = list(map(self.likelihood, args))

        self.n_like += len(log_l)

        return log_l

    def update_shell_info(self, index):
        """Update the shell information for calculation of summary statistics.

        Parameters
        ----------
        index: int
            Index of the shell.

        """
        if isinstance(self.log_l[index], list):
            log_l = np.concatenate(self.log_l[index])
        else:
            log_l = self.log_l[index]

        self.shell_n[index] = len(log_l)

        if self.shell_n[index] > 0:
            self.shell_log_v[index] = (
                self.bounds[index].volume() +
                np.log(self.shell_n_sample_shell[index] /
                       self.shell_n_sample_bound[index]))
            self.shell_log_l[index] = logsumexp(log_l) - np.log(len(log_l))
            self.shell_n_eff[index] = np.exp(2 * logsumexp(log_l) -
                                             logsumexp(2 * log_l))
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
            If true, print additional information. Default is false.

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
            self.bounds.append(
                UnitCube(self.n_dim, random_state=self.random_state))
        else:
            if verbose:
                print('Adding bound {}'.format(len(self.bounds) + 1), end='\r')
            log_l = np.concatenate(self.log_l)
            points = np.vstack(self.points)[np.argsort(log_l)]
            log_l = np.sort(log_l)
            log_l_min = 0.5 * (log_l[-self.n_live] + log_l[-self.n_live - 1])
            bound = NautilusBound(
                points, log_l, log_l_min, self.live_volume(),
                enlarge=self.enlarge,
                use_neural_networks=self.use_neural_networks,
                neural_network_kwargs=self.neural_network_kwargs,
                neural_network_thread_limit=self.neural_network_thread_limit,
                random_state=self.random_state)
            if bound.volume() > self.bounds[-1].volume():
                bound = self.bounds[-1]
            self.bounds.append(bound)

        self.points.append(np.zeros((0, self.n_dim)))
        self.log_l.append(np.zeros(0))
        self.shell_log_l_min = np.append(self.shell_log_l_min, log_l_min)
        self.points[-1] = []
        self.log_l[-1] = []

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
            If true, print additional information. Default is false.

        """
        shell_t = []
        points_t = []
        log_l_t = []

        # Check which points points from previous shells could be transferred
        # to the new bound.
        if len(self.bounds) > 1:
            for shell in range(len(self.bounds) - 1):

                in_bound = self.bounds[-1].contains(self.points[shell])
                shell_t.append(np.repeat(shell, np.sum(in_bound)))
                points_t.append(self.points[shell][in_bound])
                log_l_t.append(self.log_l[shell][in_bound])

                self.points[shell] = self.points[shell][~in_bound]
                self.log_l[shell] = self.log_l[shell][~in_bound]
                self.shell_n[shell] -= np.sum(in_bound)
                self.shell_n_sample_shell[shell] -= np.sum(in_bound)
                self.update_shell_info(shell)

            shell_t = np.concatenate(shell_t)
            points_t = np.vstack(points_t)
            log_l_t = np.concatenate(log_l_t)

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
                        total=n_update, leave=False)

        while n_update < n_update_max and n_like < n_like_max:
            points, n_bound, idx_t = self.sample_shell(-1, shell_t)
            assert len(points) + len(idx_t) == n_bound
            log_l = self.evaluate_likelihood(points)
            self.points[-1].append(points)
            self.log_l[-1].append(log_l)
            if len(idx_t) > 0:
                self.points[-1].append(points_t[idx_t])
                self.log_l[-1].append(log_l_t[idx_t])
                shell_t = np.delete(shell_t, idx_t)
                points_t = np.delete(points_t, idx_t)
                log_l_t = np.delete(log_l_t, idx_t)
            self.shell_n[-1] += n_bound
            self.shell_n_sample_shell[-1] += n_bound
            self.shell_n_sample_bound[-1] += n_bound
            n_update += np.sum(log_l >= log_l_min)
            n_like += len(points)

            if verbose:
                pbar.update(np.sum(log_l >= log_l_min))

        self.points[-1] = np.vstack(self.points[-1])
        self.log_l[-1] = np.concatenate(self.log_l[-1])
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
            points, log_w, log_l = self.posterior()
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
            points, log_w, log_l = self.posterior()
            log_v = log_w + self.evidence() - log_l
            log_v_live = log_v[np.argsort(log_l)][-self.n_live:]
            return logsumexp(log_v_live)

    def add_samples_to_shell(self, index):
        """Add samples to a shell.

        The number of points added is always equal to the batch size.

        Parameters
        ----------
        index : int
            The index of the shell for which to add points.

        """
        points, n_bound = self.sample_shell(index)
        self.shell_n_sample_shell[index] += len(points)
        self.shell_n_sample_bound[index] += n_bound
        log_l = self.evaluate_likelihood(points)
        self.points[index] = np.vstack([self.points[index], points])
        self.log_l[index] = np.concatenate([self.log_l[index], log_l])
        self.update_shell_info(index)

    def discard_points(self):
        """Discard all points drawn."""
        for i in range(len(self.bounds)):
            self.shell_n_sample_shell[i] = 0
            self.shell_n_sample_bound[i] = 0
            self.points[i] = np.zeros((0, self.n_dim))
            self.log_l[i] = np.zeros(0)
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
            If true, print additional information. Default is false.

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
            is true.

        Returns
        -------
        m : numpy.ndarray
            Two-dimensional array with occupation numbers. The element at index
            :math:`(i, j)` corresponds to the occupation of points in shell
            shell :math:`i` that also belong to bound :math:`j`. If
            `fractional` is true, this is the fraction of all points in shell
            :math:`i` and otherwise it is the absolute number.

        """
        m = np.zeros((len(self.bounds), len(self.bounds)), dtype=np.int)

        for i, points in enumerate(self.points):
            for k, bound in enumerate(self.bounds):
                m[i, k] = np.sum(bound.contains(points))

        if fractional:
            m = m / np.diag(m)[:, np.newaxis]

        return m

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
    config : dict
        Sampler run parameters.
    pool : object
        Pool used to parallelize likelihood calls.
    random_state : np.random.RandomState
        Random state of the sampler used for random number generation.
    bounds : list
        List of all the constructed bounds.
    points : list
        List of arrays where each array at position i list the points belonging
        to the i-th bound/shell.
    n_like : int
        Total number of likelihood evaluations.
    log_l : list
        List of arrays where each array at position i list the likelihood of
        points belonging to the i-th bound/shell.
    shell_info : numpy.ndarray
        Array listing several summary statistics for each bound/shell.
    tessellated : bool, optional
        Whether the space has been tessellated and the shells have been
        constructed.

    """

    def __init__(self, prior, likelihood, n_dim=None, n_live=3000,
                 n_update=None, n_like_update=None, enlarge=None, n_batch=100,
                 vectorized=False, pass_struct=None, likelihood_args=[],
                 likelihood_kwargs={}, prior_args=[], prior_kwargs={},
                 threads=1, pool=None, random_state=None):
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
        n_update : int, optional
            The maximum number of additions to the live set before a new bound
            is created. Default is `n_live`.
        n_like_update : int, optional
            The maximum number of likelihood calls before a new bounds is
            created. Default is 10 times `n_live`.
        enlarge : float, optional
            Factor by which the volume of ellipsoidal bounds is increased.
            Default is 1.1 to the power of `n_dim`, i.e. the ellipsoidal bounds
            are increased by 10% in every dimension.
        n_batch : int, optional
            Number of likelihood evaluations that are performed at each step.
            If likelihood evaluations are parallelized, should be multiple
            of the number of parallel processes. Very large numbers can
            lead to new bounds being created long after `n_update` additions to
            the live set have been achieved. This will not cause any bias but
            could reduce efficiency. Default is 100.
        vectorized : bool, optional
            If true, the likelihood function can receive multiple input sets
            at once. For example, if the likelihood function receives arrays,
            it should be able to take an array with shape (n_points, n_dim)
            and return an array with shape (n_points). Similarly, if the
            likelihood function accepts dictionaries, it should be able to
            process dictionaries where each value is an array with shape
            (n_points). Default is False.
        pass_struct : bool, optional
            If true, the likelihood function expects model parameters as
            dictionaries (if not vectorized) or structured numpy arrays. If
            false, it expects regular numpy arrays. Default is to set it to
            True if prior was a nautilus.Prior instance and False otherwise.
        likelihood_args : list, optional
            List of extra positional arguments for `likelihood`.
        likelihood_kwargs : dict, optional
            Dictionary of extra keyword arguments for `likelihood`.
        prior_args : list, optional
            List of extra positional arguments for `prior`. Only used if
            `prior` is a function.
        prior_kwargs : dict, optional
            Dictionary of extra keyword arguments for `prior`. Only used if
            `prior` is a function.
        threads : int, optional
            A positive integer determining the number of processes used. Will
            be ignored if `pool` is provided. Default is 1.
        pool : object, optional
            Object with a `map` function used for parallelization, e.g.
            a multiprocessing.Pool object. Default is None.
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
            if pass_struct is None:
                pass_struct = False
            elif pass_struct:
                raise ValueError("When passing a function as the 'prior' " +
                                 "argument, 'pass_struct' cannot be True.")
        else:
            self.n_dim = prior.dimensionality()
            if pass_struct is None:
                pass_struct = True

        if self.n_dim <= 1:
            raise ValueError(
                'Cannot run Nautilus with less than 2 parameters.')

        self.config = {}
        self.config['n_live'] = n_live

        if n_update is None:
            self.config['n_update'] = n_live
        else:
            self.config['n_update'] = n_update

        if n_like_update is None:
            self.config['n_like_update'] = 10 * n_live
        else:
            self.config['n_like_update'] = n_like_update

        if enlarge is None:
            self.config['enlarge'] = 1.1**self.n_dim
        else:
            self.config['enlarge'] = enlarge

        self.config['n_batch'] = n_batch
        self.config['vectorized'] = vectorized
        self.config['pass_struct'] = pass_struct

        if pool is not None:
            self.pool = pool
        elif threads > 1:
            self.pool = Pool(threads)
        else:
            self.pool = None

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self.bounds = []
        self.points = []
        self.n_like = 0
        self.log_l = []
        self.shell_info = np.array([], dtype=[
            ('log_v', 'f8'), ('log_l', 'f8'), ('log_z', 'f8'), ('n_eff', 'f8'),
            ('n_shell', 'i8'), ('n_bound', 'i8'),
            ('log_l_min_iteration', 'f8')])
        self.tessellated = False

    def run(self, f_live=0.01, n_shell=None, n_eff=10000,
            discard_tesselation=False, verbose=False):
        """Run the sampler until convergence.

        Parameters
        ----------
        f_live : float, optional
            Maximum fraction of the evidence contained in the live set before
            building the initial shells terminates. Default is 0.01.
        n_shell : int, optional
            Minimum number of points in each shell. The algorithm will sample
            from the shells until this is reached. Default is the batch size
            of the sampler which is 100 unless otherwise specified.
        n_eff : float, optional
            Minimum effective sample size. The algorithm will sample from the
            shells until this is reached. Default is 10000.
        discard_tesselation : bool, optional
            Whether to discard points drawn in the tesselation phase. This is
            required for an unbiased evidence estimate. Default is true.
        verbose : bool, optional
            If true, print additional information. Default is false.

        """
        if not self.tessellated:

            if verbose:
                print('#########################')
                print('### Tesselation Phase ###')
                print('#########################')
                print()

            while (self.live_evidence_fraction() > f_live):
                self.add_bound(verbose=verbose)
                self.fill_bound(verbose=verbose)

            # If some shells are unoccupied in the end, remove them. They will
            # contain close to 0 volume.
            for i_shell in reversed(range(len(self.bounds))):
                if len(self.points[i_shell]) == 0:
                    self.bounds.pop(i_shell)
                    self.points.pop(i_shell)
                    self.log_l.pop(i_shell)
                    self.shell_info = np.delete(self.shell_info, i_shell,
                                                axis=0)

            self.tessellated = True

            if discard_tesselation:
                self.discard_points()

        if n_shell is None:
            n_shell = self.config['n_batch']

        if (np.any(self.shell_info['n_shell'] < n_shell) or
                self.effective_sample_size() < n_eff):

            if verbose:
                print('#########################')
                print('##### Sampling Phase ####')
                print('#########################')
                print()

            self.add_points(n_shell=n_shell, n_eff=n_eff, verbose=verbose)

    def posterior(self, return_struct=False, equal_weight=False):
        """Return the posterior sample estimate.

        Parameters
        ----------
        return_struct : bool, optional
            If true, return `points` as a structured numpy array.
        equal_weight : bool, optional
            If true, return an equal weighted posterior.

        Returns
        -------
        points : numpy.ndarray
            Coordinates of the posterior.
        log_w : numpy.ndarray
            Weights of each coordinate of the posterior.
        log_l : numpy.ndarray
            Logarithm of the likelihood at each coordinate of the posterior.

        """
        points = np.vstack(self.points)
        select = self.shell_info['n_shell'] > 0
        log_v = np.repeat(self.shell_info['log_v'][select] -
                          np.log(self.shell_info['n_shell'])[select],
                          self.shell_info['n_shell'][select])
        log_l = np.concatenate(self.log_l)
        log_w = log_v + log_l

        if callable(self.prior):
            unit_to_physical = self.prior
        else:
            unit_to_physical = self.prior.unit_to_physical

        if not self.config['vectorized'] and callable(self.prior):
            points = np.array(list(map(unit_to_physical, points)))
        else:
            points = unit_to_physical(points)

        if return_struct:
            if callable(self.prior):
                raise ValueError('Cannot return structured array. The prior ' +
                                 'passed to the sampler must have been a ' +
                                 'nautilus.Prior object.')
            points = self.prior.physical_to_structure(points)

        if equal_weight:
            select = (self.random_state.random(len(points)) <
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
        select = self.shell_info['n_shell'] > 0
        sum_w = np.exp(self.shell_info['log_l'] + self.shell_info['log_v'] -
                       np.nanmax(self.shell_info['log_l']) -
                       np.nanmax(self.shell_info['log_v']))[select]
        sum_w_sq = sum_w**2 / self.shell_info['n_eff'][select]
        return np.sum(sum_w)**2 / np.sum(sum_w_sq)

    def evidence(self):
        r"""Estimate the global evidence :math:`\log \mathcal{Z}`.

        Returns
        -------
        log_z : float
            Estimate the global evidence :math:`\log \mathcal{Z}`.

        """
        return logsumexp(self.shell_info['log_z'])

    def sample_shell(self, index, n_shell):
        """Sample points uniformly from a shell.

        The shell at index :math:`i` is defined as the volume enclosed by the
        bound of index :math:`i` and enclosed by not other bound of index
        :math:`k` with :math:`k > i`.

        Parameters
        ----------
        index : int
            Index of the shell.
        n_shell : int
            Total number of samples.

        Returns
        -------
        points : numpy.ndarray
            Array of shape (n_shell, n_dim) containing points sampled uniformly
            from the shell.
        n_bound : int
            Number of points drawn within the bound at index :math:`i`. Will
            be different from `n_shell` if there are bounds with index
            :math:`k` with :math:`k > i`.

        """
        n_bound = 0
        points = []

        while len(points) < n_shell:
            point = self.bounds[index].sample()
            n_bound += 1
            in_shell = True

            for bound in self.bounds[index:][1:]:
                if in_shell and bound.contains(point):
                    in_shell = False

            if in_shell:
                points.append(point)

        return np.array(points), n_bound

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

        if not self.config['vectorized'] and callable(self.prior):
            args = np.array(list(map(unit_to_physical, points)))
        else:
            args = unit_to_physical(points)

        if self.config['pass_struct']:
            if self.config['vectorized']:
                args = self.prior.physical_to_structure(args)
            else:
                args = [self.prior.physical_to_structure(arg) for arg in args]

        if self.pool is not None:
            log_l = self.pool.map(self.likelihood, args)
        elif self.config['vectorized']:
            log_l = self.likelihood(args)
        else:
            log_l = list(map(self.likelihood, args))

        log_l = np.array(list(log_l))
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

        self.shell_info['n_shell'][index] = len(log_l)

        if self.shell_info['n_shell'][index] > 0:
            self.shell_info['log_v'][index] = (
                self.bounds[index].volume() +
                np.log(self.shell_info['n_shell'][index] /
                       self.shell_info['n_bound'][index]))
            self.shell_info['log_l'][index] = (
                logsumexp(log_l) - np.log(len(log_l)))
            self.shell_info['log_z'][index] = (
                self.shell_info['log_v'][index] +
                self.shell_info['log_l'][index])
            self.shell_info['n_eff'][index] = np.exp(
                2 * logsumexp(log_l) - logsumexp(2 * log_l))
        else:
            self.shell_info['log_v'][index] = -np.inf
            self.shell_info['log_l'][index] = np.nan
            self.shell_info['log_z'][index] = -np.inf
            self.shell_info['n_eff'][index] = 0

    def print_status(self):
        """Print current summary statistics."""
        print('N_like: {:>17}'.format(self.n_like))
        print('N_eff: {:>18.0f}'.format(self.effective_sample_size()))
        print('log Z: {:>18.3f}'.format(self.evidence()))
        if not self.tessellated:
            print('log V: {:>18.3f}'.format(self.shell_info['log_v'][-1]))
            print('f_live: {:>17.3f}'.format(self.live_evidence_fraction()))

    def add_bound(self, verbose=False):
        """Build a new bound from existing points.

        Parameters
        ----------
        verbose : bool, optional
            If true, print additional information. Default is false.

        """
        # If this is the first bound, use the UnitCube bound.
        if len(self.bounds) == 0:
            log_l_min_iteration = -np.inf
            self.bounds.append(
                UnitCube(self.n_dim, random_state=self.random_state))
        else:
            if verbose:
                print('Adding bound {}'.format(len(self.bounds) + 1), end='\r')
            log_l_all = np.concatenate(self.log_l)
            points_all = np.vstack(self.points)[np.argsort(log_l_all)]
            log_l_all = np.sort(log_l_all)
            log_l_min_iteration = 0.5 * (
                log_l_all[-self.config['n_live']] +
                log_l_all[-self.config['n_live'] - 1])
            new_bound = NautilusBound(
                points_all, log_l_all, log_l_min_iteration,
                self.live_volume(), enlarge=self.config['enlarge'],
                random_state=self.random_state)
            if new_bound.volume() > self.bounds[-1].volume():
                new_bound = self.bounds[-1]
            self.bounds.append(new_bound)

        self.points.append(np.zeros((0, self.n_dim)))
        self.log_l.append(np.zeros(0))
        self.shell_info = np.append(
            self.shell_info, np.array([0], dtype=self.shell_info.dtype))
        self.shell_info['log_l_min_iteration'][-1] = log_l_min_iteration

        if verbose:
            print('Adding bound {:<7} done'.format(
                str(len(self.bounds)) + ':'))
            if isinstance(self.bounds[-1], NautilusBound):
                n_neu = len(self.bounds[-1].nbounds)
                if isinstance(self.bounds[-1].sample_bound, UnitCube):
                    n_ell = 0
                else:
                    n_ell = len(self.bounds[-1].sample_bound.ells)
            else:
                n_neu = 0
                n_ell = 0
            n_ell = max(n_neu, n_ell)
            print("Neural nets: {:>12}".format(n_neu))
            print("Ellipsoids: {:>13}".format(n_ell))

    def fill_bound(self, verbose=False):
        """Fill a new bound with points until a new bound should be created.

        Parameters
        ----------
        verbose : bool, optional
            If true, print additional information. Default is false.

        """
        n_transfer = np.zeros(len(self.bounds) - 1, dtype=int)
        points_transfer = []
        log_l_transfer = []

        # Check which points points from the previous shells should be
        # transerred to the new bound.
        if len(self.bounds) > 1:
            for i_previous_shell in range(len(self.bounds) - 1):

                in_bound = self.bounds[-1].contains(
                    self.points[i_previous_shell])
                points_transfer.append(self.points[i_previous_shell][in_bound])
                log_l_transfer.append(self.log_l[i_previous_shell][in_bound])
                n_transfer[i_previous_shell] = np.sum(in_bound)

                self.points[i_previous_shell] = self.points[
                    i_previous_shell][~in_bound]
                self.log_l[i_previous_shell] = self.log_l[
                    i_previous_shell][~in_bound]
                self.update_shell_info(i_previous_shell)

        log_l_min = self.shell_info['log_l_min_iteration'][-1]
        i_update = 0
        i_like = 0
        n_update = self.config['n_update']
        n_like = self.config['n_like_update']
        if len(self.bounds) == 1:
            n_update += self.config['n_live']
            n_like = np.inf
        self.points[-1] = []
        self.log_l[-1] = []

        if verbose:
            pbar = tqdm(desc='Filling bound {}'.format(len(self.bounds)),
                        total=n_update, leave=False)

        while i_update < n_update and i_like < n_like:

            points = self.sample_shell(-1, self.config['n_batch'])[0]

            # Check if points from previous shells can be transferred.
            if np.any(n_transfer > 0):

                # Determine the shells newly drawn points belong to.
                i_shell = self.shell_association(
                    points, n_max=len(self.bounds) - 1)

                # If possible, replace newly drawn points with old points
                # from the same shell.
                use = np.ones(len(points), dtype=bool)
                for i in range(len(points)):
                    if n_transfer[i_shell[i]] > 0:
                        n_transfer[i_shell[i]] -= 1
                        k = n_transfer[i_shell[i]]
                        self.points[-1].append(points_transfer[i_shell[i]][k])
                        self.log_l[-1].append([log_l_transfer[i_shell[i]][k]])
                        use[i] = False

                points = points[use]

            log_l = self.evaluate_likelihood(points)
            self.points[-1].append(points)
            self.log_l[-1].append(log_l)
            self.shell_info['n_shell'][-1] += self.config['n_batch']
            self.shell_info['n_bound'][-1] += self.config['n_batch']
            i_update += np.sum(log_l >= log_l_min)
            i_like += len(log_l)

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
            log_w_live = log_w[np.argsort(log_l)][-self.config['n_live']:]
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
            log_v_live = log_v[np.argsort(log_l)][-self.config['n_live']:]
            return logsumexp(log_v_live)

    def add_samples_to_shell(self, index):
        """Add samples to a shell.

        The number of points added is always equal to the batch size.

        Parameters
        ----------
        index : int
            The index of the shell for which to add points.

        """
        points, n_bound = self.sample_shell(index, self.config['n_batch'])
        self.shell_info['n_shell'][index] += len(points)
        self.shell_info['n_bound'][index] += n_bound
        log_l = self.evaluate_likelihood(points)
        self.points[index] = np.vstack([self.points[index], points])
        self.log_l[index] = np.concatenate([self.log_l[index], log_l])
        self.update_shell_info(index)

    def discard_points(self):
        """Discard all points drawn."""
        for i in range(len(self.bounds)):
            self.shell_info['n_shell'][i] = 0
            self.shell_info['n_bound'][i] = 0
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
        i_shell = np.arange(len(self.bounds))[
            self.shell_info['n_shell'] < n_shell]
        if verbose and len(i_shell):
            pbar = tqdm(desc='Sampling shells ', total=len(i_shell),
                        leave=False)

        for i in i_shell:
            while self.shell_info['n_shell'][i] < n_shell:
                self.add_samples_to_shell(i)
            if verbose:
                pbar.update(1)

        if verbose and len(i_shell) > 0:
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
                self.shell_info['log_z'] -
                0.5 * np.log(self.shell_info['n_shell']) -
                0.5 * np.log(self.shell_info['n_eff']))
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
        """Determine the shells each points belongs to.

        Parameters
        ----------
        points : numpy.ndarray
            Points for which to determine shell association.
        n_max : int, optional
            The maximum number of shells to consider. Effectively, this
            determines the shell association at step `n_max` in the
            tesselation phase. Default is to consider all shells.

        Returns
        -------
        i_shell: int
            Shell association for each point.

        """
        if n_max is None:
            n_max = len(self.bounds)

        i_shell = np.repeat(-1, len(points))
        for i, bound in reversed(list(enumerate(self.bounds[:n_max]))):
            mask = i_shell >= 0
            if np.all(mask):
                break
            mask[~mask] = ~bound.contains(points[~mask])
            i_shell[~mask] = i

        return i_shell

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

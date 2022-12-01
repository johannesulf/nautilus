import os
import sys
import logging

import argparse
import numpy as np
from astropy.table import vstack, Table


import dynesty
import nautilus
import ultranest
import emcee
import pocomc

import kepler
from scipy.special import gamma
from scipy.optimize import rosen
from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
from nautilus import Prior
from tabcorr import TabCorr
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import AssembiasZheng07Sats


def prior(x):
    # Force parameters to (0, 1) due to bug in pocoMC.
    return np.minimum(np.maximum(x, 0 + 1e-10), 1 - 1e-10)


def filter_outside_unit(x, log_l):
    if x.ndim == 2:
        log_l = np.where(np.any((x < 0) | (x > 1), axis=-1), -np.inf, log_l)
    elif np.any(x < 0) or np.any(x > 1):
        log_l = -np.inf
    return log_l


def loggamma_logpdf(x, c, loc, scale):

    y = (x - loc) / scale

    return c * y - np.exp(y) - np.log(gamma(c)) - np.log(scale)


def loggamma_likelihood(x):

    d = x.shape[-1]

    if d < 2:
        raise Exception('LogGamma is only defined for d >= 2.')

    g_a = loggamma_logpdf(x[..., 0], 1.0, 1.0 / 3.0, 1.0 / 30.0)
    g_b = loggamma_logpdf(x[..., 0], 1.0, 2.0 / 3.0, 1.0 / 30.0)
    n_c = norm.logpdf(x[..., 1], 1.0 / 3.0, 1.0 / 30.0)
    n_d = norm.logpdf(x[..., 1], 2.0 / 3.0, 1.0 / 30.0)

    g_n = np.maximum(g_a, g_b)
    log_l_1 = np.log(0.5 * (np.exp(g_a - g_n) + np.exp(g_b - g_n))) + g_n

    n_n = np.maximum(n_c, n_d)
    log_l_2 = np.log(0.5 * (np.exp(n_c - n_n) + np.exp(n_d - n_n))) + n_n

    log_l = log_l_1 + log_l_2

    for i in range(2, d):
        if i <= (d + 2) / 2 - 1:
            log_l += loggamma_logpdf(x[..., i], 1.0, 2.0 / 3.0, 1.0 / 30.0)
        else:
            log_l += norm.logpdf(x[..., i], 2.0 / 3.0, 1.0 / 30.0)

    return filter_outside_unit(x, log_l)


def rosenbrock_likelihood(x):

    log_l = -rosen((x.T - 0.5) * 10)

    return filter_outside_unit(x, log_l)


def funnel_likelihood(x):
    gamma = 0.95
    log_l_1 = -0.5 * np.log(2 * np.pi) - ((x[..., 0] - 0.5) * 20)**2 / 2.0
    sigma = np.exp((x[0] - 0.5) * 20)
    cov = (np.ones((len(x) - 1, len(x) - 1)) * gamma * sigma +
           np.eye(len(x) - 1) * (1 - gamma) * sigma)
    log_l_2 = multivariate_normal.logpdf((x[1:] - 0.5) * 20, cov=cov)

    log_l = log_l_1 + log_l_2 + len(x) * np.log(20)

    return filter_outside_unit(x, log_l)


halotab = TabCorr.read('benchmarks/bolplanck.hdf5')
cens_occ_model = AssembiasZheng07Cens()
sats_occ_model = AssembiasZheng07Sats()
model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model)
n_obs = 6.37e-3
n_err = 0.75e-3
wp_obs = np.genfromtxt('benchmarks/wp_dr72_bright0_mr20.0_z0.106_nj400')[:, 1]
wp_cov = np.genfromtxt('benchmarks/wpcov_dr72_bright0_mr20.0_z0.106_nj400')
wp_pre = np.linalg.inv(wp_cov)

gal_prior = Prior()
gal_prior.add_parameter('logMmin', dist=(9, 14))
gal_prior.add_parameter('sigma_logM', dist=(0.01, 1.5))
gal_prior.add_parameter('logM0', dist=(9, 14))
gal_prior.add_parameter('logM1', dist=(10.7, 15.0))
gal_prior.add_parameter('alpha', dist=(0, 2))
gal_prior.add_parameter('mean_occupation_centrals_assembias_param1',
                        dist=(-1, +1))
gal_prior.add_parameter('mean_occupation_satellites_assembias_param1',
                        dist=(-1, +1))


def cosmology_likelihood(x):

    model.param_dict.update(
        gal_prior.physical_to_structure(gal_prior.unit_to_physical(x)))

    n_mod, wp_mod = halotab.predict(model)

    log_l = -0.5 * (
        (n_mod - n_obs)**2 / n_err**2 +
        np.inner(np.inner(wp_mod - wp_obs, wp_pre), wp_mod - wp_obs))

    return filter_outside_unit(x, log_l)


data = np.genfromtxt('benchmarks/california-planet-search.csv')

exo_v_err = np.array(data[:, 0])
exo_t = np.array(data[:, 1])
exo_v = np.array(data[:, 2])
exo_t_ref = 0.5 * (exo_t.min() + exo_t.max())

exo_prior = Prior()
t0 = [2072.7948, 2082.6251]
exo_prior.add_parameter('t_0_a', dist=norm(loc=t0[0], scale=0.0007))
exo_prior.add_parameter('t_0_b', dist=norm(loc=t0[1], scale=0.0004))
p = [20.8851, 42.3633]
exo_prior.add_parameter('P_a', dist=norm(loc=p[0], scale=0.0003))
exo_prior.add_parameter('P_b', dist=norm(loc=p[1], scale=0.0006))
k = np.array([5.05069163, 5.50983542])
exo_prior.add_parameter('log_K_a', dist=norm(loc=np.log(k[0]), scale=2))
exo_prior.add_parameter('log_K_b', dist=norm(loc=np.log(k[1]), scale=2))
exo_prior.add_parameter('ecc_a')
exo_prior.add_parameter('ecc_b')
exo_prior.add_parameter('w_a', dist=(-np.pi, np.pi))
exo_prior.add_parameter('w_b', dist=(-np.pi, np.pi))
exo_prior.add_parameter('logs', dist=norm(loc=np.log(np.median(exo_v_err)),
                                          scale=5.0))
exo_prior.add_parameter('trend_1', dist=norm(scale=0.01))
exo_prior.add_parameter('trend_2', dist=norm(scale=0.1))
exo_prior.add_parameter('trend_3', dist=norm(scale=1.0))


def solve_offset(ecc, w):
    t = np.linspace(0, 2 * np.pi, 100)
    ecc_anom = kepler.solve(t, ecc)
    f = (2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(ecc_anom / 2.0),
        np.sqrt(1 - ecc) * np.cos(ecc_anom / 2.0)))
    f[-1] = 2 * np.pi
    t = interp1d(f, t)
    if np.pi / 2 - w < 0:
        return t(np.pi / 2 - w + 2 * np.pi)
    else:
        return t(np.pi / 2 - w)


def radial_velocity(t, P, t0, ecc, w, k):

    ecc_anom = kepler.solve(2 * np.pi * (t - t0) / P + solve_offset(ecc, w),
                            ecc)
    f = (2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(ecc_anom / 2.0),
        np.sqrt(1 - ecc) * np.cos(ecc_anom / 2.0)))
    return k * (np.cos(f + w) + ecc * np.cos(w))


def exoplanet_likelihood(x):

    if np.any(x < 0) or np.any(x > 1):
        return -np.inf

    param_dict = exo_prior.physical_to_structure(exo_prior.unit_to_physical(x))
    p = [param_dict['P_a'], param_dict['P_b']]
    t0 = [param_dict['t_0_a'], param_dict['t_0_b']]
    ecc = [param_dict['ecc_a'], param_dict['ecc_b']]
    w = [param_dict['w_a'], param_dict['w_b']]
    k = np.exp([param_dict['log_K_a'], param_dict['log_K_b']])
    trend = [param_dict['trend_1'], param_dict['trend_2'],
             param_dict['trend_3']]
    v_mod = (radial_velocity(exo_t, p[0], t0[0], ecc[0], w[0], k[0]) +
             radial_velocity(exo_t, p[1], t0[1], ecc[1], w[1], k[1]) +
             np.dot(np.vander(exo_t - exo_t_ref, 3), trend))
    v_err = np.sqrt(exo_v_err**2 + np.exp(2 * param_dict['logs']))
    return np.sum(norm.logpdf(exo_v, loc=v_mod, scale=v_err))


def main():

    parser = argparse.ArgumentParser(
        description='Run a likelihood analysis multiple times with ' +
                    'different samplers.')
    parser.add_argument('likelihood', help='the likehood function to use')
    parser.add_argument('--n_run', type=int, default=1,
                        help='number likelihood runs for each sampler')
    parser.add_argument('--sampler', default='ndu',
                        help='which samplers to use')
    parser.add_argument('--dynesty', default='urs',
                        help='which dynesty sampling modes to use')
    parser.add_argument('--nautilus', default=1500, type=int,
                        help='how many live points to use')
    parser.add_argument('--emcee', default=50, type=float,
                        help='how many autocorrelation times')
    parser.add_argument('--full', action='store_true',
                        help='whether to store the full posterior')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if not args.verbose:
        sys.stdout = open(os.devnull, "w")

    if '-' in args.likelihood:
        if args.likelihood.split('-')[0] == 'loggamma':
            likelihood = loggamma_likelihood
            n_dim = int(args.likelihood.split('-')[1])
        elif args.likelihood.split('-')[0] == 'rosenbrock':
            likelihood = rosenbrock_likelihood
            n_dim = int(args.likelihood.split('-')[1])
        elif args.likelihood.split('-')[0] == 'funnel':
            likelihood = funnel_likelihood
            n_dim = int(args.likelihood.split('-')[1])
    elif args.likelihood == 'cosmology':
        likelihood = cosmology_likelihood
        n_dim = 7
    elif args.likelihood == 'exoplanet':
        likelihood = exoplanet_likelihood
        n_dim = 14
    elif args.likelihood == 'galaxy':
        from galaxy_likelihood import galaxy_likelihood
        likelihood = galaxy_likelihood
        n_dim = 5
    else:
        raise Exception("Unknown likelihood '{}'.".format(args.likelihood))

    for iteration in range(args.n_run):
        for sampling_algorithm in ['nautilus', 'nautilus-r', 'emcee',
                                   'dynesty-u', 'dynesty-r', 'dynesty-s',
                                   'UltraNest', 'pocoMC']:

            if sampling_algorithm[0].lower() not in args.sampler:
                continue

            if sampling_algorithm[:7] == 'dynesty':

                if sampling_algorithm == 'dynesty-u' and n_dim > 20:
                    continue

                sample = sampling_algorithm.split('-')[1]

                if sample.lower() not in args.dynesty:
                    continue

                if sample == 'u':
                    sample = 'unif'
                elif sample == 'r':
                    sample = 'rwalk'
                else:
                    sample = 'slice'

                sampler = dynesty.NestedSampler(
                    likelihood, prior, n_dim, sample=sample)
                sampler.run_nested(print_progress=args.verbose)

                results = sampler.results
                log_z = results.logz[-1]
                n_like = np.sum(results.ncall)
                points = results.samples
                weights = np.exp(results.logwt) / np.amax(
                    np.exp(results.logwt))
                n_eff = np.sum(weights)**2 / np.sum(weights**2)

            elif sampling_algorithm == 'UltraNest':

                sampler = ultranest.integrator.NestedSampler(
                    [str(i) for i in range(n_dim)], likelihood, prior)

                result = sampler.run()
                log_z = result['logz']
                n_like = result['ncall']
                points = result['weighted_samples']['points']
                weights = result['weighted_samples']['weights']
                n_eff = np.sum(weights)**2 / np.sum(weights**2)

            elif sampling_algorithm in ['nautilus', 'nautilus-r']:

                if sampling_algorithm == 'nautilus':
                    sampler = nautilus.Sampler(
                        prior, likelihood, n_dim, pass_struct=False,
                        n_live=args.nautilus)
                    sampler.run(verbose=args.verbose)
                else:
                    sampler.discard_points()
                    sampler.run(verbose=args.verbose, n_eff=10000)

                log_z = sampler.evidence()
                n_like = sampler.n_like
                n_eff = sampler.effective_sample_size()
                points, log_w, log_l = sampler.posterior()
                weights = np.exp(log_w - np.amax(log_w))
                if args.nautilus != 1500:
                    sampling_algorithm += '-{}'.format(args.nautilus)

            elif sampling_algorithm == 'emcee':

                logger = logging.getLogger('emcee.autocorr')
                logger.disabled = True

                if args.likelihood.split('-')[0] not in [
                        'rosenbrock', 'cosmology', 'exoplanet']:
                    continue

                if args.likelihood.split('-')[0] == 'rosenbrock':
                    vectorize = True
                else:
                    vectorize = False

                n_walkers = 1000

                if args.likelihood.split('-')[0] == 'rosenbrock':
                    thin_by = 1000
                else:
                    thin_by = 10

                sampler = emcee.EnsembleSampler(
                    n_walkers, n_dim, likelihood, vectorize=vectorize)
                coords = (np.ones((n_walkers, n_dim)) * 0.5 +
                          np.random.normal(size=(n_walkers, n_dim)) * 0.01)

                for sample in sampler.sample(coords, iterations=50000,
                                             thin_by=thin_by):

                    if sampler.iteration % 1000:
                        continue

                    chain = sampler.get_chain()
                    n_steps = len(chain)
                    discard = n_steps // 5
                    tau = emcee.autocorr.integrated_time(
                        chain[discard:], quiet=True)
                    if args.verbose:
                        print('steps: {}, tau: {:.1f}'.format(
                            n_steps, np.amax(tau)))

                    if n_steps >= 1000 and np.all(n_steps > args.emcee * tau):
                        break

                log_z = np.nan
                n_like = n_walkers * n_steps * thin_by
                n_eff = n_like / np.amax(tau) / thin_by
                points = sampler.get_chain(discard=n_steps // 5, flat=True)
                weights = np.ones(len(points))

            elif sampling_algorithm == 'pocoMC':
                n_particles = 1000

                def log_prior(x):
                    if np.any(x < 0) or np.any(x > 1.0):
                        return -np.inf
                    else:
                        return 0.0

                sampler = pocomc.Sampler(
                    n_particles, n_dim, log_likelihood=likelihood,
                    log_prior=log_prior, vectorize_likelihood=False,
                    bounds=(0.0, 1.0), infer_vectorization=False)
                prior_samples = np.random.uniform(
                    low=0.0, high=1.0, size=(n_particles, n_dim))
                sampler.run(prior_samples, progress=args.verbose)
                sampler.add_samples(9000, progress=args.verbose)

                result = sampler.results
                log_z = sampler.bridge_sampling()[0]
                n_like = result['ncall'][-1]
                points = result['samples']
                weights = np.ones(len(points))
                n_eff = len(points)

            result = {}
            result['sampler'] = sampling_algorithm
            result['log Z'] = log_z
            result['N_like'] = n_like
            result['N_eff'] = n_eff
            for i in range(n_dim):
                result['x_{}'.format(i)] = np.histogram(
                    points[:, i], np.linspace(0, 1, 1001), density=True,
                    weights=weights)[0]

            path = os.path.join('benchmarks', args.likelihood)

            try:
                os.mkdir(path)
            except FileExistsError:
                pass

            path = os.path.join(path, str(os.getpid()) + '.hdf5')

            if type(result) != list:
                result = [result, ]

            if os.path.exists(path):
                table = Table.read(path)
                table_add = Table(result)
                table = vstack([table, table_add])
            else:
                table = Table(result)

            table.write(path, overwrite=True, path='data')

            if args.full and iteration == 0:
                table = Table()
                table['points'] = points
                table['weights'] = weights
                path = os.path.join(
                    'benchmarks', args.likelihood + '_' +
                    sampling_algorithm + '_posterior.hdf5')
                table.write(path, overwrite=True, path='data')


if __name__ == '__main__':
    sys.exit(main())

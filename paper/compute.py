import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from astropy.table import vstack, Table

import dynesty
import nautilus
import ultranest
import emcee
import pocomc


def prior(x):
    return x


def write(likelihood, algorithm, n_dim, log_z, n_like, n_eff, points, log_l,
          weights, full):

    result = {}
    result['sampler'] = algorithm
    result['log Z'] = log_z
    result['N_like'] = n_like
    result['N_eff'] = n_eff
    result['bmd'] = 2 * (np.average(log_l**2, weights=weights) -
                         np.average(log_l, weights=weights)**2)
    for i in range(n_dim):
        result['x_{}'.format(i)] = np.histogram(
            points[:, i], np.linspace(0, 1, 1001), density=True,
            weights=weights)[0]

    table = Table([result, ])

    path = Path('.') / 'results' / likelihood / algorithm
    path.mkdir(parents=True, exist_ok=True)

    path = path / '{}.hdf5'.format(str(os.getpid()))

    if path.is_file():
        table = vstack([Table.read(path), table])

    table.write(path, overwrite=True, path='data')

    if full:
        table = Table()
        table['points'] = points
        table['weights'] = weights
        path = Path('.') / 'results' / '{}_{}_posterior.hdf5'.format(
            likelihood, algorithm)
        table.write(path, overwrite=True, path='data')


def main():

    parser = argparse.ArgumentParser(
        description='Run a likelihood analysis multiple times with ' +
                    'different samplers.')
    parser.add_argument('likelihood', help='the likehood function to use')
    parser.add_argument('sampler', help='which sampler(s) to use', nargs='+',
                        choices=['nautilus', 'dynesty-u', 'dynesty-r',
                                 'dynesty-s', 'pocoMC', 'UltraNest', 'emcee'])
    parser.add_argument('--n_run', type=int, default=5,
                        help='number likelihood runs for each sampler')
    parser.add_argument('--n_live', default=2000, type=int,
                        help='how many live points nautilus uses')
    parser.add_argument('--enlarge_per_dim', default=1.05, type=float,
                        help='enlargmenet factor for nautilus')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='number of jobs used by nautilus')
    parser.add_argument('--full', action='store_true',
                        help='whether to store the full posterior')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.n_run > 1 and args.full:
        raise ValueError('Can only save full posterior for one run.')

    if not args.verbose:
        sys.stdout = open(os.devnull, "w")

    if '-' in args.likelihood:
        if args.likelihood.split('-')[0] == 'loggamma':
            from likelihoods.analytic import loggamma_likelihood as likelihood
            n_dim = int(args.likelihood.split('-')[1])
        elif args.likelihood.split('-')[0] == 'rosenbrock':
            from likelihoods.analytic import rosenbrock_likelihood as likelihood
            n_dim = int(args.likelihood.split('-')[1])
        elif args.likelihood.split('-')[0] == 'funnel':
            from likelihoods.analytic import funnel_likelihood as likelihood
            n_dim = int(args.likelihood.split('-')[1])
    elif args.likelihood == 'cosmology':
        from likelihoods.cosmology import likelihood
        n_dim = 7
    elif args.likelihood == 'exoplanet':
        from likelihoods.exoplanet import likelihood
        n_dim = 14
    elif args.likelihood == 'galaxy':
        from likelihoods.galaxy import likelihood
        n_dim = 7
    else:
        raise Exception("Unknown likelihood '{}'.".format(args.likelihood))

    for iteration in range(args.n_run):
        for algorithm in args.sampler:

            if algorithm in ['dynesty-u', 'dynesty-r', 'dynesty-s']:

                if algorithm == 'dynesty-u':
                    mode = 'unif'
                elif algorithm == 'dynesty-r':
                    mode = 'rwalk'
                else:
                    mode = 'slice'

                sampler = dynesty.DynamicNestedSampler(
                    likelihood, prior, n_dim, sample=mode)
                sampler.run_nested(print_progress=args.verbose)

                results = sampler.results
                log_z = results.logz[-1]
                n_like = np.sum(results.ncall)
                points = results.samples
                log_l = results.logl
                weights = np.exp(results.logwt) / np.amax(
                    np.exp(results.logwt))
                n_eff = np.sum(weights)**2 / np.sum(weights**2)

                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)

            elif algorithm == 'UltraNest':

                sampler = ultranest.integrator.ReactiveNestedSampler(
                    [str(i) for i in range(n_dim)], likelihood, prior)

                result = sampler.run()
                log_z = result['logz']
                n_like = result['ncall']
                points = result['weighted_samples']['points']
                log_l = result['weighted_samples']['logl']
                weights = result['weighted_samples']['weights']
                n_eff = np.sum(weights)**2 / np.sum(weights**2)

                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)

            elif algorithm == 'nautilus':

                sampler = nautilus.Sampler(
                    prior, likelihood, n_dim, pass_dict=False,
                    n_live=args.n_live, enlarge_per_dim=args.enlarge_per_dim,
                    pool=args.n_jobs)
                sampler.run(verbose=args.verbose)

                log_z = sampler.evidence()
                n_like = sampler.n_like
                n_eff = sampler.effective_sample_size()
                points, log_w, log_l = sampler.posterior()
                weights = np.exp(log_w - np.amax(log_w))

                if args.n_live != 2000:
                    algorithm += '-{}'.format(args.n_live)
                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)

                sampler.discard_exploration = True
                sampler.run(verbose=args.verbose, n_eff=10000,
                            discard_exploration=True)
                log_z = sampler.evidence()
                n_like = sampler.n_like
                n_eff = sampler.effective_sample_size()
                points, log_w, log_l = sampler.posterior()
                weights = np.exp(log_w - np.amax(log_w))
                algorithm += '-r'
                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)

                sampler.run(verbose=args.verbose, n_eff=100000,
                            discard_exploration=True)
                log_z = sampler.evidence()
                n_like = sampler.n_like
                n_eff = sampler.effective_sample_size()
                points, log_w, log_l = sampler.posterior()
                weights = np.exp(log_w - np.amax(log_w))
                algorithm += '100k'
                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)

            elif algorithm == 'emcee':

                logger = logging.getLogger('emcee.autocorr')
                logger.disabled = True

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

                    if n_steps >= 1000 and np.all(n_steps > 1000 * tau):
                        break

                log_z = np.nan
                n_like = n_walkers * n_steps * thin_by
                n_eff = n_like / np.amax(tau) / thin_by
                points = sampler.get_chain(discard=n_steps // 5, flat=True)
                log_l = sampler.get_log_prob(discard=n_steps // 5, flat=True)
                weights = np.ones(len(points))

                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)

            elif algorithm == 'pocoMC':
                n_particles = 1000

                def log_prior(x):
                    if np.any(x <= 0) or np.any(x >= 1.0):
                        return -np.inf
                    else:
                        return 0.0

                sampler = pocomc.Sampler(
                    n_particles, n_dim, log_likelihood=likelihood,
                    log_prior=log_prior, vectorize_likelihood=False,
                    bounds=(1e-9, 1.0 - 1e-9), infer_vectorization=False)
                prior_samples = np.random.uniform(
                    low=1e-9, high=1.0 - 1e-9, size=(n_particles, n_dim))
                sampler.run(prior_samples, progress=args.verbose)
                sampler.add_samples(9000, progress=args.verbose)

                result = sampler.results
                log_z = sampler.bridge_sampling()[0]
                n_like = result['ncall'][-1]
                points = result['samples']
                log_l = result['loglikelihood']
                weights = np.ones(len(points))
                n_eff = len(points)

                write(args.likelihood, algorithm, n_dim, log_z, n_like, n_eff,
                      points, log_l, weights, args.full)


if __name__ == '__main__':
    sys.exit(main())

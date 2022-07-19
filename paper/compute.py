import os
import sys
import argparse
import numpy as np
from astropy.table import vstack, Table
from multiprocessing import Pool, cpu_count


import dynesty
import nautilus
import ultranest
import emcee
import pocomc

from benchmarks import (prior, loggamma_likelihood, rosenbrock_likelihood,
                        funnel_likelihood, galaxy_likelihood,
                        exoplanet_likelihood)


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
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--multi', action='store_true')

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
    elif args.likelihood == 'galaxy':
        likelihood = galaxy_likelihood
        n_dim = 7
    elif args.likelihood == 'exoplanet':
        likelihood = exoplanet_likelihood
        n_dim = 14
    else:
        raise Exception("Unknown likelihood '{}'.".format(args.likelihood))

    for iteration in range(args.n_run):
        for sampling_algorithm in ['Nautilus', 'Nautilus-resample', 'emcee',
                                   'dynesty-unif', 'dynesty-rwalk',
                                   'dynesty-slice', 'UltraNest', 'pocoMC']:

            if sampling_algorithm[0].lower() not in args.sampler:
                continue

            if sampling_algorithm[:7] == 'dynesty':

                if sampling_algorithm == 'dynesty-unif' and n_dim > 20:
                    continue

                sample = sampling_algorithm.split('-')[1]

                if sample[0].lower() not in args.dynesty:
                    continue

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

                #if n_dim > 15:
                #    continue
                sampler = ultranest.integrator.NestedSampler(
                    [str(i) for i in range(n_dim)], likelihood, prior)

                result = sampler.run()
                log_z = result['logz']
                n_like = result['ncall']
                points = result['weighted_samples']['points']
                weights = result['weighted_samples']['weights']
                n_eff = np.sum(weights)**2 / np.sum(weights**2)

            elif sampling_algorithm in ['Nautilus', 'Nautilus-resample']:

                if sampling_algorithm == 'Nautilus':
                    sampler = nautilus.Sampler(
                        prior, likelihood, n_dim, pass_struct=False)
                    sampler.run(verbose=args.verbose)
                else:
                    sampler.discard_points()
                    sampler.run(verbose=args.verbose, n_eff=10000)

                log_z = sampler.evidence()
                n_like = sampler.n_like
                n_eff = sampler.effective_sample_size()
                points, log_w, log_l = sampler.posterior()
                weights = np.exp(log_w - np.amax(log_w))

            elif sampling_algorithm == 'emcee':
                if args.likelihood not in ['rosenbrock', 'funnel', 'galaxy',
                                           'exoplanet']:
                    continue
                n_walkers = 100
                if args.likelihood == 'galaxy':
                    n_steps = 10000
                    vectorize = False
                elif args.likelihood == 'exoplanet':
                    n_steps = 25000
                    vectorize = False
                elif args.likelihood == 'funnel':
                    if n_dim <= 10:
                        n_steps = 100000
                    else:
                        n_steps = 1500000
                    vectorize = False
                else:
                    if n_dim <= 10:
                        n_steps = 100000
                    else:
                        n_steps = 1500000
                    vectorize = True

                thin_by = 10
                n_jobs = 1 if not args.multi else cpu_count()

                with Pool(n_jobs) as pool:
                    sampler = emcee.EnsembleSampler(
                        n_walkers, n_dim, likelihood, vectorize=vectorize,
                        pool=pool)
                    sampler.run_mcmc(
                        np.ones((n_walkers, n_dim)) * 0.5 +
                        np.random.normal(size=(n_walkers, n_dim)) * 0.01,
                        n_steps, progress=args.verbose, thin_by=thin_by)

                if args.verbose:
                    print('Mean autocorrelation time: {:.3f} steps'.format(
                          np.mean(sampler.get_autocorr_time())))

                log_z = np.nan
                n_like = n_walkers * n_steps * thin_by
                n_eff = n_like / np.amax(sampler.get_autocorr_time()) / thin_by
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
                    bounds=(0.0, 1.0))
                prior_samples = np.random.uniform(
                    low=0.0, high=1.0, size=(n_particles, n_dim))
                sampler.run(prior_samples, progress=args.verbose)
                sampler.add_samples(9000, progress=args.verbose)

                result = sampler.results
                log_z = result['logz'][-1]
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


if __name__ == '__main__':
    sys.exit(main())

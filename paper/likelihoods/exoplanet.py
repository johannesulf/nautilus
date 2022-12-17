import os
import kepler
import numpy as np
from nautilus import Prior
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous, norm

old_path = os.getcwd()
new_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(new_path)


def half_gauss_pdf(x, sigma):
    return 2 * (sigma * np.sqrt(2 * np.pi))**-1 * np.exp(
        - x**2 / (2 * sigma**2))


def half_gauss_cdf(x, sigma):
    return erf(x / (np.sqrt(2) * sigma))


def rayleigh_pdf(x, sigma):
    return x / sigma**2 * np.exp(- x**2 / (2 * sigma**2))


def rayleigh_cdf(x, sigma):
    return 1 - np.exp(- x**2 / (2 * sigma**2))


class eccentricity_distribution(rv_continuous):

    sigma_gauss = 0.049
    sigma_rayleigh = 0.26
    f = 0.08

    def norm(self):
        return (self.f * rayleigh_cdf(1.0, self.sigma_gauss) + (1 - self.f) *
                half_gauss_cdf(1.0, self.sigma_rayleigh))**-1

    def _pdf(self, x):
        if x < 0 or x >= 1:
            return 0
        return (self.f * rayleigh_pdf(x, self.sigma_rayleigh) + (1 - self.f) *
                half_gauss_pdf(x, self.sigma_gauss)) * self.norm()

    def _cdf(self, x):
        if x < 0:
            return 0
        if x >= 1:
            return 1
        return (self.f * rayleigh_cdf(x, self.sigma_rayleigh) + (1 - self.f) *
                half_gauss_cdf(x, self.sigma_gauss)) * self.norm()


data = np.genfromtxt('california-planet-search.csv')

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
exo_prior.add_parameter('ecc_a', dist=eccentricity_distribution())
exo_prior.add_parameter('ecc_b', dist=eccentricity_distribution())
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


def likelihood(x):

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


os.chdir(old_path)

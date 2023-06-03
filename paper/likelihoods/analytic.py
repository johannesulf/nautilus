import numpy as np
from scipy.special import gamma
from scipy.optimize import rosen
from scipy.stats import norm, multivariate_normal


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


def funnel_likelihood(x):
    if np.any(np.isnan(x)):
        return -1e99
    gamma = 0.95
    log_l_1 = -0.5 * np.log(2 * np.pi) - ((x[..., 0] - 0.5) * 20)**2 / 2.0
    sigma = np.exp((x[0] - 0.5) * 20)
    cov = (np.ones((len(x) - 1, len(x) - 1)) * gamma * sigma +
           np.eye(len(x) - 1) * (1 - gamma) * sigma)
    log_l_2 = multivariate_normal.logpdf((x[1:] - 0.5) * 20, cov=cov)

    log_l = log_l_1 + log_l_2 + len(x) * np.log(20)

    return filter_outside_unit(x, log_l)


def rosenbrock_likelihood(x):

    log_l = -rosen((x.T - 0.5) * 10)

    return filter_outside_unit(x, log_l)

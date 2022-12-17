import os
import numpy as np
from nautilus import Prior
from tabcorr import TabCorr
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import AssembiasZheng07Sats

old_path = os.getcwd()
new_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(new_path)

halotab = TabCorr.read('bolplanck.hdf5')
cens_occ_model = AssembiasZheng07Cens()
sats_occ_model = AssembiasZheng07Sats()
model = HodModelFactory(centrals_occupation=cens_occ_model,
                        satellites_occupation=sats_occ_model)
n_obs = 6.37e-3
n_err = 0.75e-3
wp_obs = np.genfromtxt('wp_dr72_bright0_mr20.0_z0.106_nj400')[:, 1]
wp_cov = np.genfromtxt('wpcov_dr72_bright0_mr20.0_z0.106_nj400')
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


def likelihood(x):

    model.param_dict.update(
        gal_prior.physical_to_structure(gal_prior.unit_to_physical(x)))

    n_mod, wp_mod = halotab.predict(model)

    log_l = -0.5 * (
        (n_mod - n_obs)**2 / n_err**2 +
        np.inner(np.inner(wp_mod - wp_obs, wp_pre), wp_mod - wp_obs))

    return log_l


os.chdir(old_path)

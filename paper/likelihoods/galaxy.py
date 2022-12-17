# The following code is based on the example notebook #4 in the bagpipes
# repository (https://github.com/ACCarnall/bagpipes).

import os
import warnings
import numpy as np
import bagpipes as pipes

old_path = os.getcwd()
new_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(new_path)


def load_goodss(ID):
    """ Load CANDELS GOODS South photometry from the Guo et al. (2013)
    catalogue. """

    # load up the relevant columns from the catalogue.
    cat = np.loadtxt(
        "hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1-1photom_cat.txt",
        usecols=(10, 13, 16, 19, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55,
                 11, 14, 17, 20, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56))

    # Find the correct row for the object we want.
    row = int(ID) - 1

    # Extract the object we want from the catalogue.
    fluxes = cat[row, :15]
    fluxerrs = cat[row, 15:]

    # Turn these into a 2D array.
    photometry = np.c_[fluxes, fluxerrs]

    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i, :] = [0., 9.9*10**99.]

    # Enforce a maximum SNR of 20, or 10 in the IRAC channels.
    for i in range(len(photometry)):
        if i < 10:
            max_snr = 20.

        else:
            max_snr = 10.

        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr

    return photometry


goodss_filt_list = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")

galaxy = pipes.galaxy("2", load_goodss,
                      spectrum_exists=False, filt_list=goodss_filt_list)

dblplaw = {}
dblplaw["tau"] = (0., 15.)
dblplaw["alpha"] = (0.01, 1000.)
dblplaw["beta"] = (0.01, 1000.)
dblplaw["alpha_prior"] = "log_10"
dblplaw["beta_prior"] = "log_10"
dblplaw["massformed"] = (1., 15.)
dblplaw["metallicity"] = (0., 2.5)

dust = {}
dust["type"] = "Calzetti"
dust["Av"] = (0., 2.)

nebular = {}
nebular["logU"] = -3.

fit_info = {}
fit_info["redshift"] = (0., 10.)
fit_info["redshift_prior"] = "Gaussian"
fit_info["redshift_prior_mu"] = 1.0
fit_info["redshift_prior_sigma"] = 0.25
fit_info["dblplaw"] = dblplaw
fit_info["dust"] = dust
fit_info["nebular"] = nebular

fit = pipes.fit(galaxy, fit_info)


def likelihood(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max(fit.fitted_model.lnlike(
            fit.fitted_model.prior.transform(np.copy(x))), -100)

likelihood(np.ones(7) * 0.5)

os.chdir(old_path)

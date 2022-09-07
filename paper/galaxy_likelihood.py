import numpy as np
from astroquery.sdss import SDSS
from prospect.fitting import lnprobfn
from prospect.models import SpecModel
from astropy.coordinates import SkyCoord
from sedpy.observate import load_filters
from prospect.sources import CSPSpecBasis
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary

bands = "ugriz"
mcol = [f"cModelMag_{b}" for b in bands]
ecol = [f"cModelMagErr_{b}" for b in bands]
cat = SDSS.query_crossid(SkyCoord(ra=204.46376, dec=35.79883, unit="deg"),
                         data_release=16,
                         photoobj_fields=mcol + ecol + ["specObjID"])
shdus = SDSS.get_spectra(plate=2101, mjd=53858, fiberID=220)[0]
filters = load_filters([f"sdss_{b}0" for b in bands])
maggies = np.array([10**(-0.4 * cat[0][f"cModelMag_{b}"]) for b in bands])
magerr = np.array([cat[0][f"cModelMagErr_{b}"] for b in bands])
magerr = np.clip(magerr, 0.05, np.inf)

obs = dict(wavelength=None, spectrum=None, unc=None,
           redshift=shdus[2].data[0]["z"], maggies=maggies,
           maggies_unc=magerr * maggies / 1.086, filters=filters)
obs = fix_obs(obs)

model_params = TemplateLibrary["parametric_sfh"]
model_params.update(TemplateLibrary["nebular"])
model_params["zred"]["init"] = obs["redshift"]

model = SpecModel(model_params)
assert len(model.free_params) == 5
noise_model = (None, None)

sps = CSPSpecBasis(zcontinuous=1)


def galaxy_likelihood(x):
    theta = np.zeros_like(x)
    theta[0] = 10**(8 + 4 * x[0])
    theta[1] = -2 + 2.19 * x[1]
    theta[2] = 2 * x[2]
    theta[3] = 0.001 + 13.799 * x[3]
    theta[4] = 10**(np.log10(0.1) + (np.log10(30.0) - np.log10(0.1)) * x[4])
    return lnprobfn(theta, model=model, obs=obs, sps=sps, noise=noise_model,
                    nested=True)

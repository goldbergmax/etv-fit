import types
import numpy as np
import emcee
from isochrones.mist import MIST_Isochrone
from isochrones import StarModel
#from scipy.stats import pearsonr

#chain = np.load('mcmc_out/5095_chains_1.npy')
#mA_mu = chain[:,5000:,11].mean()
#mA_sigma = chain[:,5000:,12].std()
#mB_mu = chain[:,5000:,12].mean()
#mB_sigma = chain[:,5000:,12].std()
#rho, _ = pearsonr(chain[:,5000:,11].flatten(), chain[:,5000:,12].flatten())

mA_mu, mA_sigma = 1.1278, 0.0756
mB_mu, mB_sigma = 1.0786, 0.0663
rho = 0.94504

mist = MIST_Isochrone()
mags = {'G':(13.4394, 0.0002), 'BP':(13.7217, 1/582.46), 'RP':(12.9942, 1/1167.71), 
        'J':(12.499, 0.023), 'H':(12.217, 0.018), 'K':(12.215, 0.024)}
mod = StarModel(mist, **mags, N=2, parallax=(0.8126,0.0148))

def mass_lnprior(p):
    return -1/(2*(1 - rho**2)) * ((p[0] - mA_mu)**2/mA_sigma**2 + (p[1] - mB_mu)**2/mB_sigma**2 
                                  - 2*rho*(p[0] - mA_mu)*(p[1] - mB_mu)/(mA_sigma*mB_sigma))

def lnpost(p):
    return mass_lnprior(p) + mod.lnprior(p) + mod.lnlike(p)

nwalkers = 50
sampler = emcee.EnsembleSampler(nwalkers, 6, lnpost, threads=20)
p0 = np.array([mA_mu, mB_mu, 9.6, 0.0, 1230, 0.3]) \
    + np.array([mA_sigma, mB_sigma, 0.2, 0.05, 10, 0.05])*np.random.normal(size=(nwalkers,6))
sampler.run_mcmc(p0, 10000)

np.save('isochrones_chain.npy', sampler.chain)
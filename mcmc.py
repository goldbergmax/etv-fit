import os
import numpy as np
import emcee
from chisq import EclipseFit
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('system', help='System name')
parser.add_argument('-n', '--nwalkers', type=int, default=100)
parser.add_argument('-s', '--steps', type=int, default=1000)
parser.add_argument('-t', '--threads', type=int, default=1)
args = parser.parse_args()
system = args.system

if not system in ['782', '5095', '3938']:
    raise ValueError('System name "{}" not recognized'.format(system))

seed = {}
spread = {}

seed['782'] = [24.23824, 69.61616, np.radians(90), 0.679, np.radians(59.23),
               991.4, 221.9, -0.198, 0.306, np.radians(81.456), np.radians(-3.38),
               1.281, 1.226, 2.43e-3, 0.067, -16.8]
spread['782'] = [1e-6, 1e-5, np.radians(0.0),  4e-4, np.radians(0.03),
                1, 2, 5e-3, 5e-3, np.radians(4), np.radians(4),
                3e-3, 3e-3, 4e-5, 2e-3, 0.1]

seed['5095'] = [18.61083,  66.86175, np.radians(85.0),  0.48,  np.radians(108.7),
                239.528,  95.537, 0.0376, -0.0401,  np.radians(86.698),  np.radians(-0.698),
                1.104,  1.053,  4.7e-03,  0.0, 4.5, 77.6]
spread['5095'] = [3e-6, 3e-5, np.radians(0.1), 1e-3, np.radians(0.1),
                  1e-3, 0.1, 1e-4, 1e-4, np.radians(0.1), np.radians(0.1),
                  1e-2, 1e-2, 2e-5, 0, 0.05, 0.1]
# high inclination start
# seed['5095'] = [18.61085, 66.86201, np.radians(85.571), 0.486, np.radians(108.2), 
#                 239.49, 95.923, 0.0551*np.cos(np.radians(-46.98)), 0.0551*np.sin(np.radians(-46.98)), np.radians(18.713), np.radians(-103.22),  
#                 1.089, 1.037, 4.801/1047.3, 0.0, 4.515, 77.5]
# spread['5095'] = [3e-5, 5e-5, np.radians(0.0), 0.01, np.radians(0.1),
#                   1e-2, 0.1, 1e-4, 1e-4, np.radians(1), np.radians(1),
#                   0.01, 0.01, 1e-4, 0, 0.0, 0.0]

seed['3938'] = [31.0242673, 60.8408653, np.radians(90),  0.433150, np.radians(-176.02),
                291.8833, 94.72500, 0.099129, -0.023864, np.radians(151.15), np.radians(25.24),
                1.23208, 0.7583, 1.0549e-4, 0.22322, -28.2254]

spread['3938'] = [1e-6, 1e-5, np.radians(0.0),  4e-4, np.radians(0.03),
                  1, 2, 5e-3, 5e-3, np.radians(4), np.radians(4),
                  3e-3, 3e-3, 4e-5, 2e-3, 0.1]

nwalkers = args.nwalkers
ndim = len(seed[system])
p0 = np.array([sigma*np.random.randn(nwalkers) + mu for mu, sigma in zip(seed[system], spread[system])]).T
print('Initial conditions set')

fit = EclipseFit(system)

def im_prior(self, els):
    P1, T01, i1, e1, omega1, P2, Tp2, ecw2, esw2, i2, Omega2, mA, mB, mp, k2, *gamma = els
    im = np.arccos(np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(Omega2))
    n1 = np.arcsin(np.sin(i2)*np.sin(Omega2)/np.sin(im))
    if np.cos(i2) > 0:
        n1 = np.pi - n1
    g1 = omega1 - n1
    if np.abs(im - 90) < 30 and (np.abs(g1 - 90) < 30 or np.abs(g1 + 90) < 30):
        return 0
    else:
        return -np.inf
    
def ecl_5095_prior(self, els):
    P1, T01, i1, e1, omega1, P2, Tp2, ecw2, esw2, i2, Omega2, mA, mB, mp, k2, *gamma = els
    a_bin = (self.G*(mA + mB)*P1/(4*np.pi**2))**(1/3)
    prim_ecl_constraint = (self.R['A'] + self.R['B'])/a_bin * (1 + e1*np.sin(omega1)/(1 - e1**2)) > np.cos(i1)
    sec_ecl_constraint =  (self.R['A'] + self.R['B'])/a_bin * (1 - e1*np.sin(omega1)/(1 - e1**2)) < np.cos(i1)
    return prim_ecl_constraint and sec_ecl_constraint

pool = Pool(args.threads)
sampler = emcee.EnsembleSampler(nwalkers, ndim, fit.evaluate, pool=pool, moves=emcee.moves.RedBlueMove(), kwargs={'constraints': [ecl_5095_prior]})

print('Starting MCMC')
nsteps = args.steps
for sample in sampler.sample(p0, iterations=nsteps, skip_initial_state_check=True):
    if sampler.iteration % 10 == 0:
        print(f'Step {sampler.iteration}')
        print(f'Min chisq: {(-2*sample.log_prob).min():.2f}')

os.makedirs('mcmc_out', exist_ok=True)
np.save(f'mcmc_out/{system}_chains_2.npy', sampler.chain)
np.save(f'mcmc_out/{system}_probs_2.npy', sampler.lnprobability)

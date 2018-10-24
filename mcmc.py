import sys
import numpy as np
import emcee
from chisq import EclipseFit

system = sys.argv[1]

if not system in ['782', '5095']:
    raise ValueError('System name "{}" not recognized'.format(system))

seed = {}
spread = {}

seed['782'] = [24.23824, 69.61616, np.radians(90), 0.679, np.radians(59.23),
               991.4, 221.9, -0.198, 0.306, np.radians(81.456), np.radians(-3.38),
               1.281, 1.226, 2.43e-3, 0.067, -16.8]
spread['782'] = [1e-6, 1e-5, np.radians(0.0),  4e-4, np.radians(0.03),
                1, 2, 5e-3, 5e-3, np.radians(4), np.radians(4),
                3e-3, 3e-3, 4e-5, 2e-3, 0.1]

seed['5095'] = [18.61088,  66.86235, np.radians(87.0),  0.5,  np.radians(109.148),
                239.505,  96.256, 0.0375, -0.041176,  np.radians(86.767),  np.radians(0.047),
                1.0778,  1.0351,  4.4586e-03,  0.0, 4.851, 89.5]
spread['5095'] = [3e-5, 5e-5, np.radians(0.2), 1e-2, np.radians(2),
                  1e-2, 0.2, 1e-3, 1e-3, np.radians(1), np.radians(0.1),
                  1e-1, 1e-1, 2e-4, 0, 0.5, 1]

nwalkers = 100
ndim = len(seed[system])
p0 = np.array([sigma*np.random.randn(nwalkers) + mu for mu, sigma in zip(seed[system], spread[system])]).T
print('Initial conditions set')

#pool = emcee.utils.MPIPool()
#if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

fit = EclipseFit(system)
sampler = emcee.EnsembleSampler(nwalkers, ndim, fit.evaluate, threads=20)

print('Starting MCMC')
nsteps = int(sys.argv[2])
for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
    if (i + 1) % 100 == 0:
        print('Step {}'.format(i+1))
        print('Min chisq: {:.2f}'.format((-2*sampler.lnprobability[:, i]).min()))

#pool.close()

np.save(system + '_chains_1.npy', sampler.chain)
np.save(system + '_probs_1.npy', sampler.lnprobability)

print(sampler.acor)

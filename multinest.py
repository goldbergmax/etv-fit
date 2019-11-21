import sys
import numpy as np
import pymultinest
from chisq import EclipseFit

system = sys.argv[1]

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

seed['5095'] = [18.61085,  66.86201, np.radians(87.0),  0.5,  np.radians(109.148),
                239.505,  96.256, 0.0375, -0.041176,  np.radians(86.767),  np.radians(0.047),
                1.0778,  1.0351,  4.4586e-03,  0.0, 4.851, 77.5]
spread['5095'] = [3e-5, 5e-5, np.radians(0.2), 1e-2, np.radians(2),
                  1e-2, 0.2, 1e-3, 1e-3, np.radians(1), np.radians(0.1),
                  1e-1, 1e-1, 2e-4, 0, 0.5, 1]
# high inclination start
#seed['5095'] = [18.61085, 66.86201, np.radians(85.571), 0.486, np.radians(108.2), 
#                239.49, 95.923, 0.0551*np.cos(np.radians(-46.98)), 0.0551*np.sin(np.radians(-46.98)), np.radians(18.713), np.radians(-103.22),  
#                1.089, 1.037, 4.801/1047.3, 0.0, 4.515, 77.5]
#spread['5095'] = [3e-5, 5e-5, np.radians(0.0), 0.01, np.radians(0.1),
#                  1e-2, 0.1, 1e-4, 1e-4, np.radians(1), np.radians(1),
#                  0.01, 0.01, 1e-4, 0, 0.0, 0.0]

seed['3938'] = [31.0242673, 60.8408653, np.radians(90),  0.433150, np.radians(-176.02),
                291.8833, 94.72500, 0.099129, -0.023864, np.radians(151.15), np.radians(25.24),
                1.23208, 0.7583, 1.0549e-4, 0.22322, -28.2254]

spread['3938'] = [1e-6, 1e-5, np.radians(0.0),  4e-4, np.radians(0.03),
                  1, 2, 5e-3, 5e-3, np.radians(4), np.radians(4),
                  3e-3, 3e-3, 4e-5, 2e-3, 0.1]

def unit_to_unif(x, middle, halfwidth):
    return middle - halfwidth + x*2*halfwidth

def prior(cube, ndim, nparams):
    cube[0] = 0.8 * cube[0]
    cube[1] = 2 * np.pi * cube[1] - np.pi
    cube[2] = unit_to_unif(cube[2], seed[system][5], 10)
    cube[3] = unit_to_unif(cube[3], seed[system][6], 100)
    cube[4] = 0.2*(2*cube[4] - 1)
    cube[5] = 0.2*(2*cube[5] - 1)
    cube[6] = cube[6] * np.pi
    cube[7] = 2 * np.pi * cube[7] - np.pi
    cube[8] = 0.5 + (1.3 - 0.5) * cube[8]
    cube[9] = cube[9] * 25 / 1047.5

fit = EclipseFit(system)

def log_like(cube, ndim, nparams):
    x = np.zeros(17)
    x[0] = seed[system][0]
    x[1] = seed[system][1]
    x[2] = np.radians(85.571)
    x[3:11] = cube[0:8]
    x[11] = 1.21
    x[12:14] = cube[8:10]
    x[14] = 0.0
    x[15] = 0.0
    x[16] = 0.0
    return fit.evaluate(x, rv=False, b=False, linearize=True)

pymultinest.run(log_like, prior, 10, verbose=True)
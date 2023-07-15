import numpy as np
from scipy.optimize import differential_evolution

from chisq import EclipseFit
from cbp_utils import get_i2_Omega2

import multiprocessing

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(arg):
    """
    Like numpy.apply_along_axis(), but and with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = arg
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

fit = EclipseFit('5095')

x = np.array([18.6108787, 66.8618618, np.radians(84.74685), 0.5022,  np.radians(109.249),
              239.52291307, 95.60531312, 0.03818475, -0.03991164, np.radians(88.01798),  np.radians(-0.701487),
              1.17172799, 1.1162874, 0.00479031,  0.0, 4.66971364, 78.00734822])

N = 200
phi = (1 + np.sqrt(5))/2
ims = np.radians(180) * np.sqrt(np.arange(N) + 1/2)/np.sqrt(N + 1/2)
g1s = 2*np.pi*np.arange(N)/phi**2

def combine_params(plan_params, x, angles):
    im, g1 = angles
    y = x.copy()
    i2, Omega2 = get_i2_Omega2(im, g1, x[2], x[4])
    y[9], y[10] = i2, Omega2
    y[5:9] = plan_params[:-1]
    y[13] = plan_params[-1]
    return y

def planet_model(plan_params, x, angles):
    y = combine_params(plan_params, x, angles)
    ecl_model, rv_model = fit.get_residuals(y, safe=False, tFin=1500)
    return ecl_model, rv_model

def planet_chisq(plan_params, x, angles):
    ecl_model, rv_model = planet_model(plan_params, x, angles)
    return fit.get_chisq(ecl_model, rv_model, ecl=True, rv=False, b=False, linearize=True)

def reduced_fit(angles):
    result = differential_evolution(planet_chisq, 
                      args=(x, angles), popsize=5, workers=8,
                      bounds=[(220, 260), (0, 250), (-0.2, 0.2), (-0.2, 0.2), (0., 20/1000.)])
    result['angles'] = angles
    print(result)
    print('\n')
    return result

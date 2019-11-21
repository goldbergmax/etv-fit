import numpy as np
from scipy.optimize import minimize, differential_evolution

from chisq import EclipseFit

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

x = np.array([18.6108497, 66.8620090, np.radians(85.571), 0.486, np.radians(108.2), 
              239.49,  95.923, 3.762e-02, -4.031e-02, np.radians(83.46), np.radians(0.341), 
              1.0887, 1.0372, 4.58337407e-03, 0.0, 4.515, 77.5])

N = 200
phi = (1 + np.sqrt(5))/2
ims = np.radians(180) * np.sqrt(np.arange(N) + 1/2)/np.sqrt(N + 1/2)
g1s = 2*np.pi*np.arange(N)/phi**2

def get_i2_Omega2(im, g1, i1, omega1):
    n1 = np.where(np.sin(omega1 - g1) > 0, omega1 - g1 + np.pi, np.pi + omega1 - g1)
    i2 = np.arccos(np.sin(im)*np.sin(i1)*np.cos(n1) + np.cos(im)*np.cos(i1))
    #i2 = np.where(np.sin(omega1 - g1) > 0, np.pi - i2, i2)
    Omega2 = np.arccos((np.cos(im) - np.cos(i1)*np.cos(i2))/(np.sin(i1)*np.sin(i2)))
    Omega2 = np.where(np.sin(omega1 - g1) > 0, Omega2, 2*np.pi - Omega2)
    return i2, Omega2

def im(i1, i2, Omega2):
    return np.arccos(np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(Omega2))

def n1(i1, i2, Omega2):
    n1 = np.arccos((np.cos(i2) - np.cos(i1)*np.cos(im(i1, i2, Omega2)))/(np.sin(i1)*np.sin(im(i1, i2, Omega2))))
    return np.where(np.sin(Omega2) < 0, n1, np.pi - n1)

def g1(i1, i2, Omega2, omega1):
    g1 = omega1 - n1(i1, i2, Omega2)
    return np.where(np.sin(Omega2) < 0, g1 + np.pi, g1)

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

results = parallel_apply_along_axis(reduced_fit, 0, np.vstack((ims, g1s)))

import pickle

outfile = open('im_g1_fit.data', 'wb')
pickle.dump(results, outfile)
outfile.close()

import numpy as np
from scipy.special import ellipk, ellipkinc

def stab_criterion(mu, e):
    return 1.6 + 5.1 * e - 2.22 * e**2 + 4.12 * mu - 4.27 * e * mu - 5.09 * mu**2 + 4.61 * e**2 * mu**2

def crit_radius(P1, P2, e, mA, mB):
    mu = mB/(mA + mB)
    return (P2/P1)**(2/3) / stab_criterion(mu, e)

# see spherical triangle identities, e.g. in Appendix D of Borkovits (2015)
def im(i1, i2, Omega2):
    return np.arccos(np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(Omega2))

def n1(i1, i2, Omega2):
    n1 = np.arccos((np.cos(i2) - np.cos(i1)*np.cos(im(i1, i2, Omega2)))/(np.sin(i1)*np.sin(im(i1, i2, Omega2))))
    return np.where(np.sin(Omega2) < 0, n1, np.pi - n1)

def g1(i1, i2, Omega2, omega1):
    g1 = omega1 - n1(i1, i2, Omega2)
    return np.where(np.sin(Omega2) < 0, g1 + np.pi, g1)

def get_i2_Omega2(im, g1, i1, omega1):
    n1 = np.where(np.sin(omega1 - g1) > 0, omega1 - g1 + np.pi, np.pi + omega1 - g1)
    i2 = np.arccos(np.sin(im)*np.sin(i1)*np.cos(n1) + np.cos(im)*np.cos(i1))
    #i2 = np.where(np.sin(omega1 - g1) > 0, np.pi - i2, i2)
    Omega2 = np.arccos((np.cos(im) - np.cos(i1)*np.cos(i2))/(np.sin(i1)*np.sin(i2)))
    Omega2 = np.where(np.sin(omega1 - g1) > 0, Omega2, 2*np.pi - Omega2)
    return i2, Omega2

def h(im, g1, e1):
    return np.cos(im)**2 - e1**2/2*np.sin(im)**2 * (3 - 5*np.cos(2*g1))

def k_sq(e1, h):
    return 5*e1**2/(1 - e1**2)*(1 - h)/(h + 4*e1**2)

def K(k_sq_):
    return np.where(k_sq_ < 1, ellipk(k_sq_), ellipkinc(np.arcsin(1/np.sqrt(k_sq_)), k_sq_))

# Farago & Laskar (2010) Eq. 2.32
def prec_timescale(P1, P2, e1, e2, im, g1, mA, mB):
    h_ = h(im, g1, e1)
    k_sq_ = k_sq(e1, h_)
    return 8/(3*np.pi) * (mA + mB)**2/(mA * mB) * (P2**7/P1**4)**(1/3) * K(k_sq_) * \
            (1 - e2**2)**2/np.sqrt((1 - e1**2) * (h_ + 4*e1**2))
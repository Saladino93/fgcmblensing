import numpy as np
from scipy import integrate as sinteg

import integrated_bispectrum

from NumbaQuadpack import quadpack_sig, dqags
import numba as nb

from numba import jit

import itertools

chi_precalculated, z_precalculated = np.loadtxt("products/zs.txt", unpack = True)
chi_kk, Wkk = np.loadtxt("products/Wkk.txt", unpack = True)

@jit
def zofchi_numba(newchis):
    return np.interp(newchis, chi_precalculated, z_precalculated)

@jit
def Wkk_numba(newchis):
    return np.interp(newchis, chi_kk, Wkk)

@jit
def integrate_numba(y, x):
    return np.trapz(y, x)


#avoid recomputing the same things
#prefactors or memoization?
def bispectrum_matter(k1, k2, k3, theta12, theta13, theta23, z, model = 'TR'):
    ksvec = [k1, k2, k3]
    combinations = list(itertools.combinations([0,1,2], 2))
    thetas = [theta12, theta13, theta23] #assume this is the order too from combinations
    return sum([2*F2ptker_vector(ksvec[comb[0]], ksvec[comb[1]], thetaij, z, model = model)*P(z, ksvec[comb[0]], grid = False)*P(z, ksvec[comb[1]], grid = False) for comb, thetaij in zip(combinations, thetas)])

@np.vectorize
def integrate_bispectrum_kkk_prova(l, angle, model = 'TR'):
    @nb.cfunc(quadpack_sig)
    def bispectrum_at_ells_of_chi(chi, data): 
        return chi**(-4)*Wkk_numba(chi)**3*integrated_bispectrum.P(1, 1)#*integrated_bispectrum.bispectrum_matter(l/chi, l/chi, l/chi, angle, angle, angle, zofchi_numba(chi), model = model)
    funcptr = bispectrum_at_ells_of_chi.address
    sol, abserr, success = dqags(funcptr, 0, integrated_bispectrum.chistar)
    return sol #sinteg.quadrature(bispectrum_at_ells_of_chi, 0, integrated_bispectrum.chistar, maxiter = 50, rtol = 1e-8)[0]

integrate_bispectrum_kkk_prova(100, np.pi/3)
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate as sinteg

import integrated_bispectrum

import vegas

import time

@np.vectorize
def integrate_bispectrum_kkk(l1, l2, l3, angle12, angle13, angle23, model = 'TR'):
    bispectrum_at_ells_of_chi = lambda chi: chi**(-4)*integrated_bispectrum.Wkk(chi)**3*integrated_bispectrum.bispectrum_matter(l1/chi, l2/chi, l3/chi, angle12, angle13, angle23, integrated_bispectrum.zofchi(chi), model = model)
    return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, integrated_bispectrum.chistar, maxiter = 50, rtol = 1e-8)[0]


def get_angle_12(L1, L2, L3):
        term = (L1**2+L2**2-L3**2)/(2*L1*L2)
        return np.arccos(term)




def get_integrand(WR, model):
    @vegas.batchintegrand
    def integrand(x):
        l, thetal = x[:, 0], x[:, 1]
        L, thetaL = x[:, 2], x[:, 3]

        lx, ly = np.cos(thetal)*l, np.sin(thetal)*l
        Lplx = L*np.cos(thetaL)+lx
        Lply = L*np.sin(thetaL)+ly
        Lplv = np.array([Lplx, Lply])
        Lpl = np.linalg.norm(Lplv, axis = 0)
        angle12 = get_angle_12(l, L, Lpl)
        angle13 = get_angle_12(l, Lpl, L)
        angle23 = get_angle_12(L, Lpl, l)
        value = L*l*WR(L)*WR(l)*WR(Lpl)*integrate_bispectrum_kkk(l, L, Lpl, angle12, angle13, angle23, model = model)
        return value/(2*np.pi)**2/(2*np.pi)**2
    return integrand


models = ['TR', 'GM', 'SC']
Rs = np.linspace(0.1, 6, 10)

R, model = 0.1, 'TR'

results = {}

s = time.time()
Rdeg = R/60
Rradians = np.deg2rad(Rdeg)
sigma = Rradians / (2.0 * np.sqrt(2.0 * np.log(2.0)))
WR = lambda l: np.exp(-(l*(l+1))/2*sigma**2)
#WR = lambda l: np.exp(-l**2*Rradians**2/2)

integrand = get_integrand(WR, model)


def main():

    lmin, lmax = 1, 4000

    integ = vegas.Integrator([[lmin, lmax], [0, 2*np.pi], [lmin, lmax], [0, 2*np.pi]], nproc = 4)
    nitn, neval = 10, 1e2

    values = []

    result = integ(integrand, nitn = nitn, neval = neval)
    values += [result.mean]
            

    print("Total time", time.time()-s)

if __name__ == '__main__':
    main()
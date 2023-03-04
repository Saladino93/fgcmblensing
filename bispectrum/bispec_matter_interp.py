import numpy as np

import numba 














def integrate_bispectrum_kkk(l1, l2, l3, angle12, angle13, angle23, model = 'TR'):
    bispectrum_at_ells_of_chi = lambda chi: chi**(-4)*Wkk_numba(chi)**3*integrated_bispectrum.bispectrum_matter(l1/chi, l2/chi, l3/chi, angle12, angle13, angle23, zofchi_numba(chi), model = model)
    #value = bispectrum_at_ells_of_chi(precomputed_chis)
    return integrate_numba(bispectrum_at_ells_of_chi(precomputed_chis), precomputed_chis)
    #return simpson(value, precomputed_chis)
    #return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, integrated_bispectrum.chistar, maxiter = 50, rtol = 1e-8)[0]
    #return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, integrated_bispectrum.chistar, maxiter = 50, rtol = 1e-8)[0]
    #return np.dot(dchis, bispectrum_at_ells_of_chi(precomputed_chis))
integrate_bispectrum_kkk_alt = np.vectorize(integrate_bispectrum_kkk_alt_, excluded = ['model'])
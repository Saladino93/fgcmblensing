import interpolated_quantities_numba as iqn

from numba import jit, prange, vectorize, float64, int64, types

import numpy as np

import itertools

import scipy.integrate as sinteg

"""
Want to try to see if I can compile with , as it seems nested functions are not that easy to implement.

@jit(nopython = True)
def getSCcoeffs():
    a1 = 0.250
    a2 = 3.50 
    a3 = 2.00 
    a4 = 1.00 
    a5 = 2.00
    a6 = -0.200 
    a7 = 1.00 
    a8 = 0.00 
    a9 = 0.00
    return a1, a2, a3, a4, a5, a6, a7, a8, a9

@jit(nopython = True)
def getGMcoeffs():
    a1 = 0.484
    a2 = 3.74
    a3 = -0.849
    a4 = 0.392
    a5 = 1.01
    a6 = -0.575
    a7 = 0.128
    a8 = -0.722
    a9 = -0.926
    return a1, a2, a3, a4, a5, a6, a7, a8, a9

@jit(nopython = True)
def get_coeffs(model):
    if model == 'GM':
        return getGMcoeffs()
    elif model == 'SC':
        return getSCcoeffs()

Q = lambda x: (4-2**x)/(1+2**(x+1))

def get_afuncs_form_coeffs_for_interp(model):
    
    a1_, a2_, a3_, a4_, a5_, a6_, a7_, a8_, a9_ = get_coeffs(model)

    def afunc(z, k, a1 = a1_, a2 = a2_, a6 = a6_): 
        return (1+s8(z)**a6*np.sqrt(0.7*Q(nefff(z, k)))*(k/kNLzf(z)*a1)**(nefff(z, k)+a2))/(1+(a1*k/kNLzf(z))**(nefff(z, k)+a2))
    
    def bfunc(z, k, a3 = a3_, a7 = a7_, a8 = a8_): 
        return (1+0.2*a3*(nefff(z, k)+3)*(k/kNLzf(z)*a7)**(nefff(z, k)+3+a8))/(1+(a7*k/kNLzf(z))**(nefff(z, k)+3.5+a8))
    
    def cfunc(z, k, a4 = a4_, a5 = a5_, a9 = a9_): 
        return (1+(4.5*a4/(1.5+(nefff(z, k)+3)**4)))*(k/kNLzf(z)*a5)**(nefff(z, k)+3+a9)/(1+(a5*k/kNLzf(z))**(nefff(z, k)+3.5+a9))
    return [afunc, bfunc, cfunc]

fit_funcs = {}
for model in ['GM', 'SC']:
    values_list = [fun(10**iqn.grid2d[:,0], 10**iqn.grid2d[:,1]).reshape((iqn.NN, iqn.NN)) for fun in get_afuncs_form_coeffs_for_interp(model)]
    fit_funcs[model] = [lambda z, k, i = i: iqn.eval_linear(iqn.grid, values_list[i], np.log10(np.array([z, k])).T) for i in range(len(values_list))]

def get_afuncs_form_coeffs(model):
    return fit_funcs[model]

afuncTR, bfuncTR, cfuncTR = [lambda k, z: np.ones_like(k)]*3

@jit(nopython = True)
def getfuncs(model):
    #if model == 'TR':
    return afuncTR, bfuncTR, cfuncTR
    #elif model in ['GM', 'SC']:
    #    return get_afuncs_form_coeffs(model)



    """
"""
#@jit(nopython = True)
def F2ptker_vector(k1, k2, theta12, z, model = 'TR'):

    Calculates F2 kernel from PT.

    Parameters
    ----------
    k1vec : array_like
        3-vector of k1, shape, (3, n), where n is the number of points to be calculated at.
    k2vec : array_like
        3-vector of k2, shape, (3, n), where n is the number of points to be calculated at.
    z : array_like
        Redshifts, shape, (1, n).


    afunc, bfunc, cfunc = getfuncs(model)
    resultG = 5/7*afunc(z, k1)*afunc(z, k2)
    resultS = bfunc(z, k1)*bfunc(z, k2)*1/2*(k1/k2 + k2/k1)*np.cos(theta12)
    resultT = 2/7*(np.cos(theta12))**2*cfunc(z, k1)*cfunc(z, k2)
    return resultG + resultS + resultT
"""


@jit(nopython = True)
def getGMcoeffs():
    a1 = 0.484
    a2 = 3.74
    a3 = -0.849
    a4 = 0.392
    a5 = 1.01
    a6 = -0.575
    a7 = 0.128
    a8 = -0.722
    a9 = -0.926
    return a1, a2, a3, a4, a5, a6, a7, a8, a9



@jit(nopython = True)
def getSCcoeffs():
    a1 = 0.250
    a2 = 3.50 
    a3 = 2.00 
    a4 = 1.00 
    a5 = 2.00
    a6 = -0.200 
    a7 = 1.00 
    a8 = 0.00 
    a9 = 0.00
    return a1, a2, a3, a4, a5, a6, a7, a8, a9



@jit(nopython = True)
def afunc(z, k, a1, a2, a6): 
    ncalc = iqn.nefff(z, k)
    kNLcalc = iqn.kNLzf(z)
    return (1+iqn.s8(z)**a6*np.sqrt(0.7*iqn.Q(ncalc))*(k/kNLcalc*a1)**(ncalc+a2))/(1+(a1*k/kNLcalc)**(ncalc+a2))

@jit(nopython = True)
def afuncGM(z, k): 
    a1GM, a2GM, a6GM = 0.484, 3.74, -0.575
    return afunc(z, k, a1GM, a2GM, a6GM)

@jit(nopython = True)
def afuncSC(z, k): 
    a1SC, a2SC, a6SC = 0.250, 3.50 , -0.200 
    return afunc(z, k, a1SC, a2SC, a6SC)

@jit(nopython = True)
def bfunc(z, k, a3, a7, a8): 
    ncalc = iqn.nefff(z, k)
    kNLcalc = iqn.kNLzf(z)
    return (1+0.2*a3*(ncalc+3)*(k/kNLcalc*a7)**(ncalc+3+a8))/(1+(a7*k/kNLcalc)**(ncalc+3.5+a8))

@jit(nopython = True)
def bfuncGM(z, k):
    a3GM, a7GM, a8GM = -0.849, 0.128, -0.722
    return bfunc(z, k, a3GM, a7GM, a8GM)



@jit(nopython = True)
def bfuncSC(z, k):
    a3SC, a7SC, a8SC = 2.00 , 1.00, 0.00 
    return bfunc(z, k, a3SC, a7SC, a8SC)

@jit(nopython = True)
def cfunc(z, k, a4, a5, a9):
    ncalc = iqn.nefff(z, k)
    kNLcalc = iqn.kNLzf(z)
    return (1+(4.5*a4/(1.5+(ncalc+3)**4)))*(k/kNLcalc*a5)**(ncalc+3+a9)/(1+(a5*k/kNLcalc)**(ncalc+3.5+a9))

@jit(nopython = True)
def cfuncGM(z, k):
    a4GM, a5GM, a9GM = 0.392, 1.01, -0.926
    return cfunc(z, k, a4GM, a5GM, a9GM)

@jit(nopython = True)
def cfuncSC(z, k):
    a4SC, a5SC, a9SC = 1.00, 2.00, 0.00
    return cfunc(z, k, a4SC, a5SC, a9SC)

##create meshgrid in z, k to interpolate over a, b, c function of GM model


@jit(nopython = True, cache = True)
def F2ptker_vector_GM(k1, k2, theta12, z):
    resultG = 5/7*afuncGM(z, k1)*afuncGM(z, k2)
    resultS = 1/2*(k1/k2 + k2/k1)*np.cos(theta12)*bfuncGM(z, k1)*bfuncGM(z, k2)
    resultT = 2/7*(np.cos(theta12))**2*cfuncGM(z, k1)*cfuncGM(z, k2)
    return resultG + resultS + resultT

@jit(nopython = True, cache = True)
def F2ptker_vector_cos_GM(k1, k2, costheta12, z):
    resultG = 5/7*afuncGM(z, k1)*afuncGM(z, k2)
    resultS = 1/2*(k1/k2 + k2/k1)*costheta12*bfuncGM(z, k1)*bfuncGM(z, k2)
    resultT = 2/7*(costheta12)**2*cfuncGM(z, k1)*cfuncGM(z, k2)
    return resultG + resultS + resultT

@jit(nopython = True, cache = True)
def F2ptker_vector_cos_SC(k1, k2, costheta12, z):
    resultG = 5/7*afuncSC(z, k1)*afuncSC(z, k2)
    resultS = 1/2*(k1/k2 + k2/k1)*costheta12*bfuncSC(z, k1)*bfuncSC(z, k2)
    resultT = 2/7*(costheta12)**2*cfuncSC(z, k1)*cfuncSC(z, k2)
    return resultG + resultS + resultT


@jit(nopython = True, cache = True)
def F2ptker_vector_TR(k1, k2, theta12, z):
    resultG = 5/7
    resultS = 1/2*(k1/k2 + k2/k1)*np.cos(theta12)
    resultT = 2/7*(np.cos(theta12))**2
    return resultG + resultS + resultT

@jit(nopython = True, cache = True)
def F2ptker_vector_cos_TR(k1, k2, costheta12, z):
    resultG = 5/7
    resultS = 1/2*(k1/k2 + k2/k1)*costheta12
    resultT = 2/7*(costheta12)**2
    return resultG + resultS + resultT


combinations = list(itertools.combinations([0,1,2], 2))

@jit(nopython = True)
def allowed(k):
    return (k >= 1e-12) & (k <= 50)

@jit(nopython = True)
def bispectrum_matter_TR(k1, k2, k3, theta12, theta13, theta23, z):
    somma = 2*F2ptker_vector_TR(k1, k2, theta12, z)*iqn.P2D(z, k1)*iqn.P2D(z, k2)
    somma += 2*F2ptker_vector_TR(k1, k3, theta13, z)*iqn.P2D(z, k1)*iqn.P2D(z, k3)
    somma += 2*F2ptker_vector_TR(k2, k3, theta23, z)*iqn.P2D(z, k2)*iqn.P2D(z, k3)
    return somma


@jit(nopython = True)
def bispectrum_matter_cos_TR(k1, k2, k3, ctheta12, ctheta13, ctheta23, z):
    somma = 2*F2ptker_vector_cos_TR(k1, k2, ctheta12, z)*iqn.P2D(z, k1)*iqn.P2D(z, k2)
    somma += 2*F2ptker_vector_cos_TR(k1, k3, ctheta13, z)*iqn.P2D(z, k1)*iqn.P2D(z, k3)
    somma += 2*F2ptker_vector_cos_TR(k2, k3, ctheta23, z)*iqn.P2D(z, k2)*iqn.P2D(z, k3)
    return somma*allowed(k1)*allowed(k2)*allowed(k3)


@jit(nopython = True)
def bispectrum_matter_cos_general(k1, k2, k3, ctheta12, ctheta13, ctheta23, z, model):

    if model == 'GM':
        A12 = F2ptker_vector_cos_GM(k1, k2, ctheta12, z)
        A13 = F2ptker_vector_cos_GM(k1, k3, ctheta13, z)
        A23 = F2ptker_vector_cos_GM(k2, k3, ctheta23, z)
    elif model == 'SC':
        A12 = F2ptker_vector_cos_SC(k1, k2, ctheta12, z)
        A13 = F2ptker_vector_cos_SC(k1, k3, ctheta13, z)
        A23 = F2ptker_vector_cos_SC(k2, k3, ctheta23, z)
    elif model == 'TR':
        A12 = F2ptker_vector_cos_TR(k1, k2, ctheta12, z)
        A13 = F2ptker_vector_cos_TR(k1, k3, ctheta13, z)
        A23 = F2ptker_vector_cos_TR(k2, k3, ctheta23, z)
    
    somma = 2*A12*iqn.P2D(z, k1)*iqn.P2D(z, k2)
    somma += 2*A13*iqn.P2D(z, k1)*iqn.P2D(z, k3)
    somma += 2*A23*iqn.P2D(z, k2)*iqn.P2D(z, k3)

    return somma*allowed(k1)*allowed(k2)*allowed(k3)


@jit(nopython = True)
def allowedks(k, kmin, kmax):
    return (k >= kmin) & (k <= kmax)

@jit(nopython = True)
def bispectrum_matter_cos_TR_for_k(k1, k2, k3, ctheta12, ctheta13, ctheta23, z, kmin, kmax):
    somma = 2*F2ptker_vector_cos_TR(k1, k2, ctheta12, z)*iqn.P2D(z, k1)*iqn.P2D(z, k2)
    somma += 2*F2ptker_vector_cos_TR(k1, k3, ctheta13, z)*iqn.P2D(z, k1)*iqn.P2D(z, k3)
    somma += 2*F2ptker_vector_cos_TR(k2, k3, ctheta23, z)*iqn.P2D(z, k2)*iqn.P2D(z, k3)
    return somma*allowedks(k1, kmin, kmax)*allowedks(k2, kmin, kmax)*allowedks(k3, kmin, kmax)

@jit(nopython = True)
def bispectrum_matter_GM(k1, k2, k3, theta12, theta13, theta23, z):
    somma = 2*F2ptker_vector_GM(k1, k2, theta12, z)*iqn.P2D(z, k1)*iqn.P2D(z, k2)
    somma += 2*F2ptker_vector_GM(k1, k3, theta13, z)*iqn.P2D(z, k1)*iqn.P2D(z, k3)
    somma += 2*F2ptker_vector_GM(k2, k3, theta23, z)*iqn.P2D(z, k2)*iqn.P2D(z, k3)
    return somma

@jit(nopython = True)
def bispectrum_matter_cos_GM(k1, k2, k3, ctheta12, ctheta13, ctheta23, z):
    somma = 2*F2ptker_vector_cos_GM(k1, k2, ctheta12, z)*iqn.P2D(z, k1)*iqn.P2D(z, k2)
    somma += 2*F2ptker_vector_cos_GM(k1, k3, ctheta13, z)*iqn.P2D(z, k1)*iqn.P2D(z, k3)
    somma += 2*F2ptker_vector_cos_GM(k2, k3, ctheta23, z)*iqn.P2D(z, k2)*iqn.P2D(z, k3)
    return somma*allowed(k1)*allowed(k2)*allowed(k3)




chistar = 13858.934986501856
@np.vectorize
def integrate_bispectrum_kkk_TR_scipy(l1, l2, l3, angle12, angle13, angle23):
    @np.vectorize
    def bispectrum_at_ells_of_chi(chi): 
        return chi**(-4)*iqn.Wkk(chi)**3*bispectrum_matter_TR(l1/chi, l2/chi, l3/chi, angle12, angle13, angle23, iqn.zofchi(chi))
    return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, chistar, miniter = 10, maxiter = 50, rtol = 1e-8)[0]


#get points and weights for Gaussian quadrature using numpy legendre module
def gaussxw(a, b, N):
    x, w = np.polynomial.legendre.leggauss(N)
    return 0.5*(b-a)*x + 0.5*(b+a), 0.5*(b-a)*w

Nquadrature = 20
xsgauss, wsgauss = gaussxw(0, chistar, Nquadrature)

@np.vectorize
def integrate_bispectrum_kkk_TR_gauss(l1, l2, l3, angle12, angle13, angle23):
    @np.vectorize
    def bispectrum_at_ells_of_chi(chi): 
        return chi**(-4)*iqn.Wkk(chi)**3*bispectrum_matter_TR(l1/chi, l2/chi, l3/chi, angle12, angle13, angle23, iqn.zofchi(chi))
    return np.dot(bispectrum_at_ells_of_chi(xsgauss), wsgauss)

@jit(nopython = True, fastmath = True)
def get_angle_12(L1, L2, L3):
    term = (L1**2+L2**2-L3**2)/(2*L1*L2)
    return np.arccos(term)

@jit(nopython = True, fastmath = True)
def get_angle_cos12(L1, L2, L3):
    return (L1**2+L2**2-L3**2)/(2*L1*L2)

def integrate_bispectrum_kkk_TR_gauss_from_triangle(l1, l2, l3):
    @np.vectorize
    def bispectrum_at_ells_of_chi(chi): 
        angle12, angle13, angle23 = get_angle_12(l1, l2, l3), get_angle_12(l1, l3, l2), get_angle_12(l2, l3, l1)
        return chi**(-4)*iqn.Wkk(chi)**3*bispectrum_matter_TR(l1/chi, l2/chi, l3/chi, angle12, angle13, angle23, iqn.zofchi(chi))
    return np.dot(bispectrum_at_ells_of_chi(xsgauss), wsgauss)



chipow_4_times_Wkk3_pre_calc = xsgauss**(-4)*iqn.Wkk(xsgauss)**3


@np.vectorize
def integrate_bispectrum_kkk_TR_cos_scipy_from_triangle(l1, l2, l3):
    @np.vectorize
    def bispectrum_at_ells_of_chi(chi): 
        cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
        return chi**(-4)*iqn.Wkk(chi)**3*bispectrum_matter_cos_TR(l1/chi, l2/chi, l3/chi, cangle12, cangle13, cangle23, iqn.zofchi(chi))
    return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, chistar, miniter = 10, maxiter = 50, rtol = 1e-12)[0]

@np.vectorize
def integrate_bispectrum_kkk_TR_cos_gauss_from_triangle(l1, l2, l3):
    @np.vectorize
    def bispectrum_at_ells_of_chi(chi): 
        cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
        return bispectrum_matter_cos_TR(l1/chi, l2/chi, l3/chi, cangle12, cangle13, cangle23, iqn.zofchi(chi))
    return np.dot(bispectrum_at_ells_of_chi(xsgauss)*chipow_4_times_Wkk3_pre_calc, wsgauss)



#@vectorize([float64(float64, float64, float64)])
@jit(nopython = True, fastmath = True, parallel = True)
def bispecTR(l1, l2, l3):
    cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
    bispec_arr = np.empty(xsgauss.size)
    #for i, x in enumerate(xsgauss):
    for i in prange(xsgauss.size):
        x = xsgauss[i]
        bispec_arr[i] = bispectrum_matter_cos_TR(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x))
    somma = np.dot(chipow_4_times_Wkk3_pre_calc*bispec_arr, wsgauss)
    return somma


@jit(nopython = True, fastmath = True, parallel = True)
def bispec_general(l1, l2, l3, model):
    cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
    bispec_arr = np.empty(xsgauss.size)
    #for i, x in enumerate(xsgauss):
    for i in prange(xsgauss.size):
        x = xsgauss[i]
        bispec_arr[i] = bispectrum_matter_cos_general(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x), model)
    somma = np.dot(chipow_4_times_Wkk3_pre_calc*bispec_arr, wsgauss)
    return somma


@jit(nopython = True, fastmath = True, parallel = True)
def bispecTRfork(l1, l2, l3, kmin, kmax):
    cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
    bispec_arr = np.empty(xsgauss.size)
    #for i, x in enumerate(xsgauss):
    for i in prange(xsgauss.size):
        x = xsgauss[i]
        bispec_arr[i] = bispectrum_matter_cos_TR_for_k(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x), kmin, kmax)
    somma = np.dot(chipow_4_times_Wkk3_pre_calc*bispec_arr, wsgauss)
    return somma

@jit(nopython = True, fastmath = True, parallel = True)
def bispecGM(l1, l2, l3):
    cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
    bispec_arr = np.empty(xsgauss.size)
    #for i, x in enumerate(xsgauss):
    for i in prange(xsgauss.size):
        x = xsgauss[i]
        bispec_arr[i] = bispectrum_matter_cos_GM(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x))
    somma = np.dot(chipow_4_times_Wkk3_pre_calc*bispec_arr, wsgauss)
    return somma


@jit(nopython = True, fastmath = True)
def bispec_phi_TR_non_vec(l1, l2, l3):
    factor = 8/(l1**2*l2**2*l3**2)
    return bispecTR(l1, l2, l3)*factor


@jit(nopython = True, fastmath = True)
def bispec_phi_TR_non_vec_for_k(l1, l2, l3, kmin, kmax):
    factor = 8/(l1**2*l2**2*l3**2)
    return bispecTRfork(l1, l2, l3, kmin, kmax)*factor



@vectorize([float64(float64, float64, float64)])
@jit(nopython = True, fastmath = True)
def bispec_phi_TR(l1, l2, l3):
    factor = 8/(l1**2*l2**2*l3**2)
    return bispecTR(l1, l2, l3)*factor


@vectorize([float64(float64, float64, float64)])
@jit(nopython = True, fastmath = True, parallel = True)
def bispec_phi_GM(l1, l2, l3):
    factor = 8/(l1**2*l2**2*l3**2)
    return bispecGM(l1, l2, l3)*factor


@vectorize([float64(float64, float64, float64, int64)])
@jit(nopython = True, fastmath = True, parallel = True)
def bispec_phi_general(l1, l2, l3, modelint):
    factor = 8/(l1**2*l2**2*l3**2)
    if modelint == 0:
        model = "TR"
    elif modelint == 1:
        model = "SC"
    elif modelint == 2:
        model = "GM"
    return bispec_general(l1, l2, l3, model)*factor

@jit(nopython = True, fastmath = True)
def bispec_phi_general_non_vec(l1, l2, l3, modelint):
    factor = 8/(l1**2*l2**2*l3**2)
    if modelint == 0:
        model = "TR"
    elif modelint == 1:
        model = "SC"
    elif modelint == 2:
        model = "GM"
    return bispec_general(l1, l2, l3, model)*factor


vectorize([float64(float64, float64, float64)])
jit(nopython = True, fastmath = True)
def bispec_check_quadrature(l1, l2, l3, Nquadrature):
    
    xsgauss, wsgauss = gaussxw(1e-5, chistar, Nquadrature)

    chipow_4_times_Wkk3_pre_calc = xsgauss**(-4)*iqn.Wkk(xsgauss)**3

    cangle12, cangle13, cangle23 = get_angle_cos12(l1, l2, l3), get_angle_cos12(l1, l3, l2), get_angle_cos12(l2, l3, l1)
    bispec_arr = np.empty(xsgauss.size)
    #for i, x in enumerate(xsgauss):
    for i in prange(xsgauss.size):
        x = xsgauss[i]
        bispec_arr[i] = bispectrum_matter_cos_TR(l1/x, l2/x, l3/x, cangle12, cangle13, cangle23, iqn.zofchi(x))
    return np.dot(chipow_4_times_Wkk3_pre_calc*bispec_arr, wsgauss)*8/(l1**2*l2**2*l3**2)
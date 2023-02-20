import camb
from camb import model as cmodel
import numpy as np
from scipy import interpolate as interp, integrate as sinteg
import itertools
from angularcls import windows, cosmoconstants


nz = 6000 #number of steps to use for the radial/redshift integration
kmax = 100  #kmax to use
#First set up parameters as usual

"""
pars = camb.CAMBparams()

H0 = 67.1
Omegam = 0.315
ommh2 = Omegam*H0**2/100**2
Omegab = 0.049
ombh2 = Omegab*H0**2/100**2
omch2 = ommh2 - ombh2
Omegam = ommh2/(H0/100)**2
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(As = 2.215*1e-9, ns=0.968)
"""

pars = camb.CAMBparams()
H0 = 67
ombh2 = 0.022445
omch2 = 0.1212
ommh2 = ombh2+omch2
h = H0/100
Omegam = ommh2/h**2
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0)
pars.InitPower.set_params(As=2.1265e-09, ns=0.96)

# reionization and recombination 
pars.Reion.use_optical_depth = True
pars.Reion.optical_depth = 0.0925
pars.Reion.delta_redshift = 0.5
pars.Recomb.RECFAST_fudge = 1.14

#non linearity
pars.NonLinear = cmodel.NonLinear_both
pars.NonLinearModel.halofit_version = 'takahashi'
pars.Accuracy.AccurateBB = True #need this to avoid small-scale ringing

#For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
#so get background results to find chistar, set up a range in chi, and calculate corresponding redshifts
results= camb.get_background(pars)
chistar = results.conformal_time(0)- results.tau_maxvis
chis = np.linspace(0, chistar, nz)
zs = results.redshift_at_comoving_radial_distance(chis)

Hzs = results.hubble_parameter(zs)

PK = camb.get_matter_power_interpolator(pars, nonlinear = True, 
    hubble_units = False, k_hunit = False, kmax = kmax, k_per_logint = None,
    var1 = 'delta_nonu', var2 = 'delta_nonu', zmax = zs[-1])

PKlin = camb.get_matter_power_interpolator(pars, nonlinear = False, 
    hubble_units = False, k_hunit = False, kmax = kmax, k_per_logint = None,
    var1 = 'delta_nonu', var2 = 'delta_nonu', zmax = zs[-1])

zm = np.logspace(-9, np.log(2000), 100)
zm = np.append(0, zm)
pars.set_matter_power(redshifts = zm, kmax = 1)

results = camb.get_results(pars)

s8 = np.array(results.get_sigma8())

Q = lambda x: (4-2**x)/(1+2**(x+1))
 
nonlinearscale = lambda z, k: 4*np.pi*k**3*PKlin.P(z, k)-1

#find root of nonlinearscale(k) = 0 with scipy
from scipy import optimize
kNLz = []
for z in zs:
    nonlinearscale_ = lambda k: nonlinearscale(z, k)
    kstar = optimize.brentq(nonlinearscale_, 1e-5, 1e5)
    kNLz.append(kstar)

kNLzf = interp.interp1d(zs, kNLz, kind = 'cubic', fill_value = 'extrapolate')

import findiff
kgrid = np.log(np.arange(1e-5, 1e5, 1e-2))
Pgrid = np.log(PKlin.P(0, np.exp(kgrid)))
dkgrid = kgrid[1]-kgrid[0]
d2_dx1 = findiff.FinDiff(0, dkgrid, 1)

neff = d2_dx1(Pgrid)
nefff = interp.interp1d(np.exp(kgrid), neff, kind = 'cubic', fill_value = 'extrapolate')

#zm, s8 = np.loadtxt('sigma8.txt', unpack = True)
s8 = interp.interp1d(zm, s8, kind = 'cubic', fill_value = 'extrapolate')

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

def get_coeffs(model):
    if model == 'GM':
        return getGMcoeffs()
    elif model == 'SC':
        return getSCcoeffs()


def get_afuncs_form_coeffs_for_interp(model):
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = get_coeffs(model)
    afuncGM = lambda z, k: (1+s8(z)**a6*np.sqrt(0.7*Q(nefff(k)))*(k/kNLzf(z)*a1)**(nefff(k)+a2))/(1+(a1*k/kNLzf(z))**(nefff(k)+a2))
    bfuncGM = lambda z, k: (1+0.2*a3*(nefff(k)+3)*(k/kNLzf(z)*a7)**(nefff(k)+3+a8))/(1+(a7*k/kNLzf(z))**(nefff(k)+3.5+a8))
    cfuncGM = lambda z, k: (1+(4.5*a4/(1.5+(nefff(k)+3)**4)))*(k/kNLzf(z)*a5)**(nefff(k)+3+a9)/(1+(a5*k/kNLzf(z))**(nefff(k)+3.5+a9))
    return [afuncGM, bfuncGM, cfuncGM]

ks = np.logspace(-5, 2, 300)
zmmesh, ksmesh = np.meshgrid(zm, ks)
fit_funcs = {}
for model in ['GM', 'SC']:
    fit_funcs[model] = [interp.RectBivariateSpline(zm, ks, fun(zmmesh, ksmesh).T) for fun in get_afuncs_form_coeffs_for_interp(model)]


def get_afuncs_form_coeffs(model):
    return [lambda k, z: f(k, z, grid = False)for f in fit_funcs[model]]


afuncTR, bfuncTR, cfuncTR = lambda z, k: np.ones_like(k), lambda z, k: np.ones_like(k), lambda z, k: np.ones_like(k)

def dot(a, b):
    return np.einsum('ij, ij->j', a, b)

def dot_alt(a, b):
    return np.sum(a*b, axis = 0)
    

def getfuncs(model):
    if model == 'TR':
        return afuncTR, bfuncTR, cfuncTR
    elif model in ['GM', 'SC']:
        return get_afuncs_form_coeffs(model)

def F2ptker_vector(k1vec, k2vec, z, model = 'TR'):
    """
    Calculates F2 kernel from PT.

    Parameters
    ----------
    k1vec : array_like
        3-vector of k1, shape, (3, n), where n is the number of points to be calculated at.
    k2vec : array_like
        3-vector of k2, shape, (3, n), where n is the number of points to be calculated at.
    z : array_like
        Redshifts, shape, (1, n).
    """
    afunc, bfunc, cfunc = getfuncs(model)
    resultG = 5/7*afunc(np.linalg.norm(k1vec, axis = 0), z)*afunc(np.linalg.norm(k2vec, axis = 0), z)
    k1vec_hat = k1vec/np.linalg.norm(k1vec, axis = 0)[None, :]
    k2vec_hat = k2vec/np.linalg.norm(k2vec, axis = 0)[None, :]
    resultS = bfunc(np.linalg.norm(k1vec, axis = 0), z)*bfunc(np.linalg.norm(k2vec, axis = 0), z)*1/2*(np.linalg.norm(k1vec, axis = 0)/np.linalg.norm(k2vec, axis = 0) + np.linalg.norm(k2vec, axis = 0)/np.linalg.norm(k1vec, axis = 0))*dot(k1vec_hat, k2vec_hat)
    resultT = 2/7*(dot(k1vec_hat, k2vec_hat))**2*cfunc(np.linalg.norm(k1vec, axis = 0), z)*cfunc(np.linalg.norm(k2vec, axis = 0), z)
    return resultG + resultS + resultT
#F2ptker = np.vectorize(F2ptker)

def F2ptker(k1, k2, z, model = 'TR'):
    afunc, bfunc, cfunc = getfuncs(model)
    resultG = 5/7*afunc(k1, z)*afunc(k2, z)
    k1vec_hat = 1
    k2vec_hat = 1
    resultS = bfunc(k1, z)*bfunc(k2, z)*1/2*(k1/k2 + k2/k1)*np.dot(k1vec_hat, k2vec_hat)
    resultT = 2/7*(np.dot(k1vec_hat, k2vec_hat))**2*cfunc(k1, z)*cfunc(k2, z)
    return (resultG + resultS + resultT)

P = PK.P
def bispectrum_matter_single(k1vec, k2vec, k3vec, z, model = 'TR'):
    ksvec = [k1vec, k2vec, k3vec]
    combinations = list(itertools.combinations([0,1,2], 2))
    return sum([F2ptker_vector(ksvec[comb[0]], ksvec[comb[1]], z, model = model)*P(z, np.linalg.norm(ksvec[comb[0]], axis = 0), grid = False)*P(z, np.linalg.norm(ksvec[comb[1]], axis = 0), grid = False) for comb in combinations])
#bispectrum_matter = np.vectorize(bispectrum_matter_single, signature='(n),(n),(n),(n)->(n)')

P = PK.P
def bispectrum_matter_single_alt(k1vec, k2vec, k3vec, z, model = 'TR'):
    ksvec = [k1vec, k2vec, k3vec]
    combinations = list(itertools.combinations([0,1,2], 2))
    return sum([F2ptker(ksvec[comb[0]], ksvec[comb[1]], z, model = model)*P(z, np.linalg.norm(ksvec[comb[0]]), grid = False)*P(z, np.linalg.norm(ksvec[comb[1]]), grid = False) for comb in combinations])

bispectrum_matter = bispectrum_matter_single

aofchis = 1/(1+zs)
zofchi = interp.interp1d(chis, zs, kind='cubic', fill_value='extrapolate', bounds_error=False)

Wkk = windows.cmblensingwindow_ofchi(chis, aofchis, H0, Omegam, interp1d = True, chistar = chistar)

Wphiphiv = np.nan_to_num(-2*(chistar-chis)/(chistar*chis))
Wphiphiv[0] = 0
Wphiphi = interp.interp1d(chis, Wphiphiv, bounds_error = True)#, fill_value = 'extrapolate')


gammav = 3/2*H0**2*Omegam/(cosmoconstants.CSPEEDKMPERSEC**2)/aofchis
gamma = interp.interp1d(chis, gammav, bounds_error = True)#, fill_value = 'extrapolate')

def bispectrum_matter_2d(l1, l2, l3, theta1, theta2, theta3, z, model = 'TR'):
    l1vec = np.array([l1*np.cos(theta1), l1*np.sin(theta1), np.zeros_like(l1)])
    l2vec = np.array([l2*np.cos(theta2), l2*np.sin(theta2), np.zeros_like(l2)])
    l3vec = np.array([l3*np.cos(theta3), l3*np.sin(theta3), np.zeros_like(l3)])
    return bispectrum_matter(l1vec, l2vec, l3vec, z, model = model)


def bispectrum_matter_2d_alt(l1, l2, l3, z, model = 'TR'):
    l1vec = l1
    l2vec = l2
    l3vec = l3
    return bispectrum_matter_single_alt(l1vec, l2vec, l3vec, z, model = model)


def integrate_bispectrum(l1, l2, l3, model = 'TR'):
    zeros = np.zeros_like(l1)
    bispectrum_at_ells_of_chi = lambda chi: -chi**2*Wphiphi(chi)**3*gamma(chi)**3/(l1*l2*l3)**2*bispectrum_matter_2d(l1/chi, l2/chi, l3/chi, zeros, zeros, zeros, zofchi(chi), model = model)
    return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, chistar, miniter = 100)[0]

ls = np.arange(10, 4000, 10)

def integrate_bispectrum_kkk(l1, l2, l3, model = 'TR'):
    zeros = np.zeros_like(l1)
    bispectrum_at_ells_of_chi = lambda chi: chi**2*Wkk(chi)**3*bispectrum_matter_2d(l1/chi, l2/chi, l3/chi, zeros, zeros, zeros, zofchi(chi), model = model)
    return sinteg.quadrature(bispectrum_at_ells_of_chi, 0, chistar, miniter = 200)[0]
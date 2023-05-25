"""
Calculates the n32 bias for general estimators.
"""

import bispectrum_3D_numba as b3n

import interpolated_quantities_numba as iqn

import numba as nb

import temperatureinfo as ti

import numpy as np

import vegas

from lenscarf import n0n1_iterative as n01

import matplotlib.pyplot as plt

from scipy import interpolate

from plancklens import utils


def getW(pp, noise, lmax_qlm):
    lmax = len(pp)
    W = np.zeros(lmax)
    W[:lmax_qlm] = pp[:lmax_qlm] * utils.cli(pp[:lmax_qlm] + noise[:lmax_qlm])
    return W


############ SETTINGS   #####################

ptt_key = "ptt"
pee_key = "p_p"

qe_key = "p_p"
itermax = 3

cls_unl = np.load("cls_unl.npy", allow_pickle = True).take(0)
cls_len = np.load("cls_len.npy", allow_pickle = True).take(0)
cls_grad = np.load("cls_grad.npy", allow_pickle = True).take(0)
fidcls_noise = np.load("fidcls_noise.npy", allow_pickle = True).take(0)

print("Setting bb noise to inf")
fidcls_noise["bb"] += 1e23
#cls_len["bb"] += 1e23
#cls_unl["bb"] += 1e23
#cls_grad["bb"] += 1e23

print("Info")

if qe_key == "ptt":
    noise, beam = 1., 1.
else:
    noise, beam = 1., 1.*np.sqrt(2)

print("noise, beam", noise, beam)

lmin, lmax = 10, 4000
print("lmin, lmax", lmin, lmax)
noise, beam = 1., 1.
print("noise, beam", noise, beam)

uTT, lTT, tTT, gTT = ti.get_interpolated(lmin, lmax, noise, beam) #QE quantities, 0 iteration


print("lmin, lmax", lmin, lmax)
uEE, lEE, tEE, gEE = ti.get_interpolatedEE(lmin, lmax, noise, beam)

lBB, gBB, tBB = ti.get_interpolatedBB(lmin, lmax, noise, beam)

Ls = np.arange(10, 3500, 200)

print("Calculate results at Ls", Ls)

print("Preparing some results with lenscarf")

try:
#    result = np.load("result.npy", allow_pickle = True).take(0)
    N0s_biased = np.load(f"N0s_biased_{qe_key}.npy", allow_pickle = True)
    N0s_unbiased = np.load(f"N0s_unbiased_{qe_key}.npy", allow_pickle = True)
    delcls_fid = np.load(f"delcls_fid_{qe_key}.npy", allow_pickle = True)
    delcls_true = np.load(f"delcls_true_{qe_key}.npy", allow_pickle = True)
    N1s_biased = np.load(f"N1s_biased_{qe_key}.npy", allow_pickle = True)
    N1s_unbiased = np.load(f"N1s_unbiased_{qe_key}.npy", allow_pickle = True)

except:
    result = n01.get_biases_iter(qe_key, nlev_t = 1, nlev_p = 1*np.sqrt(2), beam_fwhm = 1., cls_unl_fid = cls_unl, lmin_ivf = lmin, lmax_ivf = lmax, itermax = itermax, datnoise_cls = fidcls_noise,
                                lmax_qlm=None, version = 'wN1')
    Rs, N0s_biased, N0s_unbiased, delcls_fid, delcls_true, N1s_biased, N1s_unbiased = result
    np.save(f"N0s_biased_{qe_key}.npy", N0s_biased)
    np.save(f"N0s_unbiased_{qe_key}.npy", N0s_unbiased)
    np.save(f"delcls_fid_{qe_key}.npy", delcls_fid)
    np.save(f"delcls_true_{qe_key}.npy", delcls_true)
    np.save(f"N1s_biased_{qe_key}.npy", N1s_biased)
    np.save(f"N1s_unbiased_{qe_key}.npy", N1s_unbiased)
    np.save(f"Rs_{qe_key}.npy", Rs)


pps = [d["pp"] for d in delcls_fid]

for i, pp0 in enumerate(pps):
    W = getW(pp0, N0s_unbiased[i]+N1s_unbiased[i], len(N0s_unbiased[i]))
    np.savetxt(f"W_{qe_key}_{i}.txt", W)


gradients = []


def get_camb():
    import camb
    from camb import model
    #Use flat sky, get gradient spectra
    #pars =camb.set_params(H0=None, cosmomc_theta =0.010411,lmax=4200, lens_potential_accuracy=1)
    lmax = 8000
    pars = camb.CAMBparams()
    #print pars.Recomb.RECFAST_fudge

    pars.set_cosmology(H0=67, ombh2=0.022445, omch2=0.1212, mnu=0)
    pars.InitPower.set_params(As=2.1265e-09, ns=0.96)

    # reionization and recombination 
    pars.Reion.use_optical_depth = True
    pars.Reion.optical_depth = 0.0925
    pars.Reion.delta_redshift = 0.5
    pars.Recomb.RECFAST_fudge = 1.14

    # accuracy
    pars.set_for_lmax(lmax+200, lens_potential_accuracy=4)

    #non linearity
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.halofit_version = 'takahashi'
    pars.Accuracy.AccurateBB = True #need this to avoid small-scale ringing
    #config.lensing_method=lensing_method_curv_corr
    #%time data1=camb.get_results(pars)
    #config.lensing_method=lensing_method_flat_corr # default

    pars.max_l = lmax

    return camb.get_results(pars)

try:
    gradients = np.load(f"gradients_{qe_key}.npy", allow_pickle = True)
except:
    data = get_camb()
    for i, clpp in enumerate(pps):
        ls = np.arange(0, len(clpp))
        clpp = clpp*(ls*(ls+1))**2./2/np.pi
        caso = data.get_lensed_gradient_cls(lmax = len(clpp), CMB_unit = 'muK', raw_cl = True, clpp = clpp)
        TT, EE, BB = caso.T[:3]
        if qe_key == "ptt":
            g = TT
        elif qe_key == pee_key:
            g = EE
        else:
            print("Estimator not implemented")

        gradients += [g]

    np.save(f"gradients_{qe_key}.npy", gradients)



if False:
    for i, p in enumerate(pps):
        plt.loglog(p, label = "delcl_fid %d"%i)
    plt.legend()
    plt.show()

############ UTILS #####################


import sympy as sp

p1, p2 = sp.symbols('p1 p2')
expr = sp.sin(2*(p1-p2))
expr = sp.expand_trig(expr)
l1, l2, l1x, l1y, l2x, l2y = sp.symbols('l1 l2 l1x l1y l2x l2y')

expr = sp.expand(sp.simplify(expr.subs([(sp.cos(p1),l1x/l1),(sp.cos(p2),l2x/l2),
                            (sp.sin(p1),l1y/l1),(sp.sin(p2),l2y/l2)])))

lmbdasin = sp.lambdify((l1, l2, l1x, l1y, l2x, l2y), expr, "numpy")

expr = sp.cos(2*(p1-p2))
expr = sp.expand_trig(expr)
expr = sp.expand(sp.simplify(expr.subs([(sp.cos(p1),l1x/l1),(sp.cos(p2),l2x/l2),
                            (sp.sin(p1),l1y/l1),(sp.sin(p2),l2y/l2)])))
lmbdacos = sp.lambdify((l1, l2, l1x, l1y, l2x, l2y), expr, "numpy")


def dotbatch(a, b):
    return a[0, :]*b[0, :]+a[1, :]*b[1, :]

def fgradEEbatch(l1v, l2v, l1n, l2n, gradientEE):
    Lv = l1v+l2v
    #calculate cos of double the angle between l1v and l2v
    factor = lmbdacos(l1n, l2n, l1v[0, :], l1v[1, :], l2v[0, :], l2v[1, :])
    #there should be ClPPper too. ignoring it.
    return dotbatch(Lv, l1v)*gradientEE(l1n)+dotbatch(Lv, l2v)*gradientEE(l2n)*factor

def fgradEBbatch(l1v, l2v, l1n, l2n, gradientEE, gradientBB):
    Lv = l1v+l2v
    #calculate sin of double the angle between l1v and l2v
    factor = lmbdasin(l1n, l2n, l1v[0, :], l1v[1, :], l2v[0, :], l2v[1, :])
    return dotbatch(Lv, l1v)*gradientEE(l1n)+dotbatch(Lv, l2v)*gradientBB(l2n)*factor

def funlTTbatch(l1v, l2v, l1n, l2n):
    Lv = l1v+l2v
    return dotbatch(Lv, l1v)*uTT(l1n)+dotbatch(Lv, l2v)*uTT(l2n)

def flenTTbatch(l1v, l2v, l1n, l2n):
    Lv = l1v+l2v
    return dotbatch(Lv, l1v)*lTT(l1n)+dotbatch(Lv, l2v)*lTT(l2n)

def fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT):
    Lv = l1v+l2v
    return dotbatch(Lv, l1v)*gradientTT(l1n)+dotbatch(Lv, l2v)*gradientTT(l2n)

def ftotTTfTTbatch(l):
    return tTT(l)

def ftotEEfEEbatch(l):
    return tEE(l)

def ftotBBfBBbatch(l):
    return tBB(l)

def gfEEbatch(lv, Lv, l1n, l2n, gradientEE, totalEE = ftotEEfEEbatch):
    l1v, l2v = lv, Lv-lv
    return fgradEEbatch(l1v, l2v, l1n, l2n, gradientEE)/(2*totalEE(l1n)*totalEE(l2n))

def gfEBbatch(lv, Lv, l1n, l2n, gradientEE, gradientBB, totalEE = ftotEEfEEbatch, totalBB = ftotBBfBBbatch):
    l1v, l2v = lv, Lv-lv
    return fgradEBbatch(l1v, l2v, l1n, l2n, gradientEE, gradientBB)/(2*totalEE(l1n)*totalBB(l2n))

def gfTTbatch(lv, Lv, l1n, l2n, gradientTT, totalTT = ftotTTfTTbatch):
    l1v, l2v = lv, Lv-lv
    return fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT)/(2*totalTT(l1n)*totalTT(l2n))

def gfTTbatch_for_modes(l1v, l2v, l1n, l2n, gradientTT, totalTT = ftotTTfTTbatch):
    return fgradTTbatch(l1v, l2v, l1n, l2n, gradientTT)/(2*totalTT(l1n)*totalTT(l2n))

filter_batch = lambda x: (x >= lmin) & (x <= lmax)

################# CALCULATIONS #####################

def get_AL_MC(Ls, est = "ptt", gradientfunction = None, total = None):

    integ = vegas.Integrator([[lmin, lmax], [0, 2*np.pi]])
    nitn, neval = 1e2, 1e3

    ALMC = []

    for LL in Ls:
        @vegas.batchintegrand
        def integrand(x):
            l1, theta1 = x.T
            l1v = np.array([l1*np.cos(theta1), l1*np.sin(theta1)])
            L = np.ones_like(l1)*LL
            Lv = np.c_[L, np.zeros_like(l1)].T
            l3v = Lv-l1v

            l3 = np.linalg.norm(l3v, axis = 0)

            if est == "ptt":
                gradientTT, totalTT = gradientfunction, total
                fXY = fgradTTbatch(l1v, l3v, l1, l3, gradientTT)
                gXY = gfTTbatch_for_modes(l1v, l3v, l1, l3, gradientTT, totalTT)*filter_batch(l3)
            elif est == "p_p":
                gradientEE, totalEE = gradientfunction, total
                fXY = fgradEEbatch(l1v, l3v, l1, l3, gradientEE)
                gXY = gfEEbatch(l1v, Lv, l1, l3, gradientEE, totalEE)*filter_batch(l3)
            else:
                print("Estimator not implemented")

            product = fXY*gXY     
            common = l1/(2*np.pi)**2

            return product*common
        
        result = integ(integrand, nitn = nitn, neval = neval)
        ALMC += [result.mean]

    ALMC = np.array(ALMC)**-1.

    return ALMC


ALMCS = []

for i in range(itermax):

    print("Iteration %d"%i)

    current_gradient = gradients[i]
    ls = np.arange(0, len(current_gradient))

    gradientf = interpolate.interp1d(ls, current_gradient, fill_value = 0., bounds_error = False)

    noise_component = ti.get_noise(ls, noise, beam)
    
    if qe_key == "ptt":
        cmbpartial = delcls_true[i]["tt"]
    elif qe_key == "p_p":
        cmbpartial = delcls_true[i]["ee"]
    else:
        print("Estimator not implemented")

    lmaxcomm = min(len(ls), len(cmbpartial))

    total = cmbpartial[:lmaxcomm]+noise_component[:lmaxcomm]
    ls = ls[:lmaxcomm]
    totalf = interpolate.interp1d(ls, total, fill_value = 1e10, bounds_error = False)

    ALMC = get_AL_MC(Ls, est = qe_key, gradientfunction = gradientf, total = totalf)
    ALMCS += [ALMC]
    
for i, A in enumerate(ALMCS):
    plt.plot(Ls, A/ALMCS[0], label = "iter %d"%i)
plt.legend()
plt.show()


n32 = {}

integ = vegas.Integrator([[lmin, lmax], [lmin, lmax], [0, 2*np.pi], [0, 2*np.pi]], nhcube_batch = 2000)
nitn, neval = 1e1, 1e3

for it in range(itermax):

    print("Iteration %d"%it)

    current_gradient = gradients[it]
    ls = np.arange(0, len(current_gradient))

    gradientf = interpolate.interp1d(ls, current_gradient, fill_value = 0., bounds_error = False)

    noise_component = ti.get_noise(ls, noise, beam)

    if qe_key == "ptt":
        cmbpartial = delcls_true[it]["tt"]
    elif qe_key == "p_p":
        cmbpartial = delcls_true[it]["ee"]
    else:
        print("Estimator not implemented")

    lmaxcomm = min(len(ls), len(cmbpartial))

    total = cmbpartial[:lmaxcomm]+noise_component[:lmaxcomm]
    ls = ls[:lmaxcomm]
    totalf = interpolate.interp1d(ls, total, fill_value = 1e10, bounds_error = False)

    #model = "SC"
    bpmodel = "TR"
    models = [bpmodel]#, "SC", "GM"]#, "SC", "GM"]#, "SC"]#, "SC", "GM"]
    indices = {"TR": 0, "SC": 1, "GM": 2}
    #index = models.index(model)

    results_n32 = {}

    for index, model in enumerate(models):

        print("Working on model", model, ", index", index)

        keys = ["B", "PB"]
        NTOT = {k: [] for k in keys}

        for LL in Ls:
            @vegas.batchintegrand
            def integrand(x):
                l1, l2, theta1, theta2 = x.T
                cos1, cos2 = np.cos(theta1), np.cos(theta2)
                sin1, sin2 = np.sin(theta1), np.sin(theta2)

                l1v = np.array([l1*cos1, l1*sin1])
                l2v = np.array([l2*cos2, l2*sin2])

                l5v = l1v-l2v

                L = np.ones_like(l1)*LL
                Lv = np.c_[L, np.zeros_like(l1)].T

                l4v = Lv-l2v
                
                l5 = np.sqrt((l1*cos1-l2*cos2)**2+(l1*sin1-l2*sin2)**2)
                l4 = np.sqrt(LL**2+l2**2 -2*LL*l2*cos2)
                l3 = np.sqrt(LL**2+l1**2-2*LL*l1*cos1)

                l4_dot_L = (-LL*l2*cos2 + LL**2)
                l2_dot_L = (LL*l2*cos2)

                gXY = gfEEbatch(l2v, Lv, l2, l4, gradientf, totalf)*filter_batch(l4)
                gYX = gfEEbatch(l4v, Lv, l4, l2, gradientf, totalf)*filter_batch(l4)

                l1_dot_l2 = (l1*cos1*l2*cos2) + (l1*sin1*l2*sin2)
                l2_dot_l3 = l2_dot_L-l1_dot_l2
                l5_dot_l1 = l1**2-l1_dot_l2
                l5_dot_l3 = LL*l1*cos1-l2_dot_L-l1**2+l1_dot_l2

                cl5_XY = uEE(l5)*filter_batch(l5) #uTT(l5)*filter_batch(l5)
                Cl2 = gradientf(l2)*filter_batch(l2) #uTT(l2)*filter_batch(l2)


                hX_l5_l2 = lmbdacos(l5, l2, l5v[0, :], l5v[1, :], l2v[0, :], l2v[1, :])
                hY_l5_l4 = lmbdasin(l5, l4, l5v[0, :], l5v[1, :], l4v[0, :], l4v[1, :])

                hY_l2_l4 = lmbdasin(l2, l4, l2v[0, :], l2v[1, :], l4v[0, :], l4v[1, :])
                hX_l2_l4 = lmbdacos(l2, l4, l2v[0, :], l2v[1, :], l4v[0, :], l4v[1, :])

                productA1 = -l5_dot_l1*l5_dot_l3*hX_l5_l2*hY_l5_l4*cl5_XY*gXY
                productC1 = l2_dot_l3*l1_dot_l2*(gXY*hY_l2_l4+gYX*hX_l2_l4)*Cl2*1/2

                #born_term = bispectrum_Born(l1, l3, LL)*8/(l1*l3*LL)**2
                born_term = 0.

                #ff = 1+(1-Wsinterpiter1(l1))*(1-Wsinterpiter1(l3))
                bispectrum_result = b3n.bispec_phi_general(l1, l3, LL, index)#*ff #*(1+(1-rho2interp1(l1))*(1-rho2interp1(l3)))

                bispectrum_total = bispectrum_result+born_term

                common = l1*l2*bispectrum_result/(2*np.pi)**4
                common_total = l1*l2*bispectrum_total/(2*np.pi)**4

                #A1 = common*productA1
                #C1 = common*productC1
                result = (productA1+productC1)*common
                result_total = (productA1+productC1)*common_total
                #return [common*productA1, common*productC1]
                return {"B": result, "PB": result_total} #{'A1': common*productA1, 'C1': common*productC1}
            
            result = integ(integrand, nitn = 2*nitn, neval = 2*neval)
            for k in keys:
                NTOT[k] += [result[k].mean]
            
            #NA1 += [result['A1'].mean]
            #NC1 += [result['C1'].mean]
        for k in keys:
            NTOT[k] = np.array(NTOT[k])
        results_n32[model] = NTOT

    n32[it] = results_n32

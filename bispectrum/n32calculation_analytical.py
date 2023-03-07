import temperatureinfo as ti

import vegas

import numpy as np

import bispectrum_3D_numba as b3n

import time

import numba

#########

def dotbatch(a, b):
    return a[0, :]*b[0, :]+a[1, :]*b[1, :]

    
def funlTTbatch(l1v, l2v, l1n, l2n):
    Lv = l1v+l2v
    return dotbatch(Lv, l1v)*uTT(l1n)+dotbatch(Lv, l2v)*uTT(l2n)

def flenTTbatch(l1v, l2v, l1n, l2n):
    Lv = l1v+l2v
    return dotbatch(Lv, l1v)*lTT(l1n)+dotbatch(Lv, l2v)*lTT(l2n)

def fgradTTbatch(l1v, l2v, l1n, l2n):
    Lv = l1v+l2v
    return dotbatch(Lv, l1v)*lTT(l1n)+dotbatch(Lv, l2v)*lTT(l2n)

def ftotTTfTTbatch(l):
    return tTT(l)

def gfTTbatch(lv, Lv, l1n, l2n):
    l1v, l2v = lv, Lv-lv
    return flenTTbatch(l1v, l2v, l1n, l2n)/(2*ftotTTfTTbatch(l1n)*ftotTTfTTbatch(l2n))

def gfTTbatch_for_modes(l1v, l2v, l1n, l2n):
    return flenTTbatch(l1v, l2v, l1n, l2n)/(2*ftotTTfTTbatch(l1n)*ftotTTfTTbatch(l2n))



#########

lmin, lmax = 100, 3500
print("lmin, lmax", lmin, lmax)
noise, beam = 7., 1.4
print("noise, beam", noise, beam)
uTT, lTT, tTT = ti.get_interpolated(lmin, lmax, noise, beam)

Ls = np.arange(20, 3000, 10)

##########

integ = vegas.Integrator([[lmin, lmax], [0, 2*np.pi]])
nitn, neval = 100, 1e3

ALMC = []

s = time.time()
print("Starting ALMC calculation")

for LL in Ls:
    @vegas.batchintegrand
    def integrand(x):
        l1, theta1 = x.T
        l1v = np.array([l1*np.cos(theta1), l1*np.sin(theta1)])
        L = np.ones_like(l1)*LL
        Lv = np.c_[L, np.zeros_like(l1)].T
        l3v = Lv-l1v

        l3 = np.linalg.norm(l3v, axis = 0)

        fXY = flenTTbatch(l1v, l3v, l1, l3)

        gXY = gfTTbatch_for_modes(l1v, l3v, l1, l3)

        product = fXY*gXY     
        common = l1/(2*np.pi)**2

        return product*common
    
    result = integ(integrand, nitn = nitn, neval = neval)
    ALMC += [result.mean]

print("ALMC calculation finished in", time.time()-s, "seconds")

ALMC = np.array(ALMC)**-1.


########

def dot(a, b):
    return np.dot(a, b)

def fTT(l1v, l2v, l1n, l2n):
    return dot(l1v+l2v, l1v)*lTT(l1n)+dot(l1v+l2v, l2v)*lTT(l2n)

def ftotTT(l):
    return tTT(l)

@numba.jit
def filters(ells):
    return (ells>=lmin) & (ells<=lmax)

def gTT(lv, Lv, lvnorm):
    l1v, l2v = lv, Lv-lv
    l1n, l2n = lvnorm, np.linalg.norm(l2v, axis = 0)
    return fTT(l1v, l2v, l1n, l2n)/(2*ftotTT(l1n)*ftotTT(l2n))#*filters(l1n)*filters(l2n)

from interpolation import interp

@numba.jit(nopython = True)
def uTTinterp(l):
    return interp(ti.L, ti.unlensed, l) 

noise_component = ti.get_noise(ti.L, noise, beam)
@numba.jit(nopython = True)
def tTTinterp(l):
    return interp(ti.L, ti.lensed+noise_component, l) 

#lTT = sp.interpolate.interp1d(ti.L, ti.lensed, fill_value = 0., bounds_error = False)
#tTT = sp.interpolate.interp1d(ti.L, ti.lensed+ti.get_noise(L, noise, beam), fill_value = 1e10, bounds_error = False)

@numba.jit(nopython = True, fastmath = True)
def get_integral_for_L(L, lmin, lmax, l1res = 10., l2res = 10.):

    eps = 1e-4

    integralL = 0.

    l1s = np.linspace(lmin, lmax, l1res)

    l2s = np.linspace(lmin, lmax, l2res)

    for i, l1 in enumerate(l1s):

        ntheta1 =  max(32, 2*int(l1)+1) 
        ntheta1 = min(ntheta1, 128) 
        dtheta1 = (2*np.pi/ntheta1)
        theta1s = np.linspace(eps, 2*np.pi-eps, ntheta1)
        
    
        for theta1 in theta1s:
            cos1 = np.cos(theta1)
            sin1 = np.sin(theta1)
            l3 = np.sqrt(L**2+l1**2-2*L*l1*cos1) #size l1s

            weight1 = dtheta1*l1/(2*np.pi)**2 
            bispectrum_term = b3n.bispec_phi_TR_non_vec(l1, l3, L) #size l1s
            integral2 = np.empty(l2s.shape, dtype = np.float64)
        
            for i2, l2 in enumerate(l2s):

                ntheta2 =  max(32,2*int(l2)+1) 
                ntheta2 = min(ntheta2, 128) 
                dtheta2 = (2*np.pi/ntheta2)
                theta2s = np.linspace(0, 2*np.pi, ntheta2)
                
                somma = 0

                Cl2 = uTTinterp(l2)
            
                cos2s = np.cos(theta2s)
                sin2s = np.sin(theta2s)

                weight2 = dtheta2*l2/(2*np.pi)**2 
                l4 = np.sqrt(L**2+l2**2 -2*L*l2*cos2s)
                l5 = np.sqrt((l1*cos1+l2*cos2s)**2+(l1*sin1+l2*sin2s)**2) 

                Cl5 = uTTinterp(l5)

                response = ((-L*l2*cos2s + L**2)*uTTinterp(l4) + (L*l2*cos2s)*uTTinterp(l2)) 
                filter = 1/(tTTinterp(l2)*tTTinterp(l4))
                filter[l4<lmin] = 0
                filter[l4>lmax] = 0 

                gTT = response*filter

                l1_dot_l2 = (l1*cos1+l2*cos2s) + (l1*sin1+l2*sin2s)
                l2_dot_l3 = l2*cos2s*L-l1_dot_l2
                l5_dot_l1 = l1**2-l1_dot_l2
                l5_dot_l3 = L*l1*cos1-L*l2*cos2s-l1**2+l1_dot_l2

                A1terms = -l5_dot_l1*l5_dot_l3*1*1*Cl5*gTT
                C1terms = l2_dot_l3*l1_dot_l2*1*1*Cl2*gTT*2

                somma += np.sum(A1terms+C1terms)*weight2

                integral2[i2] = somma

            integral2 = np.sum(integral2)
            integral2 *= bispectrum_term
            integralL += weight1*integral2
            #weight1*integral2*bispectrum_term
            #integral2*bispectrum_term

    return integralL

            
l1res, l2res = 20, 20
get_integral_for_L(10, lmin, lmax, l1res, l2res)   


from joblib import Parallel, delayed

batch_size = 'auto'
n_jobs = 4
backend = "threading"
s = time.time()
results = Parallel(n_jobs = n_jobs, batch_size = batch_size, backend = backend, verbose = 0)(delayed(get_integral_for_L)(LL, lmin, lmax, l1res, l2res) for LL in Ls)
print("Parallel calculation finished in", time.time()-s, "seconds")

NTOT = np.array(results)
import matplotlib.pyplot as plt
plt.loglog(Ls, ALMC*abs(NTOT), label = 'N32 analytical')
plt.show()


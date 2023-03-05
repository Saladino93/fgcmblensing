import temperatureinfo as ti

import vegas

import numpy as np

import bispectrum_3D_numba as b3n

import time

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

def main(inputells):

    integ = vegas.Integrator([[lmin, lmax], [lmin, lmax], [0, 2*np.pi], [0, 2*np.pi]])
    nitn, neval = 1e3, 1e3

    NA1 = []
    NC1 = []

    def get(L):
        @vegas.batchintegrand
        def integrand(x):
            l1, l2, theta1, theta2 = x.T
            l1v, l2v = np.array([l1*np.cos(theta1), l1*np.sin(theta1)]), np.array([l2*np.cos(theta2), l2*np.sin(theta2)])
            L = np.ones_like(l1)*LL
            Lv = np.c_[L, np.zeros_like(l1)].T
            l3v = Lv-l1v
            l4v = Lv-l2v
            l5v = l1v-l2v

            l5 = np.linalg.norm(l5v, axis = 0)
            l4 = np.linalg.norm(l4v, axis = 0)
            l3 = np.linalg.norm(l3v, axis = 0)

            l5vdotl1v = dotbatch(l5v, l1v)
            l5vdotl3v = dotbatch(l5v, l3v)

            l2vdotl1v = dotbatch(l2v, l1v)
            l2vdotl3v = dotbatch(l2v, l3v)

            h_5_2_X = 1
            h_5_4_Y = 1

            h_2_4_Y = 1
            h_2_4_X = 1
    
            cl5_XY = uTT(l5)

            cl2_XY = uTT(l2)
            cl2_YX = cl2_XY

            gXY = gfTTbatch_for_modes(l2v, l4v, l2, l4)
            productA1 = (-1)*l5vdotl1v*l5vdotl3v*cl5_XY*h_5_2_X*h_5_4_Y*gXY

            #assume for now X = Y
            gYX = gXY 
            productC1 = 1/2*(gXY*cl2_XY*h_2_4_Y+gYX*cl2_YX*h_2_4_X)*l2vdotl1v*l2vdotl3v

            bispectrum_result = b3n.bispec_phi_TR(L,l1, l3)
            common = l1*l2*bispectrum_result/(2*np.pi)**4

            #print(productA1.shape, productC1.shape)

            #return [common*productA1, common*productC1]
            return {'A1': common*productA1, 'C1': common*productC1}
        
        result = integ(integrand, nitn = nitn, neval = neval)
        return result
    
    result = get(inputells)

    NA1 += [result['A1'].mean]
    NC1 += [result['C1'].mean]

    NA1 = np.array(NA1)
    NC1 = np.array(NC1)

    return (NA1, NC1)

from joblib import Parallel, delayed

batch_size = 'auto'
n_jobs = 4
backend = "loky"
s = time.time()
results = Parallel(n_jobs = n_jobs, batch_size = batch_size, backend = backend, verbose = 0)(delayed(main)(l) for l in Ls)
print("Parallel calculation finished in", time.time()-s, "seconds")

NA1 = np.array([r[0] for r in results])[:, 0]
NC1 = np.array([r[1] for r in results])[:, 0]
import matplotlib.pyplot as plt
plt.loglog(Ls, ALMC*abs(NA1+NC1), label = 'N32 vegas')
plt.show()


import camb
import symlens as s
from pixell import enmap, utils as putils
import numpy as np
import scipy as sp

TCMB = 2.73e6

potential_file = "lensedCMB_dmn1_lenspotentialCls.dat"
d = np.loadtxt(f'/Users/omard/Documents/projects/fgcmblensing/input/giulio/{potential_file}')
l = d[:, 0]
lfact = 2*np.pi/l/(l+1)
unlensed = d[:, 1]
unlensed *= lfact#/TCMB**2.

unlensedEE = d[:, 2]
unlensedEE *= lfact#/TCMB**2.

unlensedEB = unlensedEE*0.


ddlensing = d[:, 5]
kk = 2*np.pi*ddlensing/4
pp = 4/l**2/(l+1)**2*kk

TCMB = 2.73e6

dd = np.loadtxt('/Users/omard/Documents/projects/fgcmblensing/input/giulio/lensedCMB_dmn1_lensedgradCls.dat')
elgrad = dd[:, 0]
gradTT = dd[:, 1]
lfact = 2*np.pi/elgrad/(elgrad+1)
gradTT *= lfact

gradEE = dd[:, 2]
gradEE *= lfact

gradBB = dd[:, 3]
gradBB *= lfact

gradPPerp = dd[:, 4]
gradPPerp *= lfact

d = np.loadtxt('/Users/omard/Documents/projects/fgcmblensing/input/giulio/lensedCMB_dmn1_lensedCls.dat')
L, TT = d[:, 0], d[:, 1]
lensed = d[:, 1]
lfact = 2*np.pi/L/(L+1)
TT *= lfact#/TCMB**2.
#lensed *= lfact

lensedEE = d[:, 2]
lensedEE *= lfact#/TCMB**2.

lensedBB = d[:, 3]
lensedBB *= lfact#/TCMB**2.


unlensed = np.interp(L, l, unlensed)
unlensedEE = np.interp(L, l, unlensedEE)


get_noise = lambda x, level, theta: (level*np.pi/180/60)**2*np.exp(x*(x+1)*np.deg2rad(theta / 60)**2/8/np.log(2))

shape,wcs = enmap.geometry(shape=(512,512),res=2.0*putils.arcmin,pos=(0,0))

modlmap = enmap.modlmap(shape,wcs)

f = s.Ldl1 * s.e('uC_T_T_l1') + s.Ldl2 * s.e('uC_T_T_l2')

F = f / 2 / s.e('tC_T_T_l1') / s.e('tC_T_T_l2')

expr1 = f * F

fsky = 0.4

def get_norm(tellmin, tellmax, noise = 1., beam = 1.):

    feed_dict = {}
    feed_dict['uC_T_T'] = s.interp(L,TT)(modlmap)
    feed_dict['tC_T_T'] = s.interp(L,TT)(modlmap)+s.interp(L,get_noise(L, noise, beam))(modlmap)

    xmask = s.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
    integral = s.integrate(shape,wcs,feed_dict,expr1,xmask=xmask,ymask=xmask).real
    Alkappa = modlmap**4*1/integral/4

    AlkappaR = s.A_l(shape,wcs,feed_dict,"hu_ok","TT",xmask=xmask,ymask=xmask)

    bin_edges = np.arange(1,4000,20)
    binner = s.bin2D(modlmap,bin_edges)
    A_l = Alkappa

    cents, ALR1D = binner.bin(AlkappaR)

    cents, AL1D = binner.bin(A_l)

    Nl = s.N_l_from_A_l_optimal(shape,wcs,AlkappaR)
    cents, Nl1D = binner.bin(Nl)

    expr = integral
    cents, expr1D = binner.bin(expr)
    Alphi = expr1D**-1

    return cents, Alphi


def get_interpolated(lmin, lmax, noise = 1., beam = 1.):
    #selection = (L >= lmin) & (L <= lmax)
    uTT = sp.interpolate.interp1d(L, unlensed, fill_value = 0., bounds_error = False)
    lTT = sp.interpolate.interp1d(L, lensed, fill_value = 0., bounds_error = False)
    gTT = sp.interpolate.interp1d(elgrad, gradTT, fill_value = 0., bounds_error = False)
    tTT = sp.interpolate.interp1d(L, lensed+get_noise(L, noise, beam), fill_value = 1e10, bounds_error = False)
    return uTT, lTT, tTT, gTT

def get_interpolatedEE(lmin, lmax, noise = 1., beam = 1.):
    #selection = (L >= lmin) & (L <= lmax)
    uEE = sp.interpolate.interp1d(L, unlensedEE, fill_value = 0., bounds_error = False)
    lEE = sp.interpolate.interp1d(L, lensedEE, fill_value = 0., bounds_error = False)
    gEE = sp.interpolate.interp1d(elgrad, gradEE, fill_value = 0., bounds_error = False)
    tEE = sp.interpolate.interp1d(L, lensedEE+get_noise(L, noise, beam), fill_value = 1e10, bounds_error = False)
    return uEE, lEE, tEE, gEE


def get_interpolatedBB(lmin, lmax, noise = 1., beam = 1.):
    lBB = sp.interpolate.interp1d(L, lensedBB, fill_value = 0., bounds_error = False)
    gBB = sp.interpolate.interp1d(elgrad, gradBB, fill_value = 0., bounds_error = False)
    tBB = sp.interpolate.interp1d(L, lensedBB+get_noise(L, noise, beam), fill_value = 1e10, bounds_error = False)
    return lBB, gBB, tBB


from numba import jit
import numba

import numpy as np

from interpolation.splines import UCGrid, eval_linear
from interpolation.splines import extrap_options as xto

import itertools

import pathlib

productsdir = pathlib.Path("numbaproducts/")

chikk, Wkkarr = np.loadtxt(productsdir / "Wkk.txt", unpack = True)

chi_precalculated, z_precalculated = np.loadtxt(productsdir / "zs.txt", unpack = True)

zKNL, kNLz = np.loadtxt(productsdir / "zkNL_kNL.txt", unpack = True)

zs8, s8arr = np.loadtxt(productsdir / "sigma8.txt", unpack = True)

names = ["non_lin", "lin", "neff"]

grid2d = np.loadtxt("numbaproducts/z_k_matter_powers.txt")

minlogz, minlogk = grid2d.min(axis = 0)
maxlogz, maxlogk = grid2d.max(axis = 0)
NN = int(np.sqrt(grid2d.shape[0]))

grid = UCGrid((minlogz, maxlogz, NN), (minlogk, maxlogk, NN))

name = names[0]
valuesP = np.loadtxt(f"numbaproducts/{name}_matter_power.txt")
@jit(nopython = True)
def P2D(z, k): 
    return eval_linear(grid, valuesP, np.log10(np.array([z, k])).T, xto.NEAREST)






name = names[1]
valuesPlin = np.loadtxt(f"numbaproducts/{name}_matter_power.txt")
def Plin2D(z, k):
    return eval_linear(grid, valuesPlin, np.log10(np.array([z, k])).T, xto.NEAREST)

name = names[2]
valuesneff = np.loadtxt(f"numbaproducts/{name}_matter_power.txt")

@jit(nopython = True)
def nefff(z, k): 
    return eval_linear(grid, valuesneff, np.log10(np.array([z, k])).T, xto.NEAREST)

@jit(nopython = True)
def integrate(y, x):
    return np.trapz(y, x)

@jit(nopython = True)
def zofchi(newchis):
    return np.interp(newchis, chi_precalculated, z_precalculated)

@jit(nopython = True)
def Wkk(newchis):
    return np.interp(newchis, chikk, Wkkarr)

@jit(nopython = True)
def kNLzf(z):
    return np.interp(z, zKNL, kNLz)

@jit(nopython = True)
def s8(z):
    return np.interp(z, zs8, s8arr)

@jit(nopython = True)
def eval_linear_numba(grid, values, points):
    return eval_linear(grid, values, points)

@jit(nopython = True)
def Q(x): 
    return (4-2**x)/(1+2**(x+1))
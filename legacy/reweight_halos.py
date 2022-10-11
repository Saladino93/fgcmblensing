import numpy as np

import pathlib

import time

def select_z(z_real, zmin_sel, zmax_sel):
    selection = (z_real > zmin_sel) & (z_real < zmax_sel)
    return z_real[selection]

def get_nz(redshifts: np.ndarray, weights: np.ndarray = None, nbins: int = 20):
    galaxyzbins = np.linspace(redshifts.min(), redshifts.max(), nbins)
    #CREATE dn/dz
    histog = np.histogram(redshifts, galaxyzbins, weights = weights)
    z, nz = histog[1], histog[0]
    z = (z[1:]+z[:-1])/2.
    return z, nz


source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')
halo_file =  source_dir/'halo_nbody.ascii'
start = time.time()
halos = np.loadtxt(halo_file)
print(f'Took {time.time()-start} seconds to read halo file.')

zs = halos[:, 0]

#Here, you try to get the desired nz from nznorm, at the zs of the original halo catalog
weights = np.interp(zs, zmag, nznorm)

from numpy import random as nprnd

rng = nprnd.default_rng()
sumweights = sum(weights)

#Total number of galaxies for your bin, might be the one coming from real data, or an expected one from number density over your area
#Ngalaxies_for_bin = ....

indices = rng.choice(len(zs), Ngalaxies_for_bin,
              p = weights/sumweights, replace = False)

draw = zs[indices]

drawzs, drawnz = get_nz(redshifts = draw, nbins = 100)
plt.plot(drawzs, drawnz)
pplt.save
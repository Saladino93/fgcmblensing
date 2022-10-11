import read_config

import argparse

import pathlib

import numpy as np

#to make latex work on Mac
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

import matplotlib.pyplot as plt
plt.style.use('science')

import foregrounds

import SONoise


parser = argparse.ArgumentParser(description = 'Calculates lensed and unlensed CMB theory spectra')
parser.add_argument('-c','--configuration', help = 'Configuration file', required = True)
configuration = parser.parse_args().configuration

output_dir = pathlib.Path(configuration)
output_dir = pathlib.Path(output_dir.stem)

config = read_config.ConfigurationReader(configuration)

ls, uTT, lTT = np.loadtxt(output_dir/config.cmb_name, unpack = True)


'''
#this is for SO file from SO repo
#probably should write a script for noise processing to make things more general
#  units = uK^2 
#  ell_min = 40 
#  ell_max = 7979 
#  Deproj-0: standard ILC 
#  Deproj-1: tSZ deprojection (constrained ILC) 
#  Deproj-2: fiducial CIB deprojection (constrained ILC) 
#  Deproj-3: tSZ and fiducial CIB deprojection (constrained ILC) 
ell, n0, n1, n2, n3 = np.loadtxt(output_dir/config.noise_name, unpack = True)
'''

#ell, nl27, nl39, nl93, nl145, nl225, nl280, nl27x39, nl93x145, nl225x280 = np.loadtxt(output_dir/config.noise_name, unpack = True)
#nus = np.array([27, 39, 93, 145, 225, 280])

Noise = SONoise.SONoiseReader(output_dir/config.noise_name, nus = config.nus, cross_nus = config.cross_nus)
nu0 = config.nu0
nu1, nu2 = [nu0]*2
ell, nl = Noise.ell, Noise.get_noise(nu1 = nu1)


Foreground = foregrounds.Foregrounds(nu1 = nu1, nu2 = nu2)
fglabels = ['tSZ', 'kSZ', 'CIB', 'Radio PS', 'tSZ-CIB', 'Galactic Dust']
fgfuncs = [Foreground.tsz, Foreground.ksz, Foreground.cib, Foreground.radps, Foreground.tsz_cib, Foreground.galacticDust] #getattr.... with a label
fgs = [(f(ells = ell), l) for f, l in zip(fgfuncs, fglabels)]

plt.plot(ls, uTT, color = 'blue', label = 'Unlensed CMB')
plt.plot(ls, lTT, color = 'black', lw = 2, label = 'Lensed CMB')
plt.plot(ell, nl, color = 'red', ls = '--', label = f'SO Noise {nu0} GHz')
for t in fgs:
    p, l = t
    plt.plot(ell, p, label = l, ls = '-.')
#plt.plot(ell, n1, color = 'orange', label = 'tSZ deprojection (constrained ILC) ')
plt.legend()
plt.ylabel('$C_l$')
plt.xlabel('$l$')
plt.xscale('log')
plt.yscale('log')
plt.show()
#plt.savefig(output_dir/, dpi = 200)
plt.close()


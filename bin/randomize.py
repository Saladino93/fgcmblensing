'''
Generates randomized maps and gaussian maps from input maps.

Note, correlation between maps are tried to be preserved in the Gaussian maps.
'''

import numpy as np

import healpy as hp

import argparse

import foregrounds_utils as fgutils

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest = 'i', type = list, help = 'list of input fits')
#parser.add_argument('-o', dest = 'o', type = list, help = 'list of output fits')
parser.add_argument('-odir', dest = 'odir', type = str, help = 'output directory for randomized')
parser.add_argument('-nside', dest = 'nside', type = int, help = 'nside')

args = parser.parse_args()
inputs = args.i
out_dir = args.odir
#outputs = args.o
nside = args.nside

for i in inputs:
    o = out_dir+f'{i}_randomized'
    mappa = hp.read_map(f'{i}.fits')
    mappa_lm = hp.map2alm(mappa)
    hp.write_alm(f'{i}_alm.fits', mappa_lm)
    raw_spectrum = hp.alm2cl(mappa_lm)
    mappa_gauss = hp.synfast(raw_spectrum, nisde = nside)
    hp.write_map(g, mappa_out)
    randomized_lm = randomizing_fg(mappa_lm)
    hp.write_alm(f'{o}_alm.fits', randomized_lm)
    mappa_out = hp.alm2map(randomized_lm, nside = nside)
    hp.write_map(o, mappa_out)



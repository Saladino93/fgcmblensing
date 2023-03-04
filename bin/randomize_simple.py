'''
Generates randomized maps and gaussian maps from input maps.

Note, correlation between maps are tried to be preserved in the Gaussian maps.
'''

import pathlib

import healpy as hp

import argparse

import foregrounds_utils as fgutils

parser = argparse.ArgumentParser()
parser.add_argument('-idir', dest = 'idir', type = str, help = 'input directory')
parser.add_argument('-i', dest = 'i', nargs='+', help = 'list of input fits')
parser.add_argument('-odir', dest = 'odir', type = str, help = 'output directory for randomized')
parser.add_argument('-nside', dest = 'nside', type = int, help = 'nside')

args = parser.parse_args()
inp_dir = pathlib.Path(args.idir)
inputs = list(args.i)
out_dir = pathlib.Path(args.odir)
#outputs = args.o
nside = args.nside

print('Inputs', inputs)

for i in inputs:
    o = str(out_dir/f'{i}_randomized')
    mappa = hp.read_map(inp_dir/f'{i}.fits')
    mappa_lm = hp.map2alm(mappa)
    hp.write_alm(inp_dir/f'{i}_alm.fits', mappa_lm, overwrite = True)
    print('Randomizing...')
    randomized_lm = fgutils.randomizing_fg(mappa_lm)
    hp.write_alm(f'{o}_alm.fits', randomized_lm, overwrite = True)
    mappa_out = hp.alm2map(randomized_lm, nside = nside)
    hp.write_map(o+'.fits', mappa_out, overwrite = True)



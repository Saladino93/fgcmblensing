'''
Convert foregrounds maps to temperature, if they are not, and rescales to a particular frequency.

Generates randomized maps and gaussian maps from input maps.

Note, correlation between maps is NOT preserved for Gaussian and Randomized maps here.

The reason is that we are more interested, in general, to the total effect given by the sum of the maps, or to the single effect by each foreground.

There might be cases where, e.g. CIB+tSZ is wanted, so you might want to change the script to preserve the correlation between maps (e.g. Gaussian maps, input cross-corr in generation of maps too).

Output maps in muK

'''

import numpy as np

import healpy as hp

import argparse

import pathlib

import foregrounds_utils as fgutils

import hp_utils as hputils

import yaml

parser = argparse.ArgumentParser()

parser.add_argument('-c', dest = 'config', type = str, help = 'config file')

args = parser.parse_args()
configname = args.config

with open(configname, 'r') as f:
        data = yaml.safe_load(f)

cambversion = data['cambversion']

path = pathlib.Path(data['path'])
original_maps_path = path/data['original_maps_path']

maps_path = path/data['maps_path']
alms_path = path/data['alms_path']

randomized_path = path/data['randomized_path']
randomized_alms_path = path/data['randomized_alms_path']

gaussianized_path = path/data['gaussianized_path']
gaussianized_alms_path = path/data['gaussianized_alms_path']

freq = data['nu']
nside = data['nside']
lmax = data['lmax']
sky = data['sky']

tcmbunitskey = 'tcmbunits'
tszkey = 'tsz'
notinclude_in_sumkey = 'notinclude_in_sum'

Fg = fgutils.Foregrounds(freq, freq, input_version = cambversion)

somma_alm = 0

for k, v in sky.items():

    factor = Fg.tszFreqDpdceTemp(freq) if k == tszkey else 1

    print(f'Doing {k}')

    isalm = v['alm']
    data = v['data']
    data = f'{data}.fits'

    data = hp.read_map(original_maps_path/data) if not isalm else hp.read_alm(original_maps_path/data)
    data *= factor

    alms = hp.map2alm(data, lmax = lmax) if not isalm else hputils.alm_copy(data, mmaxin = None, lmaxout = lmax, mmaxout = lmax)

    alms = Fg.Jyoversr_to_muKcmb(alms, Fg.nu1) if tcmbunitskey in v.keys() else alms
    hp.write_alm(alms_path/f'{k}_alm.fits', alms, overwrite = True)
    hp.write_map(maps_path/f'{k}.fits', hp.alm2map(alms, nside = nside), overwrite = True)

    somma_alm = somma_alm+alms if notinclude_in_sumkey not in v.keys() else somma_alm

    rand_alms = fgutils.randomizing_fg(alms)
    hp.write_alm(randomized_alms_path/f'{k}_alm.fits', rand_alms, overwrite = True)
    hp.write_map(randomized_path/f'{k}.fits', hp.alm2map(rand_alms, nside = nside), overwrite = True)

    gauss_alms = hp.synalm(hp.alm2cl(alms))
    hp.write_alm(gaussianized_alms_path/f'{k}_alm.fits', gauss_alms, overwrite = True)
    hp.write_map(gaussianized_path/f'{k}.fits', hp.alm2map(gauss_alms, nside = nside), overwrite = True)

hp.write_alm(alms_path/f'sum_alm.fits', somma_alm, overwrite = True)
hp.write_map(maps_path/f'sum.fits', hp.alm2map(somma_alm, nside = nside), overwrite = True)

rand_somma_alm = fgutils.randomizing_fg(somma_alm)
hp.write_alm(randomized_alms_path/f'sum_alm.fits', rand_somma_alm, overwrite = True)
hp.write_map(randomized_path/f'sum.fits', hp.alm2map(rand_somma_alm, nside = nside), overwrite = True)

gauss_somma_alm = hp.synalm(hp.alm2cl(somma_alm))
hp.write_alm(gaussianized_alms_path/f'sum_alm.fits', gauss_somma_alm, overwrite = True)
hp.write_map(gaussianized_path/f'sum.fits', hp.alm2map(gauss_somma_alm, nside = nside), overwrite = True)




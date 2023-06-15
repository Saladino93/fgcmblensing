"""
Generates a simple mock point sources map.
"""

import numpy as np
import healpy as hp

import argparse

import os
import pathlib

SCRATCH = os.getenv('SCRATCH')

pointsourcesfolder = os.path.join(SCRATCH, 'mock_point_sources')
pointsourcesfolder = pathlib.Path(pointsourcesfolder)
pointsourcesfolder.mkdir(parents=True, exist_ok=True)

pointsourcesname = 'almpointsources'

get_our_dir = lambda nside, num, amplitude: os.path.join(pointsourcesfolder, f'nside{nside}_num{num}_amplitude{amplitude}')

parser = argparse.ArgumentParser(description='Generate mock point sources map and a randomized version.')
parser.add_argument('nside', type=int, help='nside of the map', default=2048)
parser.add_argument('num', type=int, help='number of sources', default=10000)
parser.add_argument('amplitude', type=float, help='amplitude of sources', default=200)

args = parser.parse_args()
nside = args.nside
NN = args.num
Nsources = np.random.poisson(NN)
amplitude = args.amplitude

Nsources = np.random.poisson(NN)
positions = np.random.randint(0, hp.nside2npix(nside), Nsources)
amplitudes = np.random.poisson(amplitude, Nsources)

map = np.zeros(hp.nside2npix(nside))
map[positions] = amplitudes

alms = hp.map2alm(map)

hp.write_alm(get_our_dir(nside, NN, amplitude) + pointsourcesname + '.fits', alms, overwrite=True)

def randomizing_fg(mappa: np.ndarray):
     f = lambda z: np.abs(z) * np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
     return f(mappa)

almrandomized = randomizing_fg(alms)

hp.write_alm(get_our_dir(nside, NN, amplitude) + pointsourcesname + '_randomized.fits', alms, overwrite=True)


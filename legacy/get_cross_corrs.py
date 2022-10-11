import pathlib
import numpy as np

import matplotlib.pyplot as plt

import szar
from szar import foregrounds as sfg

import healpy as hp

import scipy.stats as stats


def compute_cells(ells, cls, lmin = 10, lmax = 4000, delta_ell = 30):
    
        bins = np.arange(lmin, lmax, delta_ell)

        sums = stats.binned_statistic(ells, ells, statistic='sum', bins = bins)
        cl = stats.binned_statistic(ells, ells*cls, statistic='sum', bins = bins)
        cl = cl[0] / sums[0]
        
        ells = (bins[1:]+bins[:-1])/2.0

        return ells, cl


source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

extra_counts = ''
extra_counts = '_weighted'
counts = np.load(source_dir/f'counts{extra_counts}.npy')
maschera = np.load(source_dir/'maschera.npy')


nside = 4096


kappa_original_alm = hp.read_alm(source_dir/'kappa_alm.fits')

version = 'qe'
kappa_reconstructed_alm = np.load(source_dir/f'kappa_reconstructed_{version}.npy')
kappa_fg_reconstructed_alms = np.load(source_dir/f'kappa_fg_reconstructed_{version}.npy')
kappa_fg_masked_recon_alms = np.load(source_dir/f'kappa_fg_masked_reconstructed_{version}.npy')

filename = f'sehgalconfig/spectra_{version}{extra_counts}.txt'

try:
    ells, clkk_rec, clkk_input, cl_kk_input_output, clkg_fg_auto, clkg_fg_masked_auto, clkg_input, clkg_rec, clkg_masked_rec = np.loadtxt(filename, unpack = True)
except:

    print(f'Calculating spectra for {version} with {extra_counts} counts...')
    clkk_rec = hp.alm2cl(kappa_reconstructed_alm)
    clkk_input = hp.alm2cl(kappa_original_alm)
    cl_kk_input_output = hp.alm2cl(kappa_reconstructed_alm, kappa_original_alm)

    clkg_fg_auto = hp.alm2cl(kappa_fg_reconstructed_alms, kappa_fg_reconstructed_alms)
    clkg_fg_masked_auto = hp.alm2cl(kappa_fg_masked_recon_alms, kappa_fg_masked_recon_alms)

    mlmax = 6000
    counts_alm = hp.map2alm(counts, lmax = mlmax)

    clkg_input = hp.alm2cl(counts_alm, kappa_original_alm)
    clkg_rec = hp.alm2cl(counts_alm, kappa_reconstructed_alm)
    clkg_masked_rec = hp.alm2cl(counts_alm, kappa_fg_masked_recon_alms)

    ells = np.arange(0, len(clkg_masked_rec))

    np.savetxt(filename, np.c_[ells, clkk_rec, clkk_input, cl_kk_input_output, clkg_fg_auto, clkg_fg_masked_auto, clkg_input, clkg_rec, clkg_masked_rec])






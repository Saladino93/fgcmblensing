import pathlib

import healpy as hp

import numpy as np

mlmax = 6000

source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

maschera = np.load(source_dir/'maschera.npy')

kappa_name = 'healpix_4096_KappaeffLSStoCMBfullsky.fits'
cmb_name = 'Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits'
ksz_name = '148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits'
tsz_name = 'tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits'


cmboutname = 'cmb'
kappaoutname = 'kappa'
kszoutname = 'ksz'
tszoutname = 'tsz'

#names = [kappa_name, cmb_name]
names = [ksz_name, tsz_name]

#nomi = [kappaoutname, cmboutname]
nomi = [kszoutname, tszoutname]

masked = True
extra = '_masked' if masked else ''
maschera = maschera if masked else (maschera*0.+1.)

maps = [hp.read_map(source_dir/name)*maschera for name in names]
alms = [hp.map2alm(mappa, lmax = mlmax) for mappa in maps]


[hp.write_alm(source_dir/f'{nome}_alm{extra}.fits', alm) for nome, alm in zip(nomi, alms)]

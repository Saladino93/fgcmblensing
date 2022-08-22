import pathlib

import healpy as hp

mlmax = 7000

source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

kappa_name = 'healpix_4096_KappaeffLSStoCMBfullsky.fits'
cmb_name = 'Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits'
ksz_name = '148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits'
tsz_name = 'tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits'
radio_name = '145_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp.fits'
cib_name = '145_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75.fits'

cmboutname = 'cmb'
kappaoutname = 'kappa'
kszoutname = 'ksz'
tszoutname = 'tsz'

radiooutname = 'radio'
ciboutname = 'cib'

#names = [kappa_name, cmb_name]
#names = [ksz_name, tsz_name]
names = [kappa_name, cmb_name, ksz_name, tsz_name, radio_name, cib_name]
fgnames = [ksz_name, tsz_name, radio_name, cib_name]


#nomi = [kappaoutname, cmboutname]
#nomi = [kszoutname, tszoutname]
nomi = [kappaoutname, cmboutname, kszoutname, tszoutname, radiooutname, ciboutname]
fgnomi = [kszoutname, tszoutname, radiooutname, ciboutname]

maps = [hp.read_map(source_dir/name) for name in names]
alms = [hp.map2alm(mappa, lmax = mlmax) for mappa in maps]

[hp.write_alm(source_dir/f'{nome}_alm.fits', alm, overwrite = True) for nome, alm in zip(nomi, alms)]


nu = 145
maskf = hp.read_map(source_dir/f'source_mask_{nu}GHz.fits')

maps = [hp.read_map(source_dir/name)*maskf for name in fgnames]
alms = [hp.map2alm(mappa, lmax = mlmax) for mappa in maps]

[hp.write_alm(source_dir/f'{nome}_masked_alm.fits', alm, overwrite = True) for nome, alm in zip(fgnomi, alms)]

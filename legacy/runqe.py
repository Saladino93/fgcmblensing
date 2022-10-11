import pathlib

from falafel import utils, qe

from pixell import lensing, curvedsky

import solenspipe

import pytempura

import healpy as hp

import numpy as np

import matplotlib.pyplot as plt

import read_config

import argparse

import SONoise


parser = argparse.ArgumentParser(description = 'Calculates lensed and unlensed CMB theory spectra')
parser.add_argument('-c','--configuration', help = 'Configuration file', required = True)
configuration = parser.parse_args().configuration

output_dir = pathlib.Path(configuration)
output_dir = pathlib.Path(output_dir.stem)

config = read_config.ConfigurationReader(configuration)


##### SOME UTILITY FUNCTION ####

def gauss_beam(ell,fwhm):
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))


def dummy_teb(alms):
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def filter_alms(alms, tcls, kappa_lmin, kappa_lmax):
    if len(alms)!=3:
        alms = dummy_teb(alms)
    alms_filtered = utils.isotropic_filter(alms,
            tcls, kappa_lmin, kappa_lmax, ignore_te = True)
    return alms_filtered

#### SOME CONSTANTS ####


default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10


#### FACTORS FOR tSZ ####

def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans

def tsz_factor_for_ymap(freq, tcmb=default_tcmb):
    return fnu(freq) * tcmb * 1e6

def process_tsz(comptony, freq, tcmb=default_tcmb):
    return tsz_factor_for_ymap(freq = freq, tcmb = tcmb) * comptony


##########################


nside = config.nside
mlmax = config.mlmax

#noise_sigma = 17.
#fwhm = 1.4

kappa_lmin, kappa_lmax = config.kappa_lmin, config.kappa_lmax


source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

kappa_name = 'healpix_4096_KappaeffLSStoCMBfullsky.fits'
cmb_name = 'Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits'

tsz_power = 'Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt'
ksz_power = 'kSZ_PS_Sehgal_healpix_Nside4096_DeltaT_uK.txt'

#names = [kappa_name, cmb_name]
#maps = [hp.read_map(source_dir/name) for name in names]

#alms = [hp.map2alm(mappa, lmax = mlmax) for mappa in maps]
cmboutname = 'cmb'
kappaoutname = 'kappa'
kszoutname = 'ksz'
tszoutname = 'tsz'
nomi = [kappaoutname, cmboutname, kszoutname, tszoutname]
#[hp.write_alm(source_dir/f'{nome}_alm.fits', alm) for nome, alm in zip(nomi, alms)]

extra = ''
alms = [hp.read_alm(source_dir/f'{nome}_alm{extra}.fits') for nome in nomi]

extra = '_masked'
alms += [hp.read_alm(source_dir/f'{nome}_alm{extra}.fits') for nome in [tszoutname]]

ells = np.arange(mlmax+1)

ell, fgs = np.loadtxt(output_dir/config.fgname, unpack = True)
fgpower = np.interp(ells, ell, fgs)


Noise = SONoise.SONoiseReader(output_dir/config.noise_name, nus = config.nus, cross_nus = config.cross_nus)
nu0 = config.nu0
freq = nu0
nu1, nu2 = [nu0]*2
elln, nl = Noise.ell, Noise.get_noise(nu1 = nu1)

Nl_tt = np.interp(ells, elln, nl) #beam deconvolved noise
#HAVE TO INCLUDE FOREGROUND POWER TOO
Nl_tt = np.nan_to_num(Nl_tt+fgpower)
nells = {"TT": Nl_tt, "EE": 2*Nl_tt, "BB": 2*Nl_tt}

px = qe.pixelization(nside = config.nside)

ucls, tcls = utils.get_theory_dicts(grad = True, nells = nells, lmax = mlmax)

_, ls, Als, R_src_tt, Nl_g, Nl_c, Nl_g_bh = solenspipe.get_tempura_norms(
        'TT', 'TT', ucls, tcls, kappa_lmin, kappa_lmax, mlmax,  )

R_src_tt = pytempura.get_cross('SRC','TT', ucls, tcls, kappa_lmin, kappa_lmax,
                                   k_ellmax = mlmax)

norm_stuff = {"ls": ls, "Als": Als, "R_src_tt": R_src_tt,
                  "Nl_g": Nl_g, "Nl_c": Nl_c, "Nl_g_bh": Nl_g_bh,
    }

qfunc = solenspipe.get_qfunc(px, ucls, mlmax, "TT", Al1 = Als['TT'])
qfunc_bh = solenspipe.get_qfunc(px, ucls, mlmax, "TT", est2 = 'SRC', Al1 = Als['TT'], 
                                Al2 = Als['src'], R12 = R_src_tt)

kappa_alm, cmb_alm, ksz_alm, tsz_alm, tsz_masked_alm = alms
tsz_alm = process_tsz(tsz_alm, freq)
tsz_masked_alm = process_tsz(tsz_masked_alm, freq)

cmb_alm = utils.change_alm_lmax(cmb_alm, mlmax)
cmb_alm_filtered = filter_alms(cmb_alm, tcls, kappa_lmin, kappa_lmax)

fg_alms = utils.change_alm_lmax(tsz_alm, mlmax)
fg_alms_filtered = filter_alms(fg_alms, tcls, kappa_lmin, kappa_lmax)

fg_masked_alms = utils.change_alm_lmax(tsz_masked_alm, mlmax)
fg_masked_alms_filtered = filter_alms(fg_masked_alms, tcls, kappa_lmin, kappa_lmax)


versions = ['qe', 'bh']
functions = [qfunc, qfunc_bh]

for function, version in zip(functions, versions):
    phi_recon_alms = function(cmb_alm_filtered, cmb_alm_filtered)
    kappa_recon_alms = lensing.phi_to_kappa(phi_recon_alms)

    phi_fg_recon_alms = function(fg_alms_filtered, fg_alms_filtered)
    kappa_fg_recon_alms = lensing.phi_to_kappa(phi_fg_recon_alms)

    phi_fg_masked_recon_alms = function(fg_masked_alms_filtered, fg_masked_alms_filtered)
    kappa_fg_masked_recon_alms = lensing.phi_to_kappa(phi_fg_masked_recon_alms)

    cl_kk_input_output = curvedsky.alm2cl(kappa_recon_alms[0], kappa_alm)
    cl_kk_input = curvedsky.alm2cl(kappa_alm)

    np.save(source_dir/f'kappa_reconstructed_{version}', kappa_recon_alms[0])
    np.save(source_dir/f'kappa_fg_reconstructed_{version}', kappa_fg_recon_alms[0])
    np.save(source_dir/f'kappa_fg_masked_reconstructed_{version}', kappa_fg_masked_recon_alms[0])

    plt.loglog(cl_kk_input_output, label = '$\hat{\kappa}\kappa$')
    plt.loglog(cl_kk_input, label = '$\kappa\kappa$')
    plt.legend()
    plt.savefig(f'reconstruction_{version}.png')
    plt.close()


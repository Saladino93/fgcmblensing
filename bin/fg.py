import pathlib

import healpy as hp

import numpy as np

import matplotlib.pyplot as plt

import pathlib

from falafel import utils, qe

import solenspipe

import pytempura

import constants as const

from pixell import lensing, curvedsky


def gauss_beam(ell: np.ndarray, fwhm: float):
    '''
    Parameters
    ----------
    ell: np.ndarray
    fwhm: float, in arcmin

    Returns
    -------
    gauss_beam: np.ndarray
    '''
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))


def dummy_teb(alms):
    '''
    Creates a list of maps with the order [Tlm, Elm, Blm] and Elm, Blm = 0, 0
    '''
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def filter_alms(alms, tcls, lmin, lmax):
    '''
    Takes input alms, and makes an isotropic filtering with tcls
    '''
    if len(alms)!=3:
        alms = dummy_teb(alms)
    alms_filtered = utils.isotropic_filter(alms,
            tcls, lmin, lmax, ignore_te = True)
    return alms_filtered


def fnu(nu, tcmb = const.default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = const.H_CGS*(1e9*nu)/(const.K_CGS*tcmb)
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans

def tsz_factor_for_ymap(freq, tcmb = const.default_tcmb):
    return fnu(freq) * tcmb * 1e6

def process_tsz(comptony, freq, tcmb = const.default_tcmb):
    return tsz_factor_for_ymap(freq = freq, tcmb = tcmb) * comptony




mlmax = 7000

freq = 145

kappa_lmin = 100

kappa_lmaxes = [3000]#, 3500, 4000, 4500]

nside = 4096

source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

kappa_name = 'healpix_4096_KappaeffLSStoCMBfullsky.fits'

cmb_name = 'Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits'

tsz_name = 'tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits'
ksz_name = '148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits'

mask_name = 'source_mask_145GHz.fits'

tsz_map = hp.read_map(source_dir/tsz_name)

maskf = hp.read_map(source_dir/mask_name)

tsz_map_masked = tsz_map*maskf

cmboutname = 'cmb'
kappaoutname = 'kappa'

nomi = [kappaoutname, cmboutname]
kappa_alm, cmb_alm = [hp.read_alm(source_dir/f'{nome}_alm.fits') for nome in nomi]
kappa_alm = kappa_alm.astype(np.complex128)

key = 'cib'
tsz_map_alm = hp.read_alm(source_dir/f'{key}_alm.fits')#np.load(source_dir/'tsz_map_alm.npy')
tsz_map_masked_alm = hp.read_alm(source_dir/f'{key}_masked_alm.fits') #np.load(source_dir/'tsz_map_masked_alm.npy')

f = lambda size: np.exp(1j*np.random.uniform(0., 2.*np.pi, size = size))
size = len(tsz_map_alm)
factors = f(size)
tsz_map_randomized_alm = hp.almxfl(tsz_map_alm, factors) #hp.read_alm(f'{source_dir}/tsz_randomized_alm.fits')


allelementstosave = np.load('input_cmb_145.npy')
ells, lcmb, tsz, ksz, radio, cib, dust, nl145, totalcmb, totalnoisecmb = allelementstosave.T


ell = np.arange(mlmax+1)

fgs = tsz+ksz+radio+cib+dust
noise = fgs+nl145
noise = np.interp(ell, ells, noise)

noisepol = nl145
noisepol = np.interp(ell, ells, noisepol)

Nl_tt = np.nan_to_num(noise)

nells = {"TT": Nl_tt, "EE": 2*noisepol*0., "BB": 2*noisepol*0.}

nside = utils.closest_nside(3000)
print('NSIDE', nside)
px = qe.pixelization(nside = nside)
#px = solenspipe.get_sim_pixelization(3000, True)


ucls, tcls = utils.get_theory_dicts(grad = True, nells = nells, lmax = mlmax)

#_, ls, Als, R_src_tt, Nl_g, Nl_c, Nl_g_bh = solenspipe.get_tempura_norms(
#        est1 = 'TT', est2 = 'TT', ucls = ucls, tcls = tcls, lmin = kappa_lmin, lmax = kappa_lmax, mlmax = mlmax)

#R_src_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, kappa_lmin, kappa_lmax, k_ellmax = mlmax)

#norm_stuff = {"ls": ls, "Als": Als, "R_src_tt": R_src_tt,
#                  "Nl_g": Nl_g, "Nl_c": Nl_c, "Nl_g_bh": Nl_g_bh,
#    }
#np.save('norm_stuff', norm_stuff)

#norm_stuff = np.load('norm_stuff.npy', allow_pickle = True)
#Als = norm_stuff.item().get('Als')
#R_src_tt = norm_stuff.item().get('R_src_tt')

#qfunc2 = lambda X,Y: qe.qe_source(px, mlmax, fTalm=Y[0],xfTalm=X[0], profile = None)
#qfunc_shear = lambda X,Y: qe.qe_shear(px, mlmax, Talm = X[0], fTalm = Y[0])



#maps = [cmb_alm, process_tsz(tsz_map_alm, freq), process_tsz(tsz_map_masked_alm, freq)]

process_tsz = lambda x, y: x if key != 'tsz' else process_tsz

maps = [process_tsz(tsz_map_alm, freq), process_tsz(tsz_map_randomized_alm, freq)]

codes = [key, f'{key}_randomized'] #, 'tsz', 'tsz_masked']

for kappa_lmax in kappa_lmaxes:

    _, ls, Als, R_src_tt, Nl_g, Nl_c, Nl_g_bh = solenspipe.get_tempura_norms(
        est1 = 'TT', est2 = 'TT', ucls = ucls, tcls = tcls, lmin = kappa_lmin, lmax = kappa_lmax, mlmax = mlmax)

    opath = '/global/homes/o/omard/so-lenspipe/data/5_'
    e1, e2 = 'TT', 'TT'
    Nlg_bh_ = np.loadtxt(f'{opath}Nlg_bh_{e1}_{e2}.txt')

    #R_src_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, kappa_lmin, kappa_lmax, k_ellmax = mlmax)

    qfunc = solenspipe.get_qfunc(px, ucls, mlmax, 'TT', Al1 = Als['TT'])
    qfunc_bh = solenspipe.get_qfunc(px, ucls, mlmax, 'TT', est2 = 'SRC', Al1 = Als['TT'], 
                                    Al2 = Als['src'], R12 = R_src_tt)

    qfunc_bh = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='SRC',Al2=Als['src'],R12=R_src_tt)

    all_spectra = {}

    for map_alm, code in zip(maps, codes):

        print(f'Filtering {code}')
        input_alm = utils.change_alm_lmax(map_alm.astype(np.complex128), mlmax)
        input_alm_filtered = filter_alms(input_alm, tcls, kappa_lmin, kappa_lmax)

        versions = ['qe'] #['bh'] #['qe', 'bh'] 
        functions = [qfunc] #[qfunc_bh] #[qfunc, qfunc_bh]

        vstuff = {}

        for function, version in zip(functions, versions):

            stuff = {}
            print('Reconstruct with', version)
            
            phi_recon_alms = function(input_alm_filtered, input_alm_filtered)
            #phi_recon_alms = function(Xdat, Xdat)    

            print('Convert to kappa')
            kappa_recon_alms = lensing.phi_to_kappa(phi_recon_alms)

            print('Convert to spectra')
            cl_kk_output_output = curvedsky.alm2cl(kappa_recon_alms[0])
            cl_kk_input_output = curvedsky.alm2cl(kappa_recon_alms[0], kappa_alm)
            cl_kk_input = curvedsky.alm2cl(kappa_alm)

            np.save(source_dir/f'kappa_reconstructed_{version}{code}_{kappa_lmax}', kappa_recon_alms[0])

            stuff['oo'] = cl_kk_output_output
            stuff['io'] = cl_kk_input_output
            stuff['ii'] = cl_kk_input

            vstuff[version] = stuff

        all_spectra[code] = vstuff

    np.save(f'all_spectra_{key}_randomized_{kappa_lmax}', all_spectra)
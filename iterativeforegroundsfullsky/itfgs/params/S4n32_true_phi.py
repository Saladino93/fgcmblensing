"""

Iterative reconstruction for CMB data.

Minimal example for TT only data.

Giulio sims map0_kappa_ecp262_dmn2_lmax8000_first.fits

with Born approx and non linear effects

"""

import os
from os.path import join as opj
import numpy as np
import healpy as hp

import plancklens

from plancklens import utils, qresp, qest, qecl
from plancklens.qcinv import cd_solve
from plancklens.sims import phas
from plancklens.filt import filt_simple, filt_util

#from delensalot.sims import phas

from delensalot.core.helper import utils_scarf
from delensalot.utility import utils_sims

from delensalot.core.iterator import cs_iterator as scarf_iterator, steps
from delensalot.core.iterator import cs_iterator_fast as scarf_iterator
from delensalot.core.iterator import cs_iterator_multi_bh as scarf_iterator_multi_bh
from delensalot.utils import cli
from delensalot.utility.utils_hp import gauss_beam, almxfl, alm_copy, Alm
from delensalot.core.opfilt.MAP_opfilt_iso_t import alm_filter_nlev_wl as alm_filter_tt_wl

from delensalot.core.opfilt.MAP_opfilt_iso_tp import alm_filter_nlev_wl as alm_filter_tp_wl

from delensalot.core.opfilt.MAP_opfilt_iso_p import alm_filter_nlev_wl as alm_filter_ee_wl

#from delensalot.core.opfilt.MAP_opfilt_aniso_t import alm_filter_ninv_wl as alm_filter_tt_wl_aniso
from delensalot.core.opfilt.MAP_bh_opfilt_aniso_t import alm_filter_ninv_wl as alm_filter_aniso_bh


from lenspyx.remapping.deflection_029 import deflection
from lenspyx.remapping import utils_geom
from lenspyx.lensing import get_geom

from itfgs.sims.sims_postborn import sims_postborn
import itfgs.sims.sims_cmbs as simsit

from plancklens.helpers import mpi

import ducc0


def camb_clfile_gradient(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.
    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.
    """
    cols = np.loadtxt(fname).transpose()
    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    return cls

class SehgalSim(sims_postborn):

    kappakey = 'kappa'

    def __init__(self, sims: dict, **kwargs):
        super().__init__(**kwargs)
        self.sims = sims
    
    def get_sim_kappa(self, idx, verbose: bool = True):
        if verbose:
            print('Getting special kappa!')
        nome = self.sims[self.kappakey](idx)
        return hp.read_map(nome)
    
    def get_sim_kappa_alm(self, idx, verbose: bool = True):
        if verbose:
            print('Getting special kappa alm!')
        nome = self.sims[self.kappakey](idx)
        return hp.read_alm(nome)

include_fgs_power = False

baseSehgal = opj(os.environ['SCRATCH'], 'SKYSIMS/GIULIOSIMS/')
baseWebsky = opj(os.environ['SCRATCH'], 'SKYSIMS/WEBSKYSIMS/')
#baseSehgal = opj(os.environ['SCRATCH'], 'SehgalSims')

suffix = 'S4Giulio' # descriptor to distinguish this parfile from others...
suffixWebsky = 'S4Websky'

casostd = ""
casorand = "rand"
casogauss = "gauss"

casostdflip = "bornflipped"

casorandlog = "randlog"
casolog = "log"

casopostlog = "postlog"
casopostlogrand = "postlogrand"

casorandlogdoubleskew = "randlogdoubleskew"
casologdoubleskew = "logdoubleskew"
casogausslogdoubleskew = "gausslogdoubleskew"

casorandlogfactorskew = "randlogfactorskew"
casologfactorskew = "logfactorskew"
casogausslogfactorskew = "gausslogfactorskew"

casopostborn = "postborn"
casopostbornrand = "postbornrand"
casopostborngauss = "postborngauss"

casorot = "rot"
casorotrand = "rotrand"
casorotgauss = "rotgauss"

casowebskyborn = "websky"
casowebskybornrand = "webskyrand"
casowebskyborngauss = "webskygauss"

casowebskybornfgs = "webskyfgs"

cases = [casostd, casorand, casogauss, casorandlog, casolog, casorandlogdoubleskew, casologdoubleskew, casogausslogdoubleskew, casopostborn, casopostbornrand, casowebskyborn, casowebskybornrand, casowebskyborngauss]


def get_info(caso: str) -> tuple:

    extra_tlm = None

    if caso == casorand:

        suffixCMB = suffix+'BornRand5120plmin'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornRand5120plmin'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_first_randomized_alm.fits')

    elif caso == casogauss:
        suffixCMB = suffix+'BornGauss5120plmin'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornGauss5120plmin'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'bornGaussian/born_kappa_gauss_alm_{idx}.fits')

    elif caso == casostd:
        suffixCMB = suffix+'Born5120plmin'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'Born5120plmin'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_first_alm.fits')

        names = ['']
        SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]

    elif caso == casorotgauss:

        suffixCMB = suffix+'RotGauss5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'RotGauss5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'rotationGaussian/rotation_kappa_gauss_alm_{idx}.fits')

    elif caso == casorot:

        suffixCMB = suffix+'Rot5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'Rot5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_rotation_ecp262_dmn2_lmax8000_alm.fits')

    elif caso == casorotrand:

        suffixCMB = suffix+'RotRand5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'RotRand5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_rotation_ecp262_dmn2_lmax8000_randomized_alm.fits')


    elif caso == casostdflip:
        suffixCMB = suffix+'BornFlipped5120_3000_noiseless'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornFlipped5120_3000_noiseless'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_first_flipped_alm.fits')

        names = ['']
        SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]

    elif caso == casolog:
        suffixCMB = suffix+'BornLog5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornLog5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/lognormal_factor_1_idx_0_alm.fits')
    elif caso == casorandlog:
        suffixCMB = suffix+'BornRandLog5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornRandLog5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/lognormal_factor_1_randomized_idx_0_alm.fits')


    elif caso == casopostlog:

        suffixCMB = suffix+'PostBornLog5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'PostBornLog5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalpost/post_lognormal_alm_{0}.fits')

    elif caso == casopostlogrand:
        suffixCMB = suffix+'PostBornLogRand5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'PostBornLogRand5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalpost/post_lognormal_randomized_alm_{0}.fits')

    elif caso == casologdoubleskew:
        suffixCMB = suffix+'BornLogDoubleSkew5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornLogDoubleSkew5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/lognormal_factor_2_idx_{0}_alm.fits')

    elif caso == casorandlogdoubleskew:
        suffixCMB = suffix+'BornRandLogDoubleSkew5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornRandLogDoubleSkew5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/lognormal_factor_2_randomized_idx_{0}_alm.fits')

    elif caso == casogausslogdoubleskew:
        suffixCMB = suffix+'BornGaussLogDoubleSkew'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornGaussLogDoubleSkew'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/gaussian_lognormal_factor_2_randomized_idx_{idx}_alm.fits')


    elif caso == casologfactorskew:
        suffixCMB = suffix+'BornLogFactorSkew'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornLogFactorSkew'
        factor = 5
        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/first_lognormalfactor_{factor}_alm_0.fits')

    elif caso == casorandlogfactorskew:
        suffixCMB = suffix+'BornRandLogFactorSkew'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornRandLogFactorSkew'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/first_lognormalfactor_{factor}_randomized_alm_0.fits')
        
        
    elif caso == casopostborn:
        
        suffixCMB = suffix+'PostBorn5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'PostBorn5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_alm.fits')

        names = ['']
        SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]

    elif caso == casopostbornrand:
        
        suffixCMB = suffix+'PostBornRand5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'PostBornRand5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_randomized_alm.fits')

        names = ['']
        SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]

    elif caso == casopostborngauss:
        
        suffixCMB = suffix+'PostBornGauss5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'PostBornGauss5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'postbornGaussian/postborn_kappa_gauss_alm_{idx}.fits')

    elif caso == casowebskyborn:

        suffixCMB = suffixWebsky+'WebskyBorn'
        suffixCMBPhas = suffixWebsky
        suffixLensing = suffixWebsky+'WebskyBorn'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseWebsky, 'kap.fits')


    elif caso == casowebskybornfgs:

        suffixCMB = suffixWebsky+'WebskyBorn'
        suffixCMBPhas = suffixWebsky
        suffixLensing = suffixWebsky+'WebskyBornForegrounds'

        fgnames = ["ksz", "tsz_2048", "cib_nu0143"]
        fgnames = ["ksz", "cib_nu0143"]

        class Extra(object):

            def __init__(self, name, fgnames):
                self.name = name
                self.fgnames = fgnames

            def __call__(self, idx):
                return np.sum([hp.read_map(opj(baseWebsky, f'{fgname}.fits')) for fgname in self.fgnames], axis = 0)
            
            def get_name(self):
                return self.name
                
        extra_tlm = Extra('fgs', fgnames)


    elif caso == casowebskybornrand:

        suffixCMB = suffixWebsky+'WebskyBornRand'
        suffixCMBPhas = suffixWebsky
        suffixLensing = suffixWebsky+'WebskyBornRand'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseWebsky, 'kap_randomized.fits')

    elif caso == casowebskyborngauss:

        suffixCMB = suffixWebsky+'WebskyBornGauss'
        suffixCMBPhas = suffixWebsky
        suffixLensing = suffixWebsky+'WebskyBornGauss'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseWebsky, f'websky_kappa_gauss_{idx}.fits')


    else:
        raise ValueError('caso not recognized')

    
    return suffixCMB, suffixCMBPhas, suffixLensing, SimsShegalDict, extra_tlm


def get_all(case: str):
   
    suffixCMB, suffixCMBPhas, suffixLensing, SimsShegalDict, extra_tlm = get_info(case)

    print("Working on case", case, "with suffix", suffixCMB, suffixCMBPhas, suffixLensing)

    names = ['']
    SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]

    #os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS', '8')
    main_dir = 'n32OFFICIAL'

    SIMDIR = opj(os.environ['SCRATCH'], main_dir, suffixCMB, 'cmbs')  # This is where the postborn are (or will be saved)
    lib_dir_CMB = opj(os.environ['SCRATCH'], main_dir, suffixCMBPhas, 'cmbs') #this is where I store phas, if already computed
    TEMP =  opj(os.environ['SCRATCH'], main_dir, suffixLensing, 'lenscarfrecs')

    #lib_dir_CMB = "/users/odarwish/scratch/oldn32/S4Giulio/cmbs"
    #SIMDIR = "/users/odarwish/scratch/oldn32/S4GiulioBornGauss5120/cmbs"
    print("SIMDIR: ", SIMDIR)
    print("lib_dir_CMB: ", lib_dir_CMB)
    print("TEMP: ", TEMP)

    fgs = 0.

    if "websky" in case:
        print("Cosmology for", case)
        cls_path = opj(os.environ['HOME'], 'fgcmblensing', 'input', 'websky')
        cls_unl = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lenspotentialCls_websky.dat'))
        cls_len = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lensedCls_websky.dat'))
        cls_grad = camb_clfile_gradient(opj(cls_path, 'new_lensedCMB_dmn1_lensedgradCls_websky.dat'))
    else:
        cls_path = opj(os.environ['HOME'], 'fgcmblensing', 'input', 'giulio')
        cls_unl = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lenspotentialCls.dat'))
        cls_len = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lensedCls.dat'))
        cls_grad = camb_clfile_gradient(opj(cls_path, 'lensedCMB_dmn1_lensedgradCls.dat'))



    ll = [cls_unl, cls_len, cls_grad]
    for l in ll:
        for k, v in l.items():
            l[k] = np.nan_to_num(v)

    ll = np.arange(0, len(cls_len['tt']), 1)
    cls_foregrounds = 0.

    lmax_ivf, mmax_ivf, beam, nlev_t, nlev_p = (4000, 4000, 1., 1., 1. * np.sqrt(2.))

    nlev_t_filter = nlev_t

    lmin_tlm, lmin_elm, lmin_blm = (10, 10, 10) # The fiducial transfer functions are set to zero below these lmins
    #lmin_blm = 200
    # for delensing useful to cut much more B. It can also help since the cg inversion does not have to reconstruct those.

    lmax_phi, mmax_phi = (5120, 5120)
    lmax_qlm, mmax_qlm = (lmax_phi, mmax_phi) # Lensing map is reconstructed down to this lmax and mmax
    # NB: the QEs from plancklens does not support mmax != lmax, but the MAP pipeline does
    lmax_unl, mmax_unl = (5120, 5120) # Delensed CMB is reconstructed down to this lmax and mmax

    analysis_info = {"lmin_tlm": lmin_tlm, "lmax_ivf": lmax_ivf, "mmax_ivf": mmax_ivf, "beam": beam, "nlev_t": nlev_t, "nlev_p": nlev_p,
                     "lmax_phi": lmax_phi, "mmax_phi": mmax_phi, "lmax_qlm": lmax_qlm, "mmax_qlm": mmax_qlm}

    #----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations
    nside = 2048 if "websky" in case else 4096#CHECK
    zbounds     = (-1.,1.) # colatitude sky cuts for noise variance maps (We could exclude all rings which are completely masked)
    
    geominfo = ('healpix', {'nside': nside})
    geominfo_defl = ('thingauss', {'lmax': 4200 + 300, 'smax': 2})
    
    lenjob_geometry = get_geom(geominfo)
    lenjob_geometry_defl = get_geom(geominfo_defl)

    lensres = 0.7  # Deflection operations will be performed at this resolution
    Lmin = 1 # The reconstruction of all lensing multipoles below that will not be attempted
    stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5) # handler of the size steps in the MAP BFGS iterative search
    mc_sims_mf_it0 = np.array([]) # sims to use to build the very first iteration mean-field (QE mean-field) Here 0 since idealized


    # Multigrid chain descriptor
    chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
    libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
    cmb_dir = opj(SIMDIR)
    #------------------

    # Fiducial model of the transfer function
    transf_tlm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
    transf_elm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
    transf_blm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
    transf_d = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}

    ll = np.arange(0, len(cls_len['tt']), 1)
    fgs = 0.

    # Isotropic approximation to the filtering (used eg for response calculations)
    ftl =  cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t_filter / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2) + fgs) * (transf_tlm > 0)
    fel =  cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
    fbl =  cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

    # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
    ftl_unl =  cli(cls_unl['tt'][:lmax_ivf + 1] + (nlev_t_filter / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2) + fgs) * (transf_tlm > 0)
    fel_unl =  cli(cls_unl['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
    fbl_unl =  cli(cls_unl['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

    # -------------------------
    # ---- Input simulation libraries. Here we use the NERSC FFP10 CMBs with homogeneous noise and consistent transfer function
    #       We define explictly the phase library such that we can use the same phases for for other purposes in the future as well if needed
    #       I am putting here the phases in the home directory such that they dont get NERSC auto-purged
    pix_phas = phas.pix_lib_phas(opj(os.environ['SCRATCH'], main_dir, 'pixphas_nside%s'%nside), 3, (hp.nside2npix(nside),)) # T, Q, and U noise phases
    #       actual data transfer function for the sim generation:
    transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # (taking here full sims cmb's which are given to 4096)

    fixed_index_cmb = None
    print("Note, different cmb index!!", fixed_index_cmb)
    zero_noise = False
    fixed_noise_index = None #0 #this will allow to have always the same experimental noise realization
    print("Note, different noise index!!", fixed_noise_index)
    lmax_cmb = 4096
    dlmax = 1024

    libPHASCMB = phas.lib_phas(os.path.join(lib_dir_CMB, 'phas'), 3, lmax_cmb + dlmax)

    sims_cmb_len = SehgalSim(sims = SimsShegalDict, lib_dir = SIMDIR, lmax_cmb = lmax_cmb, cls_unl = cls_unl, dlmax = dlmax, lmin_dlm = 2, lib_pha = libPHASCMB, extra_tlm = extra_tlm)
    sims      = simsit.cmb_maps_nlev_sehgal(sims_cmb_len = sims_cmb_len, cl_transf = transf_dat, 
                                    nlev_t = nlev_t, nlev_p = nlev_p, nside = nside, pix_lib_phas = pix_phas, zero_noise = zero_noise, fixed_noise_index = fixed_noise_index, fixed_index = fixed_index_cmb)

    # Makes the simulation library consistent with the zbounds
    sims_MAP  = utils_sims.ztrunc_sims(sims, nside, [zbounds])
    # -------------------------

    ivfs   = filt_simple.library_fullsky_sepTP(opj(TEMP, 'ivfs'), sims, nside, transf_d, cls_len, ftl, fel, fbl, cache=True)
    #library_jTP, lib_dir, sim_lib, cl_weights, soltn_lib=None, cache=True

    # ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
    mc_sims_bias = np.arange(60, dtype=int)
    mc_sims_var  = np.arange(60, 300, dtype=int)
    
    fal = {}
    fal["tt"] = ftl
    fal["ee"] = fel
    fal["bb"] = fbl
    resplib = qresp.resp_lib_simple(opj(TEMP, 'qlms_dd'), lmax_ivf, cls_weight = cls_grad, cls_cmb = cls_len, fal = fal, lmax_qlm = lmax_qlm)
    qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs,   cls_len['te'], nside, lmax_qlm=lmax_qlm, resplib = resplib)
    qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)
    # -------------------------
    # This following block is only necessary if a full, Planck-like QE lensing power spectrum analysis is desired
    # This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.

    # This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
    ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
    ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

    ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
    ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

    #qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), ivfs, ivfs_d, cls_len['te'], nside, lmax_qlm=lmax_qlm)
    #qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), ivfs, ivfs_s, cls_len['te'], nside, lmax_qlm=lmax_qlm)

    #qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), qlms_ds, qlms_ds, np.array([]))  # for QE RDN0 calculations
    #qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
    # -------------------------


    def get_itlib(k:str, simidx:int, version:str, cg_tol:float, epsilon=1e-5):
        """Return iterator instance for simulation idx and qe_key type k
            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter
        """
        print("Sim index is", simidx)
        libdir_iterator = libdir_iterators(k, simidx, version)
        if not os.path.exists(libdir_iterator):
            os.makedirs(libdir_iterator)
            
        num_threads = 12
        tr = int(os.environ.get('OMP_NUM_THREADS', num_threads))
        print("Using", tr, "threads")
        cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
        cpp[:Lmin] *= 0.

        # QE mean-field fed in as constant piece in the iteration steps:
        mf_sims = np.unique(mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
        mf0 = qlms_dd.get_sim_qlm_mf(k, mf_sims)  # Mean-field to subtract on the first iteration:
        if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
            Nmf = len(mf_sims)
            mf0 = (mf0 - qlms_dd.get_sim_qlm(k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))

        path_plm0 = opj(libdir_iterator, 'phi_plm_it000.npy')
        path_plm0_QE_norm = opj(libdir_iterator, 'normalized_phi_plm_it000.npy')

        path_slm0 = opj(libdir_iterator, 's_slm_it000.npy')
        path_slm0_QE_norm = opj(libdir_iterator, 'normalized_s_slm_it000.npy')

        if not os.path.exists(path_plm0) or True:
            # We now build the Wiener-filtered QE here since not done already
            plm0  = qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:

            print("Using starting point the true lensing field, with Wiener filter.", flush = True)
            plm0_in = hp.read_alm(cmb_dir+f"/plm_in_{simidx}_lmax{lmax_qlm}.fits")

            if k == "p" or k == "p_bh_s":
                    cls_filter = {}
                    cls_filter["tt"] = cli(ftl)
                    cls_filter["ee"] = cli(fel)
                    cls_filter["bb"] = cli(fbl)
                    cls_filter["te"] = cls_len['te'][:lmax_ivf + 1]
                    fal = utils.cl_inverse(cls_filter)
            else:
                fal = {'e': fel, 'b': fbl, 't':ftl}


            plm0 -= mf0  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            #NOTE: RESPONSE OF CMB. Here I am using the grad-lensed response
            R = qresp.get_response(k, lmax_ivf, 'p', cls_weight = cls_len, cls_cmb = cls_grad, fal = fal, lmax_qlm=lmax_qlm)[0]
            np.savetxt(opj(libdir_iterator, "R.txt"), R)
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = cpp * utils.cli(cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, lmax_qlm, mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
            almxfl(plm0, utils.cli(R), mmax_qlm, True) # Normalized QE
            np.save(path_plm0_QE_norm, plm0)
            np.savetxt(opj(libdir_iterator, "WF.txt"), WF)
            almxfl(plm0, WF, mmax_qlm, True)           # Wiener-filter QE
            almxfl(plm0, cpp > 0, mmax_qlm, True)
            np.save(path_plm0, plm0)
            
            noise_l = utils.cli(R)
            np.random.seed(simidx)
            noise_lm = hp.synalm(noise_l, lmax = lmax_qlm)
            
            #plm0_in += noise_lm
            #print("Created noisy lm")

            #almxfl(plm0_in, WF, mmax_qlm, True)
            almxfl(plm0_in, cpp > 0, mmax_qlm, True)


        """if k == "ptt_bh_s":
            
            if not os.path.exists(path_slm0):
                print("Building QE for BH FOR SOURCE")
                bhkey = "stt_bh_p"
                source = "s"
                slm0 = qlms_dd.get_sim_qlm(bhkey, int(simidx))  #Unormalized quadratic estimate:
                smf0 = 0
                slm0 -= smf0  # MF-subtracted unnormalized QE
                Rs = qresp.get_response(bhkey, lmax_ivf, source, cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]

                slm0 = alm_copy(slm0,  None, lmax_qlm, lmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
                almxfl(slm0, utils.cli(Rs), lmax_qlm, True)
                np.save(path_slm0_QE_norm, slm0)
        """

        R = qresp.get_response(k, lmax_ivf, 'p', cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
        # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        WF = cpp * utils.cli(cpp + utils.cli(R))

        plm0 = np.load(path_plm0)

        #NOTE: UNLENSED RESPONSE HERE
        R_unl = qresp.get_response(k, lmax_ivf, 'p', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl_unl}, lmax_qlm=lmax_qlm)[0]
        #R = qresp.get_response(k, lmax_ivf, 'p', cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
        #np.savetxt("R_unl.txt", R_unl)
        #np.savetxt("R.txt", R)
        
        if k in ['p_p', 'ptt', 'p'] and 'wmfresp' in version:
            mf_resp = qresp.get_mf_resp(k, cls_unl, {'ee': fel_unl, 'bb': fbl_unl}, lmax_ivf, lmax_qlm)[0]
        else:
            print('*** setting mfresp it to zero')
            mf_resp = np.zeros(lmax_qlm + 1, dtype=float)
        # Lensing deflection field instance (initiated here with zero deflection)
        #ffi = remapping.deflection(lenjob_pbgeometry, lensres, np.zeros_like(plm0), mmax_qlm, tr, tr)
        ffi = deflection(lenjob_geometry_defl, np.zeros_like(plm0), mmax_qlm, numthreads=tr, verbosity=0, epsilon=epsilon)
        sht_job = utils_scarf.scarfjob()
        #sht_job.set_geometry(ninvjob_geometry)
        sht_job.set_geometry(lenjob_geometry)
        sht_job.set_triangular_alm_info(lmax_ivf, mmax_ivf)
        sht_job.set_nthreads(tr)
        if k in ['ptt', "ptt_bh_s"]:
            effective_noise = np.sqrt(nlev_t_filter**2.+fgs*(180 * 60 / np.pi) ** 2*transf_tlm**2.)
            filtr = alm_filter_tt_wl(effective_noise, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))

            datmaps_real = sims_MAP.get_sim_tmap(int(simidx))
            datmaps = sht_job.map2alm(datmaps_real)

            wflm0 = None

        if k in ["ptt"]:
            print("ptt standard case")
        elif k in ["ptt_bh_s"]:

            print("Building unnormalised MAP with delensalot anisotropic.")
            
            keyP = "p"
            keyS = "s"

            bh_starting = False
            only_bh = False
            bh_aniso_filter = False

            if bh_starting:
                slm0 = qlms_dd.get_sim_qlm("stt_bh_p", int(simidx))
                plm0 = qlms_dd.get_sim_qlm("ptt_bh_s", int(simidx))
            else:
                slm0 = qlms_dd.get_sim_qlm("stt", int(simidx))
                plm0 = qlms_dd.get_sim_qlm("ptt", int(simidx))

            mf0_p = np.zeros(hp.Alm.getsize(lmax_qlm))

            fconv = 180*60/np.pi
            fconv = fconv**2.
            pixarea = hp.nside2pixarea(nside)
            pixarea *= fconv
            invtotalnoise = np.nan_to_num(np.ones_like(datmaps_real)*pixarea/nlev_t**2.)
            
                        
            invtotalnoise = sims_MAP.ztruncify(invtotalnoise)

            ninv_geom = utils_scarf.Geom.get_healpix_geometry(nside, zbounds = zbounds)

            if bh_aniso_filter:
                filtr = alm_filter_aniso_bh(ninv_geom, invtotalnoise, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf), 8)
                datmaps = datmaps_real
            else:
                filtr = alm_filter_tt_wl(nlev_t, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))#, use_lmax_unl_for_transf = True)

            R_phi = qresp.get_response("ptt", lmax_ivf, 'p', cls_len, cls_grad,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
            R_s = qresp.get_response("stt", lmax_ivf, 's', cls_len, cls_grad,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
            R_phi_s = qresp.get_response("ptt", lmax_ivf, 's', cls_len, cls_grad,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
            R_s_phi = qresp.get_response("stt", lmax_ivf, 'p', cls_len, cls_grad,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]

            #pack all into a matrix
            R = np.array([[R_phi, R_s_phi], [R_phi_s, R_s]])
            #move last axes to the front
            R = np.moveaxis(R, -1, 0)

            #take inverse
            Rinv = R.copy()
            #select entries with invertible matrices
            Rinv[1:] = np.linalg.inv(R[1:]) #I would like to multiply this matrix by the gradient calculated at zero

            #unpack again
            R_inv_phi, R_inv_phi_s, R_inv_s_phi, R_inv_s = Rinv[:, 0, 0], Rinv[:, 0, 1], Rinv[:, 1, 0], Rinv[:, 1, 1]

            R_inv_phi_list = [R_inv_phi, R_inv_s_phi]
            R_inv_s_list = [R_inv_phi_s, R_inv_s]

            cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])

            keys = [keyP, keyS]
            lm_max_dlm = [(lmax_qlm, mmax_qlm), (lmax_qlm, mmax_qlm)]

            mf0 = [mf0_p, mf0_p]
            Rs0 = [R_inv_phi_list, R_inv_s_list]

            cth = [cpp, np.ones_like(cpp)*1e-16]
            rho2 = 0.
            factor_A = 1/(1-rho2)
            factor_B = rho2/(1-rho2)
            factors = [factor_A, factor_B]
            crossth = np.sqrt(cth[0]*cth[1]*rho2)
            ichhs_list = [[cli(cth[0])*factors[0], cli(crossth)*factors[1]], [cli(crossth)*factors[1], cli(cth[1])*factors[0]]]

            if bh_starting:
                Rs_response = qresp.get_response("stt_bh_p", lmax_ivf, "s", cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
                Rp_response = qresp.get_response("ptt_bh_s", lmax_ivf, "p", cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
            else:
                Rs_response = qresp.get_response("stt", lmax_ivf, "s", cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
                Rp_response = qresp.get_response("ptt", lmax_ivf, "p", cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]

            if only_bh:
                Rs_response_unl = qresp.get_response("stt_bh_p", lmax_ivf, 's', cls_unl, cls_unl,
                                            {'e': fel_unl, 'b': fbl_unl, 't': ftl_unl}, lmax_qlm=lmax_qlm)[0]
                Rp_response_unl = qresp.get_response("ptt_bh_s", lmax_ivf, 'p', cls_unl, cls_unl,
                                                {'e': fel_unl, 'b': fbl_unl, 't': ftl_unl}, lmax_qlm=lmax_qlm)[0]
                R_phi_s_unl = qresp.get_response("ptt_bh_s", lmax_ivf, 's', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
                R_s_phi_unl = qresp.get_response("stt_bh_p", lmax_ivf, 'p', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
            else:
                Rs_response_unl = qresp.get_response("stt", lmax_ivf, 's', cls_unl, cls_unl,
                                            {'e': fel_unl, 'b': fbl_unl, 't': ftl_unl}, lmax_qlm=lmax_qlm)[0]
                Rp_response_unl = qresp.get_response("ptt", lmax_ivf, 'p', cls_unl, cls_unl,
                                                {'e': fel_unl, 'b': fbl_unl, 't': ftl_unl}, lmax_qlm=lmax_qlm)[0]
                R_phi_s_unl = qresp.get_response("ptt", lmax_ivf, 's', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
                R_s_phi_unl = qresp.get_response("stt", lmax_ivf, 'p', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl}, lmax_qlm=lmax_qlm)[0]

            slm0 = alm_copy(slm0,  None, lmax_qlm, lmax_qlm)
            plm0 = alm_copy(plm0,  None, lmax_qlm, lmax_qlm)

            almxfl(slm0, utils.cli(Rs_response), lmax_qlm, True)
            almxfl(plm0, utils.cli(Rp_response), lmax_qlm, True)

            WFpp = cpp * utils.cli(cpp + utils.cli(Rp_response))
            WFss = cth[1] * utils.cli(cth[1] + utils.cli(Rs_response))

            almxfl(slm0, WFss, lmax_qlm, True)
            almxfl(slm0, cpp > 0, mmax_qlm, True)
            almxfl(plm0, WFpp, lmax_qlm, True)
            almxfl(plm0, cpp > 0, mmax_qlm, True)

            #this will be my starting point!!
            sol0 = [plm0, slm0]
            Rs0 = [Rp_response_unl, Rs_response_unl]

            extra = 1
            extra_cross = 1

            signal_matrix = np.array([[(cth[0]), (crossth)*extra_cross], [(crossth)*extra_cross, (cth[1])]])

            signal_matrix = np.moveaxis(signal_matrix, -1, 0)
            non_zero = cli(cth[0])>0
            inv_signal_matrix = signal_matrix.copy()

            inv_signal_matrix[non_zero, ...] = np.linalg.inv(signal_matrix[non_zero, ...])
            inv_signal_matrix = np.nan_to_num(inv_signal_matrix)

            response_matrix = np.array([[Rp_response_unl, R_s_phi_unl*extra], [R_phi_s_unl*extra, Rs_response_unl]])
            response_matrix = np.moveaxis(response_matrix, -1, 0)

            total_inv_curvature_matrix = inv_signal_matrix + response_matrix
            non_zero = Rp_response_unl>0
            total_curvature_matrix = total_inv_curvature_matrix.copy()
            total_curvature_matrix[non_zero, ...] = np.linalg.inv(total_inv_curvature_matrix[non_zero, ...])
            total_curvature_matrix = np.nan_to_num(total_curvature_matrix)
            total_curvature_matrix[0, ...] *= 0
            print("Done!")

            standard = cli(Rp_response_unl + cli(cth[0]))
            key_pair = (keyP, keyS)

            np.save(path_plm0, np.append(plm0, slm0))

        elif k in ['p_p', 'p_eb']:
            wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
            assert np.all(transf_elm == transf_blm), 'This is not supported by the alm_filter_nlev_wl (but easy to fix)'
            # Here multipole cuts are set by the transfer function (those with 0 are not considered)
            filtr = alm_filter_ee_wl(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                    wee=wee, transf_b=transf_blm, nlev_b=nlev_p)
            

            # dat maps must now be given in harmonic space in this idealized configuration
            datmaps = np.array(sht_job.map2alm_spin(sims_MAP.get_sim_pmap(int(simidx)), 2))

            wflm0 = None

        elif k in ['pee']:
            wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
            assert np.all(transf_elm == transf_blm), 'This is not supported by the alm_filter_nlev_wl (but easy to fix)'
            # Here multipole cuts are set by the transfer function (those with 0 are not considered)
            filtr = alm_filter_ee_wl(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
            
            # dat maps must now be given in harmonic space in this idealized configuration
            datmaps = np.array(sht_job.map2alm_spin(sims_MAP.get_sim_pmap(int(simidx)), 2))[0]

            wflm0 = None

        elif k in ["p"]:
            filtr = alm_filter_tp_wl(nlev_t, nlev_p, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf), transf_e = transf_elm, transf_b = transf_blm, nlev_b = nlev_p)

            datmapsT = sht_job.map2alm(sims_MAP.get_sim_tmap(int(simidx)))
            datmapsP = np.array(sht_job.map2alm_spin(sims_MAP.get_sim_pmap(int(simidx)), 2))
            datmaps = np.array([datmapsT, datmapsP[0], datmapsP[1]])
            wflm0 = lambda : np.zeros((2, Alm.getsize(filtr.lmax_sol, filtr.mmax_sol)), dtype=complex).squeeze()
            
        elif k in ["p_bh_s"]:
            return None

        else:
            assert 0

        k_geom = filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
        # Sets to zero all L-modes below Lmin in the iterations:
        #NOTE: IS USING THE R_UNL RESPONSE TO OBTAIN ~ (1/Cpp + 1/N0)^-1 OK as first response?

        
        if k == "ptt_bh_s":
            iterator = scarf_iterator_multi_bh.iterator_cstmf(libdir_iterator, keyP, lm_max_dlm, datmaps,
                                  sol0, mf0, Rs0, cth, key_pair, cls_unl, filtr, k_geom,
                                  chain_descrs(lmax_unl, cg_tol), stepper, ichhs_list = ichhs_list, wflm0=None, pp_h0s_matrix = total_curvature_matrix)
        else:
            """iterator = scarf_iterator.iterator_cstmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
                    plm0_in, mf_resp, R_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
                    ,mf0=mf0, wflm0 = wflm0)"""
            iterator = scarf_iterator.iterator_cstmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
                    plm0_in, mf0 = mf0, pp_h0 = R_unl, cpp_prior = cpp, cls_filt = cls_unl, ninv_filt = filtr, k_geom = k_geom, 
                    chain_descr = chain_descrs(lmax_unl, cg_tol), stepper = stepper
                    ,wflm0 = wflm0)
        return iterator
    
    return get_itlib, libdir_iterators, chain_descrs, lmax_unl, analysis_info, sims_cmb_len

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=7., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')
    parser.add_argument('-eps', dest='epsilon', type=float, default=7., help='-log10 of lensing accuracy')
    parser.add_argument('-case', dest='case', type=str, default="", help='case')
    parser.add_argument('-alloc', dest='alloc', type=int, default=0, help='memory allocation in GB')

    args = parser.parse_args()
    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one


    if args.alloc:
        if ducc0.misc.preallocate_memory(args.alloc):
            print('gclm2lenmap: allocated %s GB'%args.alloc)
        else:
            print('gclm2lenmap: allocation of %s GB failed'%args.alloc)

    get_itlib, libdir_iterators, chain_descrs, lmax_unl, _, _ = get_all(args.case)

    from plancklens.helpers import mpi
    
    #from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from delensalot.core.iterator.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        version = args.v + ('tol%.1f'%args.tol) * (args.tol != 5.) + ('eps%.1f'%args.epsilon) * (args.epsilon != 5.)
        lib_dir_iterator = libdir_iterators(args.k, idx, version)
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            jobs.append(idx)

    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.k, idx, args.v, 1., epsilon=10 ** (- args.epsilon))
            if itlib is not None:
                for i in range(args.itmax + 1):
                    print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                    print("****Iterator: setting solcond to %s ****"%soltn_cond(i))

                    itlib.chain_descr  = chain_descrs(lmax_unl, tol_iter(i))
                    itlib.soltn_cond   = soltn_cond(i)
                    print("doing iter " + str(i))
                    itlib.iterate(i, 'p')

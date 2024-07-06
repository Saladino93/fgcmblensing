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
from plancklens.sims import maps, phas
from plancklens.filt import filt_simple, filt_util

#from delensalot.sims import phas
#from delensalot.utility import utils_sims
#from delensalot.utils import cli

class Alm:
    """alm arrays useful statics. Directly from healpy but excluding keywords


    """
    @staticmethod
    def getsize(lmax:int, mmax:int):
        """Number of entries in alm array with lmax and mmax parameters

        Parameters
        ----------
        lmax : int
          The maximum multipole l, defines the alm layout
        mmax : int
          The maximum quantum number m, defines the alm layout

        Returns
        -------
        nalm : int
            The size of a alm array with these lmax, mmax parameters

        """
        return ((mmax+1) * (mmax+2)) // 2 + (mmax+1) * (lmax-mmax)

    @staticmethod
    def getidx(lmax:int, l:int or np.ndarray, m:int or np.ndarray):
        """Returns index corresponding to (l,m) in an array describing alm up to lmax.

        In HEALPix C++ and healpy, :math:`a_{lm}` coefficients are stored ordered by
        :math:`m`. I.e. if :math:`\ell_{max}` is 16, the first 16 elements are
        :math:`m=0, \ell=0-16`, then the following 15 elements are :math:`m=1, \ell=1-16`,
        then :math:`m=2, \ell=2-16` and so on until the last element, the 153th, is
        :math:`m=16, \ell=16`.

        Parameters
        ----------
        lmax : int
          The maximum l, defines the alm layout
        l : int
          The l for which to get the index
        m : int
          The m for which to get the index

        Returns
        -------
        idx : int
          The index corresponding to (l,m)
        """
        return m * (2 * lmax + 1 - m) // 2 + l

    @staticmethod
    def getlmax(s:int, mmax:int or None):
        """Returns the lmax corresponding to a given healpy array size.

        Parameters
        ----------
        s : int
          Size of the array
        mmax : int
          The maximum m, defines the alm layout

        Returns
        -------
        lmax : int
          The maximum l of the array, or -1 if it is not a valid size.
        """
        if mmax is not None and mmax >= 0:
            x = (2 * s + mmax ** 2 - mmax - 2) / (2 * mmax + 2)
        else:
            x = (-3 + np.sqrt(1 + 8 * s)) / 2
        if x != np.floor(x):
            return -1
        else:
            return int(x)

def cli(cl):
    """Pseudo-inverse for positive cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def almxfl(alm:np.ndarray, fl:np.ndarray, mmax:int or None, inplace:bool):
    """Multiply alm by a function of l.

    Parameters
    ----------
    alm : array
      The alm to multiply
    fl : array
      The function (at l=0..fl.size-1) by which alm must be multiplied.
    mmax : None or int
      The maximum m defining the alm layout. Default: lmax.
    inplace : bool
      If True, modify the given alm, otherwise make a copy before multiplying.

    Returns
    -------
    alm : array
      The modified alm, either a new array or a reference to input alm,
      if inplace is True.

    """
    lmax = Alm.getlmax(alm.size, mmax)
    if mmax is None or mmax < 0:
        mmax = lmax
    assert fl.size > lmax, (fl.size, lmax)
    if inplace:
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            alm[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return
    else:
        ret = np.copy(alm)
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            ret[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return ret


def gauss_beam(fwhm:float, lmax:int):
    """Gaussian beam

    Parameters
    ----------
    fwhm : float
        The full-width half-maximum in radians of the beam
    lmax : int
        Maximum multipole of the beam

    Returns
    -------
    bl: ndarray
        The beam transfer function from multipole 0 to lmax


    """
    l = np.arange(lmax + 1)
    bl = np.exp(-0.5 * l * (l + 1) * (fwhm / np.sqrt(8.0 * np.log(2.0))) ** 2)
    return bl

def synalm(cl:np.ndarray, lmax:int, mmax:int or None):
    """Creates a Gaussian field alm from input cl array

    Parameters
    ----------
    cl : ndarray
        The power spectrum of the map
    lmax : int
        Maximum multipole simulated
    mmax: int
        Maximum m defining the alm layout, defaults to lmax if None or < 0

    Returns
    -------
    alm: ndarray
        harmonic coefficients of Gaussian field with lmax, mmax parameters

    """
    assert lmax + 1 <= cl.size
    if mmax is None or mmax < 0:
        mmax = lmax
    alm_size = Alm.getsize(lmax, mmax)
    alm = rng.standard_normal(alm_size) + 1j * rng.standard_normal(alm_size)
    almxfl(alm, np.sqrt(cl[:lmax+1] * 0.5), mmax, True)
    real_idcs = Alm.getidx(lmax, np.arange(lmax + 1, dtype=int), 0)
    alm[real_idcs] = alm[real_idcs].real * np.sqrt(2.)
    return alm

def alm2cl(alm:np.ndarray, blm:np.ndarray or None, lmax:int or None, mmax:int or None, lmaxout:int or None):
    """Auto- or cross-power spectrum between two alm arrays

    Parameters
    ----------
    alm : ndarray
        First alm harmonic coefficient array
    blm : ndarray or None
        Second alm harmonic coefficient array, can set this to same alm object or to None if same as alm
    lmax : int or None
        Maximum multipole defining the alm layout
    mmax: int or None
        Maximum m defining the alm layout, defaults to lmax if None or < 0
    lmaxout: the spectrum is calculated down to this multipole (defaults to lmax is None)

    Returns
    -------
    cl: ndarray
        (cross-)power of the input alm and blm arrays

    """
    if lmax is None: lmax = Alm.getlmax(alm.size, mmax)
    if lmaxout is None: lmaxout = lmax
    if mmax is None: mmax = lmax
    assert lmax == Alm.getlmax(alm.size, mmax), (lmax, Alm.getlmax(alm.size, mmax))
    lmaxout_ = min(lmaxout, lmax)
    if blm is not alm: # looks like twice faster than healpy implementation... ?!
        assert lmax == Alm.getlmax(blm.size, mmax), (lmax, Alm.getlmax(blm.size, mmax))
        cl = 0.5 * alm[:lmaxout_ + 1].real * blm[:lmaxout_ + 1].real
        for m in range(1, min(mmax, lmaxout_) + 1):
            m_idx = Alm.getidx(lmax,  m, m)
            a = alm[m_idx:m_idx + lmaxout_ - m + 1]
            b = blm[m_idx:m_idx + lmaxout_ - m + 1]
            cl[m:] += a.real * b.real + a.imag * b.imag
    else:
        a = alm[:lmaxout_ + 1].real
        cl = 0.5 * a.real * a.real
        for m in range(1, min(mmax, lmaxout_) + 1):
            m_idx = Alm.getidx(lmax,  m, m)
            a = alm[m_idx:m_idx + lmaxout_ - m + 1]
            cl[m:] += a.real * a.real + a.imag * a.imag
    cl *= 2. / (2 * np.arange(len(cl)) + 1)
    if lmaxout > lmaxout_:
        ret = np.zeros(lmaxout + 1, dtype=float)
        ret[:lmaxout_ + 1] = cl
        return ret
    return cl

def alm_copy(alm:np.ndarray, mmaxin:int or None, lmaxout:int, mmaxout:int):
    """Copies the healpy alm array, with the option to change its lmax

        Parameters
        ----------
        alm :ndarray
            healpy alm arrays to copy.
        mmaxin: int or None
            mmax parameter of input array (can be set to None or negative for default)
        lmaxout : int
            new alm lmax
        mmaxout: int
            new alm mmax


    """
    alms = np.atleast_2d(alm)
    ret = []
    for alm in alms:
        lmaxin = Alm.getlmax(alm.size, mmaxin)
        if mmaxin is None or mmaxin < 0: mmaxin = lmaxin
        if (lmaxin == lmaxout) and (mmaxin == mmaxout):
            ret.append(np.copy(alm))
        else:
            _ret = np.zeros(Alm.getsize(lmaxout, mmaxout), dtype=alm.dtype)
            lmax_min = min(lmaxout, lmaxin)
            for m in range(0, min(mmaxout, mmaxin) + 1):
                idx_in =  m * (2 * lmaxin + 1 - m) // 2 + m
                idx_out = m * (2 * lmaxout+ 1 - m) // 2 + m
                _ret[idx_out: idx_out + lmax_min + 1 - m] = alm[idx_in: idx_in + lmax_min + 1 - m]
            ret.append(_ret)
    ret = np.array(ret)
    if ret.shape[0] == 1:
        return ret[0]
    else:
        return ret
    


from lenspyx.lensing import get_geom

from itfgs.sims.sims_postborn import sims_postborn
import itfgs.sims.sims_cmbs as simsit

class mpi():
    rank = 0
    size = 1
    barrier = None


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

casopostborn = "postborn"
casopostbornrand = "postbornrand"
casopostborngauss = "postborngauss"

casowebskyborn = "websky"
casowebskybornrand = "webskyrand"
casowebskyborngauss = "webskygauss"

casowebskybornfgs = "webskyfgs"

cases = [casostd, casorand, casogauss, casorandlog, casolog, casorandlogdoubleskew, casologdoubleskew, casogausslogdoubleskew, casopostborn, casopostbornrand, casowebskyborn, casowebskybornrand, casowebskyborngauss]


def get_info(caso: str) -> tuple:

    extra_tlm = None

    if caso == casorand:

        suffixCMB = suffix+'BornRand5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornRand5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_first_randomized_alm.fits')

    elif caso == casogauss:
        suffixCMB = suffix+'BornGauss5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornGauss5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'bornGaussian/born_kappa_gauss_alm_{idx}.fits')

    elif caso == casostd:
        suffixCMB = suffix+'Born5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'Born5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_first_alm.fits')

        names = ['']
        SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]


    elif caso == casostdflip:
        suffixCMB = suffix+'BornFlipped5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornFlipped5120'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000_first_flipped_alm.fits')

        names = ['']
        SimsShegalDict[0] = [lambda idx: opj(baseSehgal, nome) for nome in names]

    elif caso == casolog:
        suffixCMB = suffix+'BornLogNew'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornLogNew'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/first_lognormal_alm_{idx}.fits')
    elif caso == casorandlog:
        suffixCMB = suffix+'BornRandLogNew'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'BornRandLogNew'

        SimsShegalDict = {}
        SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, f'lognormalfirst/first_lognormal_randomized_alm_{idx}.fits')


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
        
        
    elif caso == casopostborn:
        
        suffixCMB = suffix+'PostBorn5120'
        suffixCMBPhas = suffix
        suffixLensing = suffix+'PostBorn5120'

        SimsShegalDict = {}
        #SimsShegalDict['kappa'] = lambda idx: opj(baseSehgal, 'map0_kappa_ecp262_dmn2_lmax8000.fits')
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
    #stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5) # handler of the size steps in the MAP BFGS iterative search
    mc_sims_mf_it0 = np.array([]) # sims to use to build the very first iteration mean-field (QE mean-field) Here 0 since idealized


    # Multigrid chain descriptor
    chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
    libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
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

    zero_noise = False
    fixed_noise_index = 0 #this will allow to have always the same experimental noise realization
    lmax_cmb = 4096
    dlmax = 1024

    libPHASCMB = phas.lib_phas(os.path.join(lib_dir_CMB, 'phas'), 3, lmax_cmb + dlmax)

    sims_cmb_len = SehgalSim(sims = SimsShegalDict, lib_dir = SIMDIR, lmax_cmb = lmax_cmb, cls_unl = cls_unl, dlmax = dlmax, lmin_dlm = 2, lib_pha = libPHASCMB, extra_tlm = extra_tlm)
    sims      = simsit.cmb_maps_nlev_sehgal(sims_cmb_len = sims_cmb_len, cl_transf = transf_dat, 
                                    nlev_t = nlev_t, nlev_p = nlev_p, nside = nside, pix_lib_phas = pix_phas, zero_noise = zero_noise, fixed_noise_index = fixed_noise_index)

    # Makes the simulation library consistent with the zbounds
    #sims_MAP  = utils_sims.ztrunc_sims(sims, nside, [zbounds])
    # -------------------------

    ivfs   = filt_simple.library_fullsky_sepTP(opj(TEMP, 'ivfs'), sims, nside, transf_d, cls_len, ftl, fel, fbl, cache=True)

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
            
        num_threads = 24
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

        if not os.path.exists(path_plm0):
            print("Getting QEEEEEE")
            # We now build the Wiener-filtered QE here since not done already
            plm0  = qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:
            plm0 -= mf0  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            #NOTE: RESPONSE OF CMB. Here I am using the grad-lensed response
            R = qresp.get_response(k, lmax_ivf, 'p', cls_weight = cls_len, cls_cmb = cls_grad, fal = {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
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

        return None
    
    return get_itlib

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

    get_itlib = get_all(args.case)
    
    #from plancklens.helpers import mpi
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        jobs.append(idx)

    for idx in jobs[mpi.rank::mpi.size]:
        itlib = get_itlib(args.k, idx, args.v, 1., epsilon=10 ** (- args.epsilon))

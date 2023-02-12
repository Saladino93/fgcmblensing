"""Simulation module including GF non-linear kappa maps, although you can include also any kappa map, including gaussian ones.

You just have to give the path for the map.



# From GF:
# map0_kappa_ecp262_dmn2_lmax8000_first.fits: Born approximation + Non linear effects
# map0_kappa_ecp262_dmn2_lmax8000.fits : full post-Born+ Non linear effect
        
"""


import os
from lenscarf import utils_hp
import healpy as hp
import numpy as np

from plancklens import utils
from itfgs.sims import sims_cmbs

class sims_postborn(sims_cmbs.sims_cmb_len):
    """Simulations of CMBs each having the same GF postborn + non linear effect deflection kappa
        Args:
            lib_dir: the phases of the CMB maps and the lensed CMBs will be stored there
            lmax_cmb: cmb maps are generated down to this max multipole
            cls_unl: dictionary of unlensed CMB spectra
            dlmax, nside_lens, facres, nbands: lenspyx lensing module parameters
        This just redefines the sims_cmbs.sims_cmb_len method to feed the nonlinear kmap
    """
    def __init__(self, lib_dir, lmax_cmb, cls_unl:dict,
                 dlmax=1024, lmin_dlm = 2, nside_lens=4096, facres=0, nbands=8, cache_plm=True, lib_pha = None):

        lmax_plm = lmax_cmb + dlmax
        mmax_plm = lmax_plm

        cmb_cls = {}
        for k in cls_unl.keys():
            if 'p' not in k :
                cmb_cls[k] = np.copy(cls_unl[k][:lmax_cmb + dlmax + 1])
        self.lmax_plm = lmax_plm
        self.mmax_plm = mmax_plm
        self.cache_plm = cache_plm
        super(sims_postborn, self).__init__(lib_dir,  lmax_cmb, cmb_cls,
                                            dlmax=dlmax, nside_lens=nside_lens, facres=facres, nbands=nbands, lmin_dlm = lmin_dlm, lib_pha = lib_pha)

    def get_sim_kappa(self, idx: int):
        pass

    def get_sim_plm(self, idx: int):
        """Returns the lensing potential alm of the idx-th simulation coming from the callable function to get files.
        """
        fn = os.path.join(self.lib_dir, f'plm_in_{idx}_lmax{self.lmax_plm}.fits')
        if not os.path.exists(fn):
            p2k = 0.5 * np.arange(self.lmax_plm + 1) * np.arange(1, self.lmax_plm + 2, dtype=float)
            #plm = utils_hp.almxfl(hp.map2alm(hp.read_map(self.path, dtype=float), lmax=self.lmax_plm), utils.cli(p2k), self.mmax_plm, False)
            plm = utils_hp.almxfl(hp.map2alm(self.get_sim_kappa(idx), lmax = self.lmax_plm), utils.cli(p2k), self.mmax_plm, False)
            if self.cache_plm:
                hp.write_alm(fn, plm)
            return plm
        return hp.read_alm(fn)
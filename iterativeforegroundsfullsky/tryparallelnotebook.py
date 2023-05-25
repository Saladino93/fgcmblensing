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

from lenscarf import utils_scarf, utils_sims
from lenscarf.iterators import cs_iterator as scarf_iterator, steps
from lenscarf.utils import cli
from lenscarf.utils_hp import gauss_beam, almxfl, alm_copy
from lenscarf.opfilt.opfilt_iso_tt_wl import alm_filter_nlev_wl as alm_filter_tt_wl

from lenspyx.remapping.deflection_029 import deflection
from lenspyx.remapping import utils_geom

import itfgs
from itfgs.sims.sims_postborn import sims_postborn
import itfgs.sims.sims_cmbs as simsit

from plancklens.helpers import mpi


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

    args = parser.parse_args()
    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one

    from plancklens.helpers import mpi
    
    #from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from lenscarf.iterators.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        print(mpi.rank, idx)
        a = np.zeros(4)
        np.savetxt(f"{mpi.rank}.txt", a)

    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.k, idx, args.v, 1., epsilon=10 ** (- args.epsilon))
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))
                print("doing iter " + str(i))

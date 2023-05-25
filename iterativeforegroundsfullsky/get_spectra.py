#from plancklens.helpers import mpi
from plancklens.qcinv.util_alm import alm_copy

from lenscarf.iterators import statics

import healpy as hp


import itfgs.params.SOGaussianOnly as SO

import pathlib

import numpy as np


### Some settings

outputdir = pathlib.Path('results')

nmin, nmax = 0, 1 #64

lmax = SO.lmax_qlm #for analysis

iters = [0, 1] #, 2, 3, 4, 5]

v = 'tol7'

TEMP = lambda idx: f'/pscratch/sd/o/omard/n32/lenscarfrecs/{SO.suffix}/ptt_sim{idx:04}{v}'

### Calculate spectra for each sim

jobs = np.arange(nmin, nmax)

#for idx in jobs[mpi.rank::mpi.size]:
for idx in jobs:
    print(f'Index number {idx}')

    plms = statics.rec.load_plms(TEMP(idx), iters)
    #SO.lensed_cmbs.get_sim_plm(idx)
    #print('Finished')
    plm_in = alm_copy(SO.sims_MAP.sims.sims_cmb_len.get_sim_plm(idx), lmax = lmax) #GF input postborn + NL map

    auto_in = hp.alm2cl(plm_in)

    crosses = [hp.alm2cl(plm_in, plm) for plm in plms]
    autos = [hp.alm2cl(plm) for plm in plms]

    data = {}
    data['ii'] = [auto_in]
    data['ir'] = crosses
    data['rr'] = autos

    np.save(outputdir/f'{idx}{v}.npy', data)







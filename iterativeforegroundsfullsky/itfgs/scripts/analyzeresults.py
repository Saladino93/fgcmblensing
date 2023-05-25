import numpy as np

import healpy as hp

import pathlib

import os

from os.path import join as opj

import sys
sys.path.append('../itfgs/')

from itfgs.params import S4n32 as SOB

from plancklens.helpers import mpi

import argparse



print(f"Rank is {mpi.rank}")



outputdir = pathlib.Path(os.environ['SCRATCH'])/"n32spectra"

results = {}

parser = argparse.ArgumentParser()
parser.add_argument("-v", dest = "v", type=str, help="version of the iterated maps file")
parser.add_argument("-s", dest = "s", type=str, help="study case")
parser.add_argument("-k", dest = "k", type=str, help="key of the case", default = "ptt")
parser.add_argument("-itmax", dest = "itmax", type=int, help="number of iterations", default = 2)

parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')

kind = "giulio"
kind = "websky"

args = parser.parse_args()

version = args.v
studycase = args.s
qe_key = args.k
itmax = args.itmax+1

imin = args.imin
imax = args.imax


from healpy import Alm

def palm_copy(alm, lmax=None):
    """Copies the alm array, with the option to reduce its lmax.

    """
    if hasattr(alm, 'alm_copy'):
        return alm.alm_copy(lmax=lmax)

    lmox = Alm.getlmax(len(alm))
    assert (lmax <= lmox)

    if (lmox == lmax) or (lmax is None):
        ret = np.copy(alm)
    else:
        ret = np.zeros(Alm.getsize(lmax), dtype=np.complex128)
        for m in range(0, lmax + 1):
            ret[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] = \
            alm[((m * (2 * lmox + 1 - m) // 2) + m):(m * (2 * lmox + 1 - m) // 2 + lmax + 1)]
    return ret


from plancklens import utils
from os.path import join as opj

cls_path = opj(os.environ['HOME'], 'fgcmblensing', 'input', kind)
cls_unl = utils.camb_clfile(opj(cls_path, f'lensedCMB_dmn1_lenspotentialCls_{kind}.dat'))
cls_len = utils.camb_clfile(opj(cls_path, f'lensedCMB_dmn1_lensedCls_{kind}.dat'))
cls_grad = SOB.camb_clfile_gradient(opj(cls_path, f'new_lensedCMB_dmn1_lensedgradCls_{kind}.dat'))

ll = [cls_unl, cls_len, cls_grad]
for l in ll:
    for k, v in l.items():
        l[k] = np.nan_to_num(v)

cases = SOB.cases
get_info = SOB.get_info
get_all = SOB.get_all


Simulationsdir = pathlib.Path(os.environ['SCRATCH'])/'SKYSIMS/GIULIOSIMS/'


keyB = 'NL Born'
keyBg = 'NL Born Gauss'
keyBr = 'NL Born Rand'
keyBL = 'NL Born Lognormal'
keyBLr = 'NL Born Lognormal Rand'


keyBLs = 'NL Born Lognormal Double Skew'
keyBLsr = 'NL Born Lognormal Rand Double Skew'

keyPB = "NL Post-Born"
keyPBr = "NL Post-Born Rand"
keyPBg = "NL Post-Born Gauss"

keyW = "NL Websky Born"
keyWr = "NL Websky Born Rand"
keyWg = "NL Websky Born Gauss"

if studycase == "lognormal":
    cases = [SOB.casolog, SOB.casorandlog, SOB.casogauss]
    keys = [keyBL, keyBLr, keyBg]#, keyBLr, keyBL]
elif studycase == "lognormaldoubleskew":
    cases = [SOB.casologdoubleskew, SOB.casorandlogdoubleskew, SOB.casogauss]
    keys = [keyBL, keyBLr, keyBg]#, keyBLr, keyBL]
elif studycase == "born":
    cases = [SOB.casostd, SOB.casorand, SOB.casogauss]
    keys = [keyB, keyBr, keyBg]
elif studycase == "postborn":
    cases = [SOB.casopostborn, SOB.casopostbornrand, SOB.casogauss]
    keys = [keyPB, keyPBr, keyBg]
elif studycase == "websky":
    cases = [SOB.casowebskyborn, SOB.casowebskybornrand, SOB.casowebskyborngauss]
    keys = [keyW, keyWr, keyWg]

SOdict = {k: c for k, c in zip(cases, keys)}

print("Dict of cases and keys: ", SOdict)

def get_sim_len_lib(case):
    _, _, _, _, analysis_info, sims_cmb_len = get_all(case)
    return sims_cmb_len

def get_analysis_info(case):
    _, _, _, _, analysis_info, _ = get_all(case)
    return analysis_info

def gettemplensing(case):
    _, _, suffixLensing, _ = get_info(case)
    return opj(os.environ['SCRATCH'], 'n32', suffixLensing, 'lenscarfrecs')



class Config(object):
    def __init__(self, cls_unl, cls_len, cls_weight, nlev_t = 7., beam = 1.7, lmax_qlm = 4500, lminrec = 40, lmaxrec = 4000):

        self.nlev_t = nlev_t
        self.nlev_p = np.sqrt(2)*self.nlev_t
        self.beam = beam

        self.lmin_tlm = lminrec
        self.lmax_ivf = lmaxrec

        self.lmax_qlm = lmax_qlm

        self.cls_unl = cls_unl
        self.cls_len = cls_len
        self.cls_weight = cls_weight


tt = cls_len['tt']
ee = cls_len['ee']
pp = cls_unl['pp']

analysis_info = get_analysis_info(cases[0])

nlev_t = analysis_info["nlev_t"]
nlev_p = nlev_t*np.sqrt(2)
beam = analysis_info["beam"]
cls_unl_fid = cls_unl
lmin_tlm, lmax_ivf = analysis_info["lmin_tlm"], analysis_info["lmax_ivf"]
lmax_qlm = analysis_info["lmax_qlm"]

SO = Config(cls_unl, cls_len, cls_grad, nlev_t = nlev_t, beam = beam, lmax_qlm = lmax_qlm, lminrec = lmin_tlm, lmaxrec = lmax_ivf)

sim_len_libs = {c: get_sim_len_lib(c) for c in SOdict.keys()}


size = mpi.size
rank = mpi.rank

Ntot = imax-imin+1
delta = int(Ntot/size) if Ntot>size else 1

iMin = rank*delta+imin
iMax = (rank+1)*delta+imin

simset = list(range(iMin, iMax))

imin = min(simset)
imax = max(simset)


input_plm_maps = {k: [palm_copy(sims_cmb_len.get_sim_plm(i), lmax = lmax_qlm) for i in simset] for k, sims_cmb_len in sim_len_libs.items()}

temps = {c: gettemplensing(c) for c in cases}

from lenscarf.iterators import statics

plms_QE_dict = {c: [np.load(f'{temps[c]}/{qe_key}_sim{i:04}{version}/normalized_phi_plm_it000.npy') for i in simset] for c in SOdict.keys()}

p2k = np.arange(4001) * np.arange(1, 4002) * 0.5

auto_in = {k: [hp.alm2cl(p) for p in plm_in] for k, plm_in in input_plm_maps.items()}
crosses_dict_qe =  {k: [hp.alm2cl(r, p) for r, p in zip(plms_QE_dict[k], plm_in)] for k, plm_in in input_plm_maps.items()}
auto =  {k: [hp.alm2cl(p, p) for p in plms] for k, plms in plms_QE_dict.items()}


del plms_QE_dict

mean = lambda x: np.mean(x, axis = 0)


results["auto_in"] = auto_in
results["crosses_qe"] = crosses_dict_qe
results["auto_qe"] = auto

from lenscarf.iterators import statics

iters = [i for i in range(itmax)]


plms_dict = {c: [statics.rec.load_plms(f'{temps[c]}/{qe_key}_sim{i:04}{version}/', iters) for i in simset] for c in SOdict.keys()}

rho_iters_dict = {}
crosses_dict = {}
autos_dict = {}
autos_in_dict = {}

crosses_dict_lognormal = {}
autos_dict_lognormal = {}
autos_in_dict_lognormal = {}
rho_iters_dict_lognormal = {}

combined_dict = {}

for k, plms in plms_dict.items(): 
    #plms list over simulation indices
    auto_in_temp = auto_in[k] #one for each simulation index
    combined_ = np.array([[[hp.alm2cl(p_, pin), hp.alm2cl(p_)] for p_ in plm_] for plm_, pin in zip(plms, input_plm_maps[k])])
    #combined_dict[k] = combined_
    #cs_ = np.array([[hp.alm2cl(p_, pin) for p_ in plm_] for plm_, pin in zip(plms, input_plm_maps[k])])
    #as_ = np.array([[hp.alm2cl(p_) for p_ in plm_] for plm_ in plms])

    cs_ = combined_[:, :, 0, :]
    as_ = combined_[:, :, 1, :]
    
    crosses_dict[k] = cs_
    autos_dict[k] = as_
    autos_in_dict[k] = auto_in_temp

    
del input_plm_maps

results["crosses_dict"] = crosses_dict
results["autos_dict"] = autos_dict
results["autos_in_dict"] = autos_in_dict


input_plm_maps_born_gaussian = {k: sim_len_libs[SOB.casowebskyborngauss].get_sim_plm(k) for k in simset}

plm_in_ins_born_gaussian = input_plm_maps_born_gaussian #{k: palm_copy(d, lmax = lmax_qlm) for k, d in input_plm_maps_born_gaussian.items()} #GF input postborn + NL map

auto_in_born_gaussian = np.array([hp.alm2cl(palm_copy(plm_in, lmax = lmax_qlm)) for plm_in in plm_in_ins_born_gaussian.values()])

crosses_born_gaussian = np.array([[hp.alm2cl(palm_copy(plm_in, lmax = lmax_qlm), plm_rec) for plm_rec in statics.rec.load_plms(f'{gettemplensing(SOB.casowebskyborngauss)}/{qe_key}_sim{k:04}{version}', iters)] for k, plm_in in plm_in_ins_born_gaussian.items()])

results["auto_in_born_gaussian"] = auto_in_born_gaussian
results["crosses_born_gaussian"] = crosses_born_gaussian

np.save(outputdir/f"results_{version}_{studycase}_{imin}_{imax}", results)

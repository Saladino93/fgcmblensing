import os
import pathlib

import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process the results of the simulations.")
parser.add_argument("-iMin", dest = "iMin", type=int, help="minimal sim index", default=0)
parser.add_argument("-iMax", dest = "iMax", type=int, help="maximal sim index", default=63)
parser.add_argument("-totpertime", dest = "totpertime", type=int, help="number of simulations per time", default=2)
parser.add_argument("-est", dest = "est", type=str, help="estimator", default="ptt")
parser.add_argument("-maxiter", dest = "maxiter", type=int, help="maximal number of iterations", default=2)
parser.add_argument("-studycase", dest = "studycase", type=str, help="study case", default="born")
parser.add_argument("-version", dest = "version", type=str, help="version of the iterated maps file", default="")
parser.add_argument("-nonorm", dest = "nonorm", help="no normalisation", action = "store_true")

args = parser.parse_args()
iMin, iMax, totpertime = args.iMin, args.iMax, args.totpertime
estimator = args.est
maxiter = args.maxiter
studycase = args.studycase
version = args.version
nonorm = args.nonorm


strnorm = "" if not nonorm else "nonorm_"
print("Normalisation string = ", strnorm)

iters = np.arange(0, maxiter+1)

delta = totpertime #int(totpertime/ranksnumbers)


imin = iMin
imax = delta+imin-1

#version= ""
#estimator, version, studycase = "ptt", "logprior", "lognormaldoubleskew"


def cli(cl):
    """Pseudo-inverse for positive cl-arrays.
    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret


keyB = 'NL Born'
keyBg = 'NL Born Gauss'
keyBr = 'NL Born Rand'
keyBL = 'NL Born Lognormal'
keyBLr = 'NL Born Lognormal Rand'

keyBFlipped = "NL Born Flipped"

keyBflipped = "NL Born Flipped"

keyBLs = 'NL Born Lognormal Double Skew'
keyBLsr = 'NL Born Lognormal Rand Double Skew'

keyPLs = 'NL Post-Born Lognormal Double Skew'
keyPLsr = 'NL Post-Born Lognormal  RandDouble Skew'

keyPL = 'NL Post-Born Lognormal'
keyPLr = 'NL Post-Born Lognormal Rand'

keyPB = "NL Post-Born"
keyPBr = "NL Post-Born Rand"
keyPBg = "NL Post-Born Gauss"

keyW = "NL Websky Born"
keyWr = "NL Websky Born Rand"
keyWg = "NL Websky Born Gauss"



mean = lambda x: np.mean(x, axis=0)
std = lambda x: np.std(x, axis=0)

lmax = ""
#lmax = "3000_noiseless" 

outputdir = pathlib.Path(os.environ['SCRATCH'])/f"n32spectra{lmax}"


stringa = f"{estimator}_"#"" if estimator == "ptt" else f"{estimator}_"

while imax <= iMax:
    print("Read", outputdir/f"results_{stringa}{version}_{studycase}_{imin}_{imax}.npy")
    try:
        results = np.load(outputdir/f"results_{stringa}{version}_{studycase}_{imin}_{imax}.npy", allow_pickle = True).take(0)
        if imin == 0:
            auto_in = results["auto_in"]
            crosses_qe = results["crosses_qe"]
            auto_qe = results["auto_qe"]
            
            crosses_dict = results["crosses_dict"]
            autos_dict = results["autos_dict"]
            
            autos_in_dict = results["autos_in_dict"]
            
            auto_in_born_gaussian = results["auto_in_born_gaussian"]
            crosses_born_gaussian = results["crosses_born_gaussian"]
        else:
            auto_in = {k: np.vstack((np.array(el), np.array(results["auto_in"][k]))) for k, el in auto_in.items()}
            crosses_qe = {k: np.vstack((np.array(el), np.array(results["crosses_qe"][k]))) for k, el in crosses_qe.items()}
            auto_qe = {k: np.vstack((np.array(el), np.array(results["auto_qe"][k]))) for k, el in auto_qe.items()}
            
            crosses_dict = {k: np.vstack((np.array(el), np.array(results["crosses_dict"][k]))) for k, el in crosses_dict.items()}
            autos_dict = {k: np.vstack((np.array(el), np.array(results["autos_dict"][k]))) for k, el in autos_dict.items()}
            autos_in_dict = {k: np.vstack((np.array(el), np.array(results["autos_in_dict"][k]))) for k, el in autos_in_dict.items()}
            
            auto_in_born_gaussian = np.vstack((auto_in_born_gaussian, results["auto_in_born_gaussian"]))
            crosses_born_gaussian = np.vstack((crosses_born_gaussian, results["crosses_born_gaussian"]))
            
        
    except Exception as e:
        print(e)

    imin = imax+1
    imax += delta
        


allkeys = list(crosses_dict.keys())
cases = allkeys
if studycase == "postlog":
    keys = [keyPL, keyPLr, keyPBg]#, keyBLr, keyBL]
elif studycase == "lognormal":
    keys = [keyPL, keyPLr, keyPBg]
elif studycase == "bornflipped":
    keys = [keyBflipped, keyPLr, keyPBg]#, keyBLr, keyBL]
elif studycase == "lognormaldoubleskew":
    keys = [keyPLs, keyPLsr, keyPBg]#, keyBLr, keyBL]
elif studycase == "born":
    keys = [keyB, keyBr, keyBg]
elif studycase == "born_pin":
    keys = [keyB, keyBr, keyBg]
elif studycase == "postborn":
    keys = [keyPB, keyPBr, keyBg]
elif studycase == "websky":
    keys = [keyW, keyWr, keyWg]
elif studycase == "rot":
    keys = [keyB, keyBr, keyBg]
elif studycase == "bornflipped":#assumes the flipped Gaussian/Randomized do not give any difference compared to the standard one
    keys = [keyB, keyBFlipped, keyBg]

cases = allkeys

SOdict = {k: v for k, v in zip(cases, keys)}


norms_gauss_born_gaussian = {}

empirical_noises = {}

# always assume in this notebook that you have all iterations from 0 to itmax in iters
norms_gauss_born_gaussian = np.nanmean(
    np.swapaxes(crosses_born_gaussian, 0, 1) / auto_in_born_gaussian, axis=1
)


import scipy
#from scipy import signal
from scipy import stats


def bin_theory(l, lcl, bin_edges):
    sums = stats.binned_statistic(l, l, statistic="sum", bins=bin_edges)
    cl = stats.binned_statistic(l, lcl, statistic="sum", bins=bin_edges)
    cl = cl[0] / sums[0]
    return cl

Nsims = iMax-iMin


print("Nsims", Nsims)

print(cases)

title = cases[0]

# plt.figure(figsize = (3, 6))

AA, BB = 0, 1
autoin_ = mean(autos_in_dict[cases[AA]])

#elbin = ls_
#process = lambda x: signal.savgol_filter(x, 53, 3)  # window size used for filtering
# process = lambda x: x
bin_edges = np.arange(30, 3000, 141)

ls_ = np.arange(0, len(autoin_))
process = lambda x: bin_theory(ls_, ls_ * x, bin_edges)
elbin = (bin_edges[:-1] + bin_edges[1:]) / 2

#autoin_ = mean(autos_in_dict[cases[BB]])

values_A = crosses_dict[cases[AA]]/autoin_[None, None, :]/(norms_gauss_born_gaussian[None, :] if not nonorm else 1)
values_B = crosses_dict[cases[BB]]/autoin_[None, None, :]/(norms_gauss_born_gaussian[None, :] if not nonorm else 1)

np.save(f"cross_{estimator}_{studycase}", values_A)
np.save(f"cross_{estimator}_{studycase}_rand", values_B)

values = values_A - values_B

values_binned_A = np.array([np.array([process(c) for c in cross]) for cross in values_A])
values_binned_B = np.array([np.array([process(c) for c in cross]) for cross in values_B])


cross_A = np.mean(values_binned_A, axis = 0) #mean(crosses_dict[cases[AA]])
cross_B = np.mean(values_binned_B, axis = 0) #mean(crosses_dict[cases[BB]])

std_A_B = np.std(values_binned_A-values_binned_B, axis = 0)

directory = f"/users/odarwish/n32plots/data{lmax}/{estimator}/"
directory = pathlib.Path(directory)
directory.mkdir(parents=True, exist_ok=True)


colors = []
for it, cross_elements in enumerate(zip(cross_A, cross_B, std_A_B)):
    A, B, sA_B = cross_elements

    normA = norms_gauss_born_gaussian[it]
    #normA = 1
    value = A-B
    svalue = sA_B

    outdir = directory/studycase
    outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(f"{outdir}/binned_{estimator}_n32_cross_{version}{studycase}_{strnorm}{it}.txt", np.c_[elbin, value, svalue/np.sqrt(Nsims)])


autoin_ = mean(autos_in_dict[cases[AA]])

ls_ = np.arange(0, len(autoin_))
process = lambda x: bin_theory(ls_, ls_ * x, bin_edges)
elbin = (bin_edges[:-1] + bin_edges[1:]) / 2

"""cross_A = mean(autos_dict[cases[AA]])
cross_B = mean(autos_dict[cases[BB]])

std_A_B = std(autos_dict[cases[AA]] - autos_dict[cases[BB]])"""

AA, BB = 0, 1

values_A = autos_dict[cases[AA]]/autoin_[None, None, :]/norms_gauss_born_gaussian[None, :]**2
values_B = autos_dict[cases[BB]]/autoin_[None, None, :]/norms_gauss_born_gaussian[None, :]**2

np.save(f"auto_{estimator}_{studycase}", values_A)
np.save(f"auto_{estimator}_{studycase}_rand", values_B)

values = values_A - values_B

values_binned_A = np.array([np.array([process(c) for c in cross]) for cross in values_A])
values_binned_B = np.array([np.array([process(c) for c in cross]) for cross in values_B])


cross_A = np.mean(values_binned_A, axis = 0) #mean(crosses_dict[cases[AA]])
cross_B = np.mean(values_binned_B, axis = 0) #mean(crosses_dict[cases[BB]])

std_A_B = np.std(values_binned_A-values_binned_B, axis = 0)
#print(std_A_B)


for it, cross_elements in enumerate(zip(cross_A, cross_B, std_A_B)):
    A, B, sA_B = cross_elements

    value = A-B #process((A - B) / norms_gauss_born_gaussian[it] ** 2 / autoin_)
    svalue = sA_B #process((sA_B) / norms_gauss_born_gaussian[it] ** 2 / autoin_)

    np.savetxt(f"{outdir}/binned_{estimator}_n32_auto_{version}{studycase}_{strnorm}{it}.txt", np.c_[elbin, value, svalue/np.sqrt(Nsims)])


print("Done!")



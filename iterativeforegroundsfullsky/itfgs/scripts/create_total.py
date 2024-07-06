import os
import pathlib

import numpy as np

from scipy import stats


def bin_theory(l, lcl, bin_edges):
    sums = stats.binned_statistic(l, l, statistic="sum", bins=bin_edges)
    cl = stats.binned_statistic(l, lcl, statistic="sum", bins=bin_edges)
    cl = cl[0] / sums[0]
    return cl

bin_edges = np.arange(10, 4000, 150)


outdirectory = pathlib.Path("/users/odarwish/n32plots/data")


estimators = ["p"]
cases = ["born", "postborn"]
specs = ["auto", "cross"]

out_name = "diff"

for estimator in estimators:
    directory = outdirectory / estimator
    outdir = directory / out_name
    outdir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        A, B = [np.load(f"{spec}_{estimator}_{cases[i]}.npy") for i in [1, 0]]
        Arand, Brand = [np.load(f"{spec}_{estimator}_{cases[i]}_rand.npy") for i in [1, 0]]

        Nsims = A.shape[0]
        print("Nsims", Nsims)
        
        values_A = A - B
        values_B = Arand - Brand

        ls_ = np.arange(0, values_A.shape[-1])
        process = lambda x: bin_theory(ls_, ls_ * x, bin_edges)
        elbin = (bin_edges[:-1] + bin_edges[1:]) / 2

        values_binned_A = np.array([np.array([process(c) for c in cross]) for cross in values_A])
        values_binned_B = np.array([np.array([process(c) for c in cross]) for cross in values_B])

        cross_A = np.mean(values_binned_A, axis = 0) #mean(crosses_dict[cases[AA]])
        cross_B = np.mean(values_binned_B, axis = 0) #mean(crosses_dict[cases[BB]])

        std_A_B = np.std(values_binned_A-values_binned_B, axis = 0)

        for it, cross_elements in enumerate(zip(cross_A, cross_B, std_A_B)):
            A, B, sA_B = cross_elements

            value = A-B
            svalue = sA_B

            np.savetxt(f"{outdir}/binned_{estimator}_n32_{spec}_diff_{it}.txt", np.c_[elbin, value, svalue/np.sqrt(Nsims)])








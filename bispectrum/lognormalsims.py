import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import lognormalutils as lu


Nsims = 1

nside = 4096

lmax_gen = 8000

input_data = np.loadtxt("kappa_first.txt")
input_spectrum = input_data

input_map = "/Users/omard/Downloads/SCRATCHFOLDER/giuliosims/map0_kappa_ecp262_dmn2_lmax8000_first.fits"
mappa = hp.read_map(input_map)
smoothed_input = mappa

skewness = lu.get_skew_from_map(smoothed_input)
variance = lu.get_variance_from_map(smoothed_input)
mean = lu.get_mean_from_map(smoothed_input)
lamb = lu.get_lambda_from_skew(skewness, variance, mean)

alpha = lu.get_alpha(mean, lamb)
sigmaG = lu.get_sigma_gauss(alpha, variance)
muG = lu.get_mu_gauss(alpha, variance)

thirdmomentinput = np.mean((mappa-np.mean(mappa))**3)

for seed in range(Nsims):
    rng = np.random.default_rng(seed)
    outmap = lu.create_lognormal_single_map(inputcl = input_spectrum, nside = nside, lmax_gen = lmax_gen, mu = mean, lamb = lamb)
    outmap_alm = hp.map2alm(outmap)
    outmap_randomized_alm = lu.randomizing_fg(outmap_alm)
    outmap_randomized = hp.alm2map(outmap_randomized_alm, nside = nside, pol = False)

    


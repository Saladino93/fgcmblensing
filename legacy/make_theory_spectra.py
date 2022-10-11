import camb

import read_config

import argparse

import numpy as np

import pathlib

import foregrounds


parser = argparse.ArgumentParser(description = 'Calculates lensed and unlensed CMB theory spectra')
parser.add_argument('-c','--configuration', help = 'Configuration file', required = True)
configuration = parser.parse_args().configuration

output_dir = pathlib.Path(configuration)
output_dir = pathlib.Path(output_dir.stem)
output_dir.mkdir(parents = True, exist_ok = True)

config = read_config.ConfigurationCosmoReader(configuration)

pars = camb.CAMBparams()
pars.set_cosmology(H0 = config.H0, ombh2 = config.ombh2, omch2 = config.omch2)
pars.InitPower.set_params(ns = config.ns, As = config.As)
pars.set_for_lmax(config.lmax, lens_potential_accuracy = config.lens_potential_accuracy)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit = 'muK')

totCL = powers['total']
unlensedCL = powers['unlensed_scalar']

#TT, EE, BB, TE (with BB = 0 for unlensed scalar only primordial fluctuations)

lTT = totCL[:, 0]
uTT = unlensedCL[:, 0]

ls = np.arange(0, totCL.shape[0], 1)
factor = np.nan_to_num(2*np.pi/(ls*(ls+1)))

lTT *= factor
uTT *= factor

np.savetxt(output_dir/config.cmb_name, np.c_[ls, uTT, lTT])

#load and save noise
noise = np.loadtxt(config.noise_name)
np.savetxt(output_dir/config.noise_name, noise)


Foreground = foregrounds.Foregrounds(nu1 = config.nu0, nu2 = config.nu0)
fgfuncs = [Foreground.tsz, Foreground.ksz, Foreground.cib, Foreground.radps, Foreground.tsz_cib, Foreground.galacticDust]
fgs = sum([f(ells = ls) for f in fgfuncs])
np.savetxt(output_dir/config.fgname, np.c_[ls, fgs])



from camb.sources import GaussianSourceWindow, SplinedSourceWindow

import camb

from camb import model as cmodel

import argparse

import pathlib

import read_config

import numpy as np


parser = argparse.ArgumentParser(description = 'Calculates lensed and unlensed CMB theory spectra')
parser.add_argument('-c','--configuration', help = 'Configuration file', required = True)
configuration = parser.parse_args().configuration

output_dir = pathlib.Path(configuration)
output_dir = pathlib.Path(output_dir.stem)
output_dir.mkdir(parents = True, exist_ok = True)

config = read_config.ConfigurationCosmoReader(configuration)

direc = '/global/homes/o/omard/actxdes/pipeline/measure/explore/output/'
z, nz = np.loadtxt(direc+'galaxy_z_nz_0.2-0.4.txt', unpack = True)

pars = camb.CAMBparams()
pars.set_cosmology(H0 = config.H0, ombh2 = config.ombh2, omch2 = config.omch2)
pars.InitPower.set_params(ns = config.ns, As = config.As)
pars.set_for_lmax(config.lmax, lens_potential_accuracy = config.lens_potential_accuracy)
pars.NonLinear = cmodel.NonLinear_both


pars.SourceWindows = [SplinedSourceWindow(bias = 1., dlog10Ndm = -0.2, z = z, W = nz)]

results = camb.get_results(pars)
cls = results.get_source_cls_dict()
ls =  np.arange(2, config.lmax+1)
factor = 2*np.pi/(ls*(ls+1))
factor_p_to_k = ls*(ls+1)/2

key = 'PxW1'
clpg = cls[key]
clkg = clpg*factor_p_to_k

np.savetxt(ls, clkg)

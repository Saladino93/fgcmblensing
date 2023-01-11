'''
Gets the sum of several maps in a list
'''

import numpy as np
import healpy as hp

import pathlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest = 'c', type = str, help = 'common directory')
parser.add_argument('-i', dest = 'i', type = list, help = 'list of input fits')
parser.add_argument('-o', dest = 'o', type = str, help = 'output fits')
parser.add_argument('-freq', dest = 'freq', type = int, help = 'frequency')

args = parser.parse_args()
inputs = args.i
output = args.o
common = args.c
common = pathlib.Path(common)

somma = sum(list(map(lambda x: hp.read_map(f'{str(common/x)}.fits'), inputs)))
hp.write_map(f'{str(common/output)}.fits', somma)


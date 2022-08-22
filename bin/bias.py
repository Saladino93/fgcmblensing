from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs
import numpy as np
import os,sys
from falafel import qe,utils
import solenspipe
#from solenspipe import bias,biastheory
import pytempura
import healpy as hp
from enlib import bench
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

import pathlib

import constants as const

"""
Here we calculate RDN0, MCN0 and MCN1 for any estimator combination, e.g.
MVMV
Mvpol Mvpol
TTTT
TTTE
TTEE
TTEB
etc.
and compare against theory N0 on the full noiseless sky
for both gradient and curl
with and without bias hardening of TT.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Verify and benchmark RDN0 on the full noiseless sky.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("est1", type=str,help='Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("est2", type=str,help='Estimator 2, same as above.')
parser.add_argument("--nsims-n0",     type=int,  default=1,help="Number of sims.")
parser.add_argument("--nsims-n1",     type=int,  default=1,help="Number of sims.")
parser.add_argument( "--healpix", action='store_true',help='Use healpix instead of CAR.')
parser.add_argument( "--new-scheme", action='store_true',help='New simulation scheme.')
parser.add_argument( "--lmax",     type=int,  default=3000,help="Maximum multipole for lensing.")
parser.add_argument( "--lmin",     type=int,  default=100,help="Minimum multipole for lensing.")
parser.add_argument( "--biases",     type=str,  default="",help="Maximum multipole for lensing.")
args = parser.parse_args()

biases = args.biases.split(',')
opath = f"{solenspipe.opath}/{args.version}_"

# Multipole limits and resolution
lmin = args.lmin; lmax = args.lmax
#mlmax = int(4000 * (args.lmax / 3000)) # for alms
mlmax = 7000

grad = True # Use gradient-field spectra in norm

nsims_n0 = args.nsims_n0 # number of sims to test RDN0
nsims_n1 = args.nsims_n1 # number of sims to test MCN1
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()

#nside = 4096
#px = qe.pixelization(nside = nside) 

# Get CMB Cls for response and total Cls corresponding to a noiseless configuration

allelementstosave = np.load('input_cmb_145.npy')
ells, lcmb, tsz, ksz, radio, cib, dust, nl145, totalcmb, totalnoisecmb = allelementstosave.T


ell = np.arange(mlmax+1)
fgs = tsz+ksz+radio+cib+dust
noise = np.interp(ell, ells, nl145+fgs)
Nl_tt = np.nan_to_num(noise)
nells = {"TT": Nl_tt, "EE": Nl_tt*0, "BB": Nl_tt*0}

ucls,tcls = utils.get_theory_dicts(grad = grad, nells = nells, lmax = mlmax)

e1 = args.est1.upper()
e2 = args.est2.upper()

'''
# Get norms
bh,ells,Als,R_src_tt,Nl_g,Nl_c,Nl_g_bh = solenspipe.get_tempura_norms(args.est1,args.est2,ucls,tcls,args.lmin,args.lmax,mlmax)

e1 = args.est1.upper()
e2 = args.est2.upper()
diag = (e1==e2)
if rank==0:
    np.savetxt(f'{opath}Nlg_{e1}_{e2}.txt',Nl_g)
    np.savetxt(f'{opath}Nlc_{e1}_{e2}.txt',Nl_c)
    if bh: np.savetxt(f'{opath}Nlg_bh_{e1}_{e2}.txt',Nl_g_bh)


# Plot expected noise performance
theory = cosmology.default_theory()

if rank==0:
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(ells[1:],Nl_g[1:],ls='--',label='grad')
    pl.add(ells[2:],Nl_c[2:],ls='-.',label='curl')
    if bh:
        pl.add(ells[1:],Nl_g_bh[1:],ls=':',label='grad BH')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{opath}bh_noise_{e1}_{e2}.png')

bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)[1]
cents = binner.cents


# These are the qfunc lambda functions we will use with RDN0 and MCN1
q_nobh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2=None,Al2=None,R12=None)
q_nobh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2=None,Al2=None,R12=None)
if bh:
    q_bh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e1 in ['TT','MV'] else q_nobh_1
    q_bh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e2 in ['TT','MV'] else q_nobh_2

'''



# Build get_kmap functions


def dummy_teb(alms):
    '''
    Creates a list of maps with the order [Tlm, Elm, Blm] and Elm, Blm = 0, 0
    '''
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def get_sehgal(nome):
    import pathlib
    source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')
    cmb_alm = hp.read_alm(source_dir/f'{nome}_alm.fits')
    if nome != 'kappa':
        cmb_alm = dummy_teb(cmb_alm)
    return cmb_alm


def get_sehgalnpy(nome):
    import pathlib
    source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')
    cmb_alm = np.load(source_dir/f'{nome}_alm.npy')
    cmb_alm = dummy_teb(cmb_alm)
    return cmb_alm

def get_kmap():
    cmboutname = 'cmb'
    dalm = get_sehgal(cmboutname)#solenspipe.get_cmb_alm(s_i,s_set)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)


def get_fgmap(name):
    dalm = get_sehgal(name)#solenspipe.get_cmb_alm(s_i,s_set)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)



def fnu(nu, tcmb = const.default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = const.H_CGS*(1e9*nu)/(const.K_CGS*tcmb)
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans

def tsz_factor_for_ymap(freq, tcmb = const.default_tcmb):
    return fnu(freq) * tcmb * 1e6

def process_tsz(comptony, freq, tcmb = const.default_tcmb):
    return tsz_factor_for_ymap(freq = freq, tcmb = tcmb) * np.array(comptony)


freq = 145

mappe = [get_kmap(), process_tsz(get_fgmap('tsz'), 145), process_tsz(get_fgmap('tsz_masked'), 145), get_fgmap('cib'), get_fgmap('cib_masked')]
codes = ['', 'tsz', 'tsz_masked', 'cib', 'cib_masked']

source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

kappa_lmin = 100
kappa_lmaxes = [3000]

for kappa_lmax in kappa_lmaxes:

    print(f'Calculating for lmax of reconstruction of {kappa_lmax}')


    # Geometry
    px = solenspipe.get_sim_pixelization(kappa_lmax, args.healpix,verbose=(rank==0))


    _, ls, Als, R_src_tt, Nl_g, Nl_c, Nl_g_bh = solenspipe.get_tempura_norms(
        est1 = 'TT', est2 = 'TT', ucls = ucls, tcls = tcls, lmin = kappa_lmin, lmax = kappa_lmax, mlmax = mlmax)



    #opath = '/global/homes/o/omard/so-lenspipe/data/5_'
    #e1, e2 = 'TT', 'TT'
    #Nlg_bh_ = np.loadtxt(f'{opath}Nlg_bh_{e1}_{e2}.txt')

    #R_src_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, kappa_lmin, kappa_lmax, k_ellmax = mlmax)

    qfunc = solenspipe.get_qfunc(px, ucls, mlmax, 'TT', Al1 = Als['TT'])
    qfunc_bh = solenspipe.get_qfunc(px, ucls, mlmax, 'TT', est2 = 'SRC', Al1 = Als['TT'], 
                                    Al2 = Als['src'], R12 = R_src_tt)

    qfunc_bh = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='SRC',Al2=Als['src'],R12=R_src_tt)

    all_spectra = {}

    for map_alm, code in zip(mappe, codes):

        print(f'Filtering {code}')
        #input_alm = utils.change_alm_lmax(map_alm.astype(np.complex128), mlmax)
        #input_alm_filtered = filter_alms(input_alm, tcls, kappa_lmin, kappa_lmax)
        input_alm_filtered = map_alm

        versions = ['qe', 'bh'] 
        functions = [qfunc, qfunc_bh]

        vstuff = {}

        for function, version in zip(functions, versions):

            stuff = {}
            print('Reconstruct with', version)
            
            phi_recon_alms = function(input_alm_filtered, input_alm_filtered)
            #phi_recon_alms = function(Xdat, Xdat)    

            print('Convert to kappa')
            kappa_recon_alms = plensing.phi_to_kappa(phi_recon_alms)

            kappa_alm = utils.change_alm_lmax(get_sehgal('kappa').astype(np.complex128), mlmax)

            print('Convert to spectra')
            cl_kk_output_output = cs.alm2cl(kappa_recon_alms[0])
            cl_kk_input_output = cs.alm2cl(kappa_recon_alms[0], kappa_alm)
            cl_kk_input = cs.alm2cl(kappa_alm)

            np.save(source_dir/f'kappa_reconstructed_{version}{code}_{kappa_lmax}', kappa_recon_alms[0])

            stuff['oo'] = cl_kk_output_output
            stuff['io'] = cl_kk_input_output
            stuff['ii'] = cl_kk_input

            vstuff[version] = stuff

        all_spectra[code] = vstuff

    np.save(f'all_spectra_{kappa_lmax}', all_spectra)




'''
for code, Xdat in zip(codes, mappe):
    # Get data
    dlmax = hp.Alm.getlmax(Xdat[0].size)
    if rank==0: print(f"Data lmax: {dlmax}")

    # Let's make a single map and cross-correlate with input
    if rank==0:
        # Get kappa alm
        #ikalm = utils.change_alm_lmax(utils.get_kappa_alm(1999).astype(np.complex128),mlmax) # TODO: fix hardcoding
        ikalm = utils.change_alm_lmax(get_sehgal('kappa').astype(np.complex128), mlmax)

        # New convention in falafel means maps are potential; we convert to convergence
        r_nobh_1 = plensing.phi_to_kappa(q_nobh_1(Xdat,Xdat))
        r_nobh_2 = r_nobh_1 #plensing.phi_to_kappa(q_nobh_2(Xdat,Xdat))

        uicls = cs.alm2cl(ikalm,ikalm)
        uxcls_nobh_1 = cs.alm2cl(r_nobh_1[0],ikalm)
        uxcls_nobh_2 = cs.alm2cl(r_nobh_2[0],ikalm)
        uacls_nobh = cs.alm2cl(r_nobh_1,r_nobh_2)
        np.save(f'{opath}uicls_{e1}_{e2}{code}.npy',uicls)
        np.save(f'{opath}uxcls_nobh_1_{e1}_{e2}{code}.npy',uxcls_nobh_1)
        np.save(f'{opath}uxcls_nobh_2_{e1}_{e2}{code}.npy',uxcls_nobh_2)
        np.save(f'{opath}uacls_nobh_{e1}_{e2}{code}.npy',uacls_nobh)

        if bh:
            phi = q_bh_1(Xdat,Xdat)

            print(len(phi))
            print(np.mean(phi[0]))

            r_bh_1 = plensing.phi_to_kappa(phi)

            r_bh_2 = r_bh_1 #plensing.phi_to_kappa(q_bh_2(Xdat,Xdat))
            uxcls_bh_1 = cs.alm2cl(r_bh_1[0],ikalm)
            uxcls_bh_2 = cs.alm2cl(r_bh_2[0],ikalm)
            uacls_bh = cs.alm2cl(r_bh_1,r_bh_2)
            np.save(f'{opath}uxcls_bh_1_{e1}_{e2}{code}.npy',uxcls_bh_1)
            np.save(f'{opath}uxcls_bh_2_{e1}_{e2}{code}.npy',uxcls_bh_2)
            np.save(f'{opath}uacls_bh_{e1}_{e2}{code}.npy',uacls_bh)

        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:], uxcls_bh_1[2:],color='k')
        pl.add(ells[2:], uxcls_bh_2[2:],color='r')
        pl.add(ells[2:], uicls[2:],color='b', label = 'Input')
        pl.done(f'{opath}bh_spectra{code}.png')'''
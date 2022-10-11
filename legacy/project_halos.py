import numpy as np

import pathlib

import healpy as hp

from pixell import enmap, utils


def get_weighted_counts_from_coords(nside: int, decs: np.ndarray, ras: np.ndarray, weights: np.ndarray) -> np.ndarray:
    '''
    Given coordinates in the form of decs, ras (in degrees), and weights, 
    returns a healpix map with some nside resolution.
    Parameters
    ----------
    Returns
    ----------
    Map counts
    '''
    
    shape = hp.nside2npix(nside)
    pixels = hp.ang2pix(nside, ras, decs, lonlat = True)

    mappa = np.zeros(shape)
    counts = np.add.at(mappa, pixels, weights)#np.histogram(pixels, bins = shape, weights = weights, range = [0, shape], density = False)[0].astype(np.float32)
    
    return mappa


source_dir = pathlib.Path('/global/cscratch1/sd/omard/scatteringtfms/sims/')

halo_file = source_dir/'halo_nbody.ascii'
halos = np.loadtxt(halo_file).T

columns = ['Z','RA','DEC','POSX','POSY','POSZ','VX','VY','VZ','MFOF','MVIR','RVIR','M200','R200','M500','R500','M1500','R1500','M2500','R2500']

def get_all(ra, dec):
    theta,phi = np.pi/2 - dec, ra
    all_theta,all_phi = [],[]
    for n in range(4):
        theta_N = theta
        phi_N = phi + n*np.pi/2
        theta_S = np.pi-theta
        phi_S = np.pi/2 - phi + n*np.pi/2
        all_theta += list(theta_N)
        all_theta += list(theta_S)
        all_phi += list(phi_N)
        all_phi += list(phi_S)
    all_theta,all_phi = np.array(all_theta),np.array(all_phi)
    all_dec,all_ra = np.pi/2 - all_theta, all_phi
    return all_ra, all_dec


direc = '/global/homes/o/omard/actxdes/pipeline/measure/explore/output/'
z, nz = np.loadtxt('sehgalconfig/lsst.txt', unpack = True)#np.loadtxt(direc+'galaxy_z_nz_0.2-0.4.txt', unpack = True)
total_number = np.trapz(nz, z)
nznorm = nz/total_number

nside = 4096
M_200_ = halos[12, :]
zs_ = halos[0, :]
weights_ = np.interp(zs_, z, nznorm)
ras_, decs_ = halos[1, :], halos[2, :]
ras_, decs_  = np.radians(ras_), np.radians(decs_)
poles = [1, -1]
shifts = list(range(4))
ras, decs, M_200, zs, weights = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
for p in poles:
    for shift in shifts:
        ras = np.append(ras, (p*ras_+np.pi/2*shift)+(p == -1)*np.pi/2)
        decs = np.append(decs, (p)*decs_)
        M_200 = np.append(M_200, M_200_)
        zs = np.append(zs, zs_)
        weights = np.append(weights, weights_)

ra_, dec_ = get_all(ras_, decs_)

#ras, decs = ra_, dec_

ras = np.array(ras)
#print('SHAPE', ras.shape, decs.shape)
decs = np.array(decs)

ras, decs = np.degrees(ras), np.degrees(decs)

extra = ''
extra = '_weighted'

#weights = np.interp(zs, z, nznorm)
#weights = np.ones_like(decs)

print('Computing counts')
counts = get_weighted_counts_from_coords(nside = nside, decs = decs, ras = ras, weights = weights)


np.save(source_dir/f'counts{extra}', counts)
hp.write_map(source_dir/f'counts{extra}.fits', counts)

#print('Taking alms of counts')
#import healpy as hp
#hp.write_alm(source_dir/'counts_alm.fits', hp.map2alm(counts))

print('Plotting')
import matplotlib.pyplot as plt
hp.mollview(counts)
plt.savefig('counts.png')

print('Generating Halo Mask')

def get_halo_mask(mass, ras, decs, m_min, mask_radius,
                    zmax=None, num_halo=None):

    use = mass > m_min
    ras, decs = ras[use], decs[use]
    num_halo = len(ras)

    print("masking %d halos"%num_halo)
    #Get halo ra/dec
    ra_deg, dec_deg = ras, decs
    dec, ra = np.radians(dec_deg) ,np.radians(ra_deg)

    r = mask_radius*utils.arcmin

    srcs = np.array([dec, ra])
    halo_mask = (enmap.distance_from_healpix(
        nside, srcs, rmax=r) >= r)
    print("halo masking %d/%d pixels (f_sky = %.2f) in fgs"%(
        (~halo_mask).sum(),len(halo_mask),
        float((~halo_mask).sum())/len(halo_mask)))
    return halo_mask

mass, ras, decs, m_min, mask_radius, zmax = M_200, ras, decs, 1e15, 5., 4.
maschera = get_halo_mask(mass, ras, decs, m_min, mask_radius)

np.save(source_dir/f'maschera{extra}', maschera)
hp.write_map(source_dir/f'maschera{extra}', maschera)

hp.mollview(maschera)
plt.savefig(f'maschera{extra}.png')

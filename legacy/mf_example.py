import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def get_match_filtered_map(
        input_map, B_l, C_l
        ):
    """        
    Get a matched filtered map,
    following McEwan et al. 2008
    (astro-ph/0612688)
    
    Parameters                                                                                                                             
    ----------                                                                                                                             
    input_map: np.array                                                                                                                    
      healpix input map                                                                                                                    
    B_l: np.array                                                                                                                           
      filter for each l
    C_l: np.array
      noise for each l
    """
    lmax = len(B_l)-1
    ells = np.arange(lmax+1)
    nside = hp.npix2nside(len(input_map))

    #Calculate filter normalization from eqn. 21
    #Not sure if I should also have a 2l+1 factor
    #in there
    norm = np.sum(B_l**2/C_l)

    #Make filter following eqn. 20
    psi_l = B_l/C_l/norm
    
    #filter map following eqn. 13                                                                                                                            
    f_lm = hp.map2alm(input_map, lmax=lmax)
    fac = np.sqrt(4*np.pi/(2*ells+1))
    u_lm = hp.almxfl( f_lm, psi_l*fac ) 
    
    #and convert back to a map
    return hp.alm2map(u_lm, nside)

def main():
    #Test it 
    nside=1024
    beam_fwhm_deg = 1.  #arcmin
    lmax=3*nside+1

    B_l = hp.sphtfunc.gauss_beam(np.radians(beam_fwhm_deg),
                                 lmax=lmax)

    #just use constant noise
    C_l = np.ones(lmax+1)

    #Make a map that has one point source
    #at a pixel index ps_ind
    m = np.zeros(hp.nside2npix(nside))
    ps_ind=int(hp.nside2npix(nside)/2)
    print("inserting point-source with flux=100")

    m[ps_ind] = 100.

    #Convolve map with beam
    m = hp.smoothing(m, fwhm=np.radians(beam_fwhm_deg))
    print("value of smoothed map at point-source position:")
    print(m[ps_ind])

    #Now get match filtered map
    u = get_match_filtered_map(m, B_l, C_l)

    hp.mollview(m)
    plt.show()
    plt.close()
    hp.mollview(u)
    plt.show()
    plt.close()

    #And check the value at the index where we placed the 
    #point source. If correct, this should equal the point source
    #amplitude
    print("value of filtered map at point-source position:")
    print(u[ps_ind])


main()
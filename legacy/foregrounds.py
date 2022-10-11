from szar import foregrounds as sforegrounds

import numpy as np

class Foregrounds(object):

    def __init__(self, nu1: float, nu2: float = None, Tcmb = 2.726):
        if nu2 is None:
            nu2 = nu1

        self.nu1 = nu1 #in GHz
        self.nu2 = nu2 #in GHz
        self.Tcmb = Tcmb #in K

        self.fdl_to_cl = lambda l: 1./( l*(l+1.)/(2.*np.pi) )

        # constants
        self.c = 3.e8  # m/s
        self.h = 6.63e-34 # SI
        self.kB = 1.38e-23   # SI
        self.Jansky = 1.e-26 # W/m^2/Hz

    def tsz(self, ells: np.ndarray, A_tsz: float = None):
        #note, szar foregrounds wants tcmb in muK units
        return sforegrounds.power_tsz(ells = ells, nu1 = self.nu1, nu2 = self.nu2, A_tsz = A_tsz, tcmb = self.Tcmb*1e6)

    def tsz_cib(self, ells: np.ndarray, A_tsz: float = None, A_cibc: float = None, Td: float = None, zeta: float = None):
        return sforegrounds.power_tsz_cib(ells = ells, nu1 = self.nu1, nu2 = self.nu2, al_cib = None, Td = None, A_tsz = None, A_cibc = None, zeta = zeta)

    def cib(self, ells: np.ndarray, A_cibp: float = None, A_cibc: float = None, al_cib: float = None, Td: float = None, n_cib: float = None):
        result =  sforegrounds.power_cibp(ells = ells, nu1 = self.nu1, nu2 = self.nu2, A_cibp = A_cibp, al_cib = al_cib, Td = Td)
        result += sforegrounds.power_cibc(ells = ells, nu1 = self.nu1, nu2 = self.nu2, A_cibc = A_cibc, n_cib = n_cib, al_cib = al_cib, Td = Td)
        return result

    def radps(self, ells: np.ndarray, A_ps: float = None, al_ps: float = None):
        return sforegrounds.power_radps(ells = ells, nu1 = self.nu1, nu2 = self.nu2, A_ps = A_ps, al_ps = al_ps)
    
    def ksz(self, ells: np.ndarray, A_rksz: float = 1, A_lksz: float = 1):
        result = sforegrounds.power_ksz_reion(ells = ells, A_rksz = A_rksz)
        result += sforegrounds.power_ksz_late(ells = ells, A_lksz = A_lksz)
        return result

    #copied from https://github.com/EmmanuelSchaan/LensQuEst/blob/master/cmb.py#L317
    def galacticDust(self, ells: np.ndarray):
        beta_g = 3.8
        n_g = -0.7
        a_ge = 0.9
        a_gs = 0.7  #95% confidence limit
        #modified the 1e9, given that the object takes freqs in GHz
        return a_gs * (ells/3000.)**2 * (self.nu1*self.nu2/150**2.)**beta_g * self.g(self.nu1, self.Tcmb)*self.g(self.nu2, self.Tcmb)/self.g(150, self.Tcmb)**2 * self.fdl_to_cl(ells)

    #copied from https://github.com/EmmanuelSchaan/LensQuEst/blob/master/cmb.py#L171
    # d(blackbody)/dT at T
    # output in SI
    def dBdT(self, nu, T):
        nu *= 1e9 #modified, given that the object takes freqs in GHz
        x = self.h*nu/(self.kB*T)
        result = 2.*self.h**2*nu**4
        result /= self.kB*T**2*self.c**2
        result *= np.exp(x) / (np.exp(x) - 1.)**2
        return result
   
    #copied from https://github.com/EmmanuelSchaan/LensQuEst/blob/master/cmb.py#L182
    # dT/d(blackbody) at T
    # output in SI
    def g(self, nu, T):
        return 1./self.dBdT(nu, T)
import numpy as np
from scipy.interpolate import *

import constants as const

import scipy



def randomizing_fg(mappa: np.ndarray):
     f = lambda z: np.abs(z) * np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
     return f(mappa)

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

def get_tsz_from_comptony(comptony, freq, tcmb = const.default_tcmb):
    return tsz_factor_for_ymap(freq = freq, tcmb = tcmb) * comptony


class Foregrounds():

    def __init__(self, nu1, nu2, input_dir: str = "../input/", input_version: str = 'Sehgal'):
        '''
        nu1, nu2 in GHz
        '''
        self.nu1 = nu1*1e9
        self.nu2 = nu2*1e9

        self.c = const.C
        self.h = const.H
        self.kB = const.KB
        self.Tcmb = const.TCMB
        self.Jansky = const.JANKSY

        self.fdl_to_cl = lambda l: 1./( l*(l+1.)/(2.*np.pi) )


        #data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
        data = np.genfromtxt(input_dir+f"{input_version}/{input_version}_lenspotentialCls.dat")
        data = np.nan_to_num(data)
        self.funlensedTT_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
        lmin_unlensedCMB = data[0,0]
        lmax_unlensedCMB = data[-1,0]
        self.funlensedTT = np.vectorize(lambda l: (l>=lmin_unlensedCMB and l<=lmax_unlensedCMB) * self.funlensedTT_template(l) * self.fdl_to_cl(l))


        # lensed CMB
        #data = np.genfromtxt("./input/universe_Planck15/camb/lensedCls.dat")
        data = np.genfromtxt(input_dir+f"{input_version}/{input_version}_lensedCls.dat")
        data = np.nan_to_num(data)
        self.flensedTT_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
        lmin_lensedCMB = data[0,0]
        lmax_lensedCMB = data[-1,0]
        self.flensedTT = np.vectorize(lambda l: (l>=lmin_lensedCMB and l<=lmax_lensedCMB) * self.flensedTT_template(l) * self.fdl_to_cl(l))


        # tSZ: Dunkley et al 2013
        data = np.genfromtxt(input_dir+"/cmb/digitizing_SZ_template/tSZ.txt")
        ftSZ_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
        a_tSZ = 4.0
        lmin_tSZ = data[0,0]
        lmax_tSZ = data[-1,0]
        self.ftSZ = np.vectorize(lambda l: (l>=lmin_tSZ and l<=lmax_tSZ) * a_tSZ * self.tszFreqDpdceTemp(self.nu1)*self.tszFreqDpdceTemp(self.nu2)/self.tszFreqDpdceTemp(150.e9)**2 * ftSZ_template(l) * self.fdl_to_cl(l))

        # kSZ: Dunkley et al 2013
        data = np.genfromtxt(input_dir+"/cmb/digitizing_SZ_template/kSZ.txt")
        fkSZ_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
        a_kSZ = 1.5  # 1.5 predicted by Battaglia et al 2010. Upper limit from Dunkley+13 is 5.
        lmin_kSZ = data[0,0]
        lmax_kSZ = data[-1,0]
        self.fkSZ = np.vectorize(lambda l: (l>=lmin_kSZ and l<=lmax_kSZ) * a_kSZ * fkSZ_template(l) * self.fdl_to_cl(l))

        # tSZ x CMB: Dunkley et al 2013
        xi = 0.2 # upper limit at 95% confidence
        a_tSZ = 4.0
        a_CIBC = 5.7
        betaC = 1.2 #2.1
        Td = 24 #9.7
        # watch for the minus sign
        data = np.genfromtxt (input_dir+"/cmb/digitizing_tSZCIB_template/minus_tSZ_CIB.txt")
        ftSZCIB_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
        lmin_tSZ_CIB = data[0,0]
        lmax_tSZ_CIB = data[-1,0]
        self.ftSZ_CIB = np.vectorize(lambda l: (l>=lmin_tSZ_CIB and l<=lmax_tSZ_CIB) * (-2.)*xi*np.sqrt(a_tSZ*a_CIBC)* self.fprime(self.nu1, self.nu2, betaC, Td)/self.fprime(150.e9, 150.e9, betaC, Td) * ftSZCIB_template(l) * self.fdl_to_cl(l))


    def Jyoversr_to_muKcmb(self, mappa, freq): 
        return mappa * 1.e-26 / self.dBdT(freq, self.Tcmb) * 1.e6

    def convertIntSITo(self, nu, kind="intSI"):
        '''kind: "intSI", "intJy/sr", "tempKcmb", "tempKrj"
        '''
        if kind=="intSI":
            result = 1.
        elif kind=="intJy/sr":
            result = 1. / self.Jy
        elif kind=="tempKcmb":
            result = 1. / self.dBdT(nu, self.Tcmb)
        elif kind=="tempKrj":
            result = 1. / self.dBdTrj(nu, self.Tcmb)
        return result

    def kszFreqDpdceTemp(self, nu):
        '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
        arbitrary normalization
        '''
        return self.blackbody(nu, self.Tcmb)*self.convertIntSITo(nu, kind="tempKcmb")


    def cibPoissonFreqDpdceTemp(self, nu):
        '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
        arbitrary normalization
        '''
        Td = 24 #9.7
        betaP = 1.2 #2.1
        result = self.mu(nu, betaP, Td) 
        result /= self.mu(145.e9, betaP, Td)
        #return nu**betaP* self.blackbody(nu, Td)*self.convertIntSITo(nu, kind="tempKcmb")
        return result        

    def radioPoissonFreqDpdceTemp(self, nu):
        '''Intensity units ([W/Hz/m^2/sr] or [Jy/sr])
        arbitrary normalization
        '''
        alpha_s = -0.5
        return nu**alpha_s*self.convertIntSITo(nu, kind="tempKcmb")


    # blackbody function
    # nu in Hz
    # output in W / Hz / m^2 / sr
    def blackbody(self, nu, T):
        x = self.h*nu/(self.kB*T)
        result = 2.*self.h*nu**3 /self.c**2
        result /= np.exp(x) - 1.
        return result


    # dlnBlackbody/dlnT
    # output in SI
    def dlnBdlnT(self, nu, T):
        x = self.h*nu/(self.kB*T)
        return x * np.exp(x) / (np.exp(x) - 1.)

    # d(blackbody)/dT at T
    # output in SI
    def dBdT(self, nu, T):
        x = self.h*nu/(self.kB*T)
        result = 2.*self.h**2*nu**4
        result /= self.kB*T**2*self.c**2
        result *= np.exp(x) / (np.exp(x) - 1.)**2
        return result

    # dT/d(blackbody) at T
    # output in SI
    def g(self, nu, T):
        return 1./self.dBdT(nu, T)

    # blackbody modified with power law
    # expressed in temperature units
    # relevant for CIB
    def mu(self, nu, beta, T):
        return nu**beta * self.blackbody(nu, T) * self.g(nu, self.Tcmb)


    # frequency dependence for tSZ
    # dT/T = freqDpdceTSZTemp * y
    def tszFreqDpdceTemp(self, nu):
        x = self.h*nu/(self.kB*self.Tcmb)
        return x*(np.exp(x)+1.)/(np.exp(x)-1.) -4.

    # frequency dependence for tSZ
    # dI/I = freqDpdceTSZIntensity * y
    def freqDpdceTSZIntensity(self, nu):
        return self.tszFreqDpdceTemp(nu) * self.dlnBdlnT(nu, self.Tcmb)


    def fprime(self, nu1, nu2, beta, T):
        return self.tszFreqDpdceTemp(nu1) * self.mu(nu2, beta, T) + self.tszFreqDpdceTemp(nu2) * self.mu(nu1, beta, T)

    def fgalacticDust(self, l):
      beta_g = 3.8
      n_g = -0.7
      a_ge = 0.9
      a_gs = 0.7  # 95% confidence limit
      return a_gs * (l/3000.)**2 * (self.nu1*self.nu2/150.e9**2)**beta_g * self.g(self.nu1, self.Tcmb)*self.g(self.nu2, self.Tcmb)/self.g(150.e9, self.Tcmb)**2 * self.fdl_to_cl(l)


    # CIB Poisson and clustered

    def fCIBPoisson(self, l, nu1=None, nu2=None):
        a_CIBP = 7.0
        Td = 24 #9.7
        betaP = 1.2 #2.1
        if nu1 is None:
            nu1 = self.nu1
        if nu2 is None:
            nu2 = self.nu2
        return a_CIBP * (l/3000.)**2 * self.mu(nu1, betaP, Td)*self.mu(nu2, betaP, Td)/self.mu(150.e9, betaP, Td)**2 * self.fdl_to_cl(l)

    def fCIBClustered(self, l, nu1=None, nu2=None):
        a_CIBC = 5.7
        n = 1.2
        Td = 24 #9.7
        betaC = 1.2 #2.1
        if nu1 is None:
            nu1 = self.nu1
        if nu2 is None:
            nu2 = self.nu2
        return a_CIBC * (l/3000.)**(2-n) * self.mu(nu1, betaC, Td)*self.mu(nu2, betaC, Td)/self.mu(150.e9, betaC, Td)**2 * self.fdl_to_cl(l)

    def fCIB(self, l, nu1=None, nu2=None):
        return self.fCIBPoisson(l, nu1, nu2) + self.fCIBClustered(l, nu1, nu2)



    # radio point sources, Poisson only

    def fradioPoisson(self, l):
        alpha_s = -0.5
        a_s = 3.2
        return a_s * (l/3000.)**2 * (self.nu1*self.nu2/150.e9**2)**alpha_s * self.g(self.nu1, self.Tcmb)*self.g(self.nu2, self.Tcmb)/self.g(150.e9, self.Tcmb)**2 * self.fdl_to_cl(l)


    # total
    def ftotal(self, l):
        result = 0.
        result += self.fCIBPoisson(l)
        result += self.fCIBClustered(l)
        result += self.ftSZ(l)
        result += self.fkSZ(l)
        result += self.ftSZ_CIB(l)
        result += self.fradioPoisson(l)
        return result


    # outputs the uncertainty on amplitude of profile
    # given the total power in the map
    # fprofile: isotropic profile (before beam convolution)
    # if none, use the beam as profile (ie point source)
    # If temperature map in muK, then output in muK*sr
    # If temperature map in Jy/sr, then output in Jy
    def fsigmaMatchedFilter(self, fprofile = None, ftotalTT = None):
        if ftotalTT is None:
            ftotalTT = self.ftotalTT
        if fprofile is None:
            f = lambda l: l/(2.*np.pi) / ftotalTT(l)
        else:
            f = lambda l: l/(2.*np.pi) * fprofile(l) / ftotalTT(l)
        result = scipy.integrate.quad(f, self.lMin, self.lMaxT, epsabs=0., epsrel=1.e-3)[0]
        result = 1./np.sqrt(result)
        return result
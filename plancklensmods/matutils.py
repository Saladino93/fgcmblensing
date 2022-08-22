from scipy.interpolate import interp1d

import re

import numpy as np



def validate_map_type(mapXYType):
    assert not(re.search('[^TEB]', mapXYType)) and (len(mapXYType)==2), \
      mapXYType+"\" is an invalid map type. XY must be a two" + \
      " letter combination of T, E and B. e.g TT or TE."

class TheorySpectra:
    '''
    Essentially just an interpolator that takes a CAMB-like
    set of discrete Cls and provides lensed and unlensed Cl functions
    for use in integrals
    '''
    

    def __init__(self):

        self.always_unlensed = False
        self.always_lensed = False
        self._uCl={}
        self._lCl={}
        self._gCl = {}


    def loadGenericCls(self,ells,Cls,keyName,lpad=9000,fill_zero=True):
        if not(fill_zero):
            fillval = Cls[ells<lpad][-1]
            print(fillval)
            print(ells[ells<lpad],Cls[ells<lpad])
            self._gCl[keyName] = lambda x: np.piecewise(x, [x<=lpad,x>lpad], [lambda y: interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=0.)(y),lambda y: fillval*(lpad/y)**4.])
            print(self._gCl[keyName](ells[ells<lpad]))

        else:
            fillval = 0.            
            self._gCl[keyName] = interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=fillval)
        

        

    def gCl(self,keyName,ell):

        if len(keyName)==3:
            # assume uTT, lTT, etc.
            ultype = keyName[0].lower()
            if ultype=="u":
                return self.uCl(keyName[1:],ell)
            elif ultype=="l":
                return self.lCl(keyName[1:],ell)
            else:
                raise ValueError
        
        try:
            return self._gCl[keyName](ell)
        except:
            return self._gCl[keyName[::-1]](ell)
        
    def loadCls(self,ell,Cl,XYType="TT",lensed=False,interporder="linear",lpad=9000,fill_zero=True):

        # Implement ellnorm

        mapXYType = XYType.upper()
        validate_map_type(mapXYType)


        if not(fill_zero):
            fillval = Cl[ell<lpad][-1]
            f = lambda x: np.piecewise(x, [x<=lpad,x>lpad], [lambda y: interp1d(ell[ell<lpad],Cl[ell<lpad],bounds_error=False,fill_value=0.)(y),lambda y: fillval*(lpad/y)**4.])

        else:
            fillval = 0.            
            f = interp1d(ell[ell<lpad],Cl[ell<lpad],bounds_error=False,fill_value=fillval)
                    
        if lensed:
            self._lCl[XYType]=f
        else:
            self._uCl[XYType]=f

    def _Cl(self,XYType,ell,lensed=False):

            
        mapXYType = XYType.upper()
        validate_map_type(mapXYType)

        if mapXYType=="ET": mapXYType="TE"
        ell = np.array(ell)

        try:
            if lensed:    
                retlist = np.array(self._lCl[mapXYType](ell))
                return retlist
            else:
                retlist = np.array(self._uCl[mapXYType](ell))
                return retlist

        except:
            zspecs = ['EB','TB']
            if (XYType in zspecs) or (XYType[::-1] in zspecs):
                return ell*0.
            else:
                raise

    def uCl(self,XYType,ell):
        if self.always_lensed:
            assert not(self.always_unlensed)
            return self.lCl(XYType,ell)
        return self._Cl(XYType,ell,lensed=False)
    def lCl(self,XYType,ell):
        if self.always_unlensed:
            assert not(self.always_lensed)
            return self.uCl(XYType,ell)
        return self._Cl(XYType,ell,lensed=True)
    




def loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=True,skip_lens=False,dells=False,scalcls=True):
    '''
    Given a CAMB path+output_root, reads CMB and lensing Cls into 
    an orphics.theory.gaussianCov.TheorySpectra object.

    The spectra are stored in dimensionless form, so TCMB has to be specified. They should 
    be used with dimensionless noise spectra and dimensionless maps.

    All ell and 2pi factors are also stripped off.

 
    '''
    if not(get_dimensionless): TCMB = 1.
    if useTotal:
        uSuffix = "_totCls.dat"
        lSuffix = "_lensedtotCls.dat"
    else:
        if scalcls:
            uSuffix = "_scalCls.dat"
        else:
            uSuffix = "_lenspotentialCls.dat"
        lSuffix = "_lensedCls.dat"

    uFile = cambRoot+uSuffix
    lFile = cambRoot+lSuffix

    theory = TheorySpectra()

    ell, lcltt, lclee, lclbb, lclte = np.loadtxt(lFile,unpack=True,usecols=[0,1,2,3,4])
    lfact = 2.*np.pi/ell/(ell+1.) if not(dells) else 1
    mult = lfact/TCMB**2.
    lcltt *= mult
    lclee *= mult
    lclte *= mult
    lclbb *= mult
    theory.loadCls(ell,lcltt,'TT',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclte,'TE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclee,'EE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclbb,'BB',lensed=True,interporder="linear",lpad=lpad)

    if not(skip_lens):
        try:
            elldd, cldd = np.loadtxt(cambRoot+"_lenspotentialCls.dat",unpack=True,usecols=[0,5])
            clkk = 2.*np.pi*cldd/4.
        except:
            elldd, cldd = np.loadtxt(cambRoot+"_scalCls.dat",unpack=True,usecols=[0,4])
            clkk = cldd*(elldd+1.)**2./elldd**2./4./TCMB**2.

        theory.loadGenericCls(elldd,clkk,"kk",lpad=lpad)


    if unlensedEqualsLensed:

        theory.loadCls(ell,lcltt,'TT',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,lclte,'TE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,lclee,'EE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,lclbb,'BB',lensed=False,interporder="linear",lpad=lpad)

    else:
        if scalcls:
            ell, cltt, clee, clte = np.loadtxt(uFile,unpack=True,usecols=[0,1,2,3])
            lfact = 2.*np.pi/ell/(ell+1.) if not(dells) else 1
            mult = lfact/TCMB**2.
            cltt *= mult
            clee *= mult
            clte *= mult
            clbb = clee*0.
        else:
            ell, cltt, clee, clbb, clte = np.loadtxt(uFile,unpack=True,usecols=[0,1,2,3,4])
            lfact = 2.*np.pi/ell/(ell+1.) if not(dells) else 1
            mult = lfact/TCMB**2.
            cltt *= mult
            clee *= mult
            clte *= mult
            clbb *= mult

        theory.loadCls(ell,cltt,'TT',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,clte,'TE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,clee,'EE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,clbb,'BB',lensed=False,interporder="linear",lpad=lpad)

    theory.dimensionless = get_dimensionless
    return theory


def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)

def get_theory_dicts(nells=None,lmax=9000,grad=True):
    #thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
    thloc = '/global/homes/o/omard/fgestimates/'
    print('THEORY', thloc)
    ls = np.arange(lmax+1)
    ucls = {}
    tcls = {}
    theory = loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    ells,gt,ge,gb,gte = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1,2,3,4])
    if nells is None: nells = {'TT':0,'EE':0,'BB':0}
    ucls['tt'] = interp(ells,gt)(ls) if grad else theory.lCl('TT',ls)
    ucls['te'] = interp(ells,gte)(ls) if grad else theory.lCl('TE',ls)
    ucls['ee'] = interp(ells,ge)(ls) if grad else theory.lCl('EE',ls)
    ucls['bb'] = interp(ells,gb)(ls) if grad else theory.lCl('BB',ls)
    ucls['kk'] = theory.gCl('kk',ls)
    tcls['tt'] = theory.lCl('TT',ls) + nells['TT']
    tcls['te'] = theory.lCl('TE',ls)
    tcls['ee'] = theory.lCl('EE',ls) + nells['EE']
    tcls['bb'] = theory.lCl('BB',ls) + nells['BB']
    lcls = {}
    lcls['tt'] = theory.lCl('TT',ls) 
    lcls['te'] = theory.lCl('TE',ls)
    lcls['ee'] = theory.lCl('EE',ls)
    lcls['bb'] = theory.lCl('BB',ls)

    return ucls, tcls, lcls


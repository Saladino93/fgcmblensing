import yaml

import camb

from scipy import optimize as sopt


def read_configuration(filename: str) -> dict:
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data



class ConfigurationReader(object):
    def __init__(self, filename: str):
        self.filename = filename
        config = self._read_configuration(filename = filename)
        self._set_attrs(config, self)

    @staticmethod
    def _set_attrs(config: dict, object_name):
        for key, value in config.items():
            if type(value) is dict:
                #allow only for 'shallow' dictionaries, in the sense that a dictionary does not containt another one (apart from the main file)
                for k, v in value.items():
                    if 'lambda' in str(v):
                        vv = eval(str(v))
                    else:
                        vv = v
                    setattr(object_name, k, vv)
            else:
                if 'lambda' in str(value):
                    vvalue = eval(str(value))
                else:
                    vvalue = value
                setattr(object_name, key, eval(str(vvalue)))
        return None

    @staticmethod
    def _read_configuration(filename: str) -> dict:
        return read_configuration(filename = filename)


class ConfigurationCosmoReader(ConfigurationReader):
    def __init__(self, filename: str):
        
        super().__init__(filename = filename)
        self.H0 = self.h*100
        self.ombh2 = self.omega_b*self.h**2
        self.omegac = self.omega_m-self.omega_b
        self.omch2 = self.omegac*self.h**2.
        try:
            getattr(self, 'As')
        except:
            self.As = self.As_from_sigma8(self.H0, self.ombh2, self.omch2, self.ns, sigma8_wanted = self.sigma8)
            

    @staticmethod
    def get_sigma8_function(pars, ns, sigma8_wanted):
        def get_sigma8(As):
            pars.InitPower.set_params(ns = ns, As = As)
            pars.WantTransfer = True
            #pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)
            results = camb.get_results(pars)
            s8 = results.get_sigma8()[0] #should be at redshift 0
            return s8-sigma8_wanted
        return get_sigma8

    @staticmethod
    def As_from_sigma8(H0: float, ombh2: float, omch2: float, ns: float, sigma8_wanted: float, Asinit: float = 2e-9, arel: float = 0.5, brel: float = 1.5):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0 = H0, ombh2 = ombh2, omch2 = omch2)
        get_sigma8 = ConfigurationCosmoReader.get_sigma8_function(pars, ns, sigma8_wanted)
        #NOTE: this root finding is ok for fixed cosmology of other parameters
        #otherwise you would have to run in a different way
        #do not check if 'converged' key is True or not... 
        return sopt.root_scalar(get_sigma8, bracket = [Asinit*arel, Asinit*brel], method = 'bisect', rtol = 1e-5).root

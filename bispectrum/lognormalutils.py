import numpy as np
import flt
import healpy as hp


def cl2xi(cl: np.ndarray, closed = False):
    ls = np.arange(0, len(cl))
    factorcl = (2*ls+1)/(4*np.pi)
    coeffs = cl*factorcl
    return flt.idlt(coeffs, closed = closed)


def theta(n, closed = False):
    '''
    Returns the theta for which the cl2xi are calculated, for a given n
    '''
    return flt.theta(n, closed = closed)


def xi2cl(xi: np.ndarray, closed = False):
    ls = np.arange(0, len(xi))
    factorcl = (2*ls+1)/(4*np.pi)
    return flt.dlt(xi, closed = closed)/factorcl


def get_mean_from_map(mappa: np.ndarray):
    return np.mean(mappa)

def get_variance_from_map(mappa: np.ndarray):
    return np.mean(mappa**2.)-np.mean(mappa)**2.

def get_skew_from_map(mappa: np.ndarray):
    return np.mean((mappa-get_mean_from_map(mappa))**3.)/np.mean(mappa**2.)**1.5

def y_skew(skew):
    '''
    Formula (12) from https://arxiv.org/pdf/1602.08503.pdf
    '''
    result = 2+skew**2.+skew*np.sqrt(4+skew**2.)
    result /= 2
    return np.power(result, 1/3.)

def get_lambda_from_skew(skew, var, mu):
    lmbda = np.sqrt(var)/skew*(1+y_skew(skew)+1/y_skew(skew))-mu
    return lmbda 

def get_alpha(mu, lmbda):
    '''
    Below formula (7) from https://arxiv.org/pdf/1602.08503.pdf
    '''
    return mu+lmbda

def get_mu_gauss(alpha, var):
    '''
    Gets the mu parameter for the Gaussian distribution for the log-normal
    '''
    result = np.log(alpha**2./np.sqrt(var+alpha**2.))
    return result

def get_sigma_gauss(alpha, var):
    '''
    Gets the sigma parameter for the Gaussian distribution for the log-normal.

    Here the variance is the variance of the wanted log-normal field.
    '''
    result = np.log(1+var/alpha**2.)
    result = np.sqrt(result)
    return result

suppress = lambda l, lsup, supindex: np.exp(-1.0*np.power(l/lsup, supindex))

def process_cl(inputcl: np.ndarray, lsup: float = 7000, supindex: float = 10):
    ls = np.arange(0, len(inputcl))
    result = inputcl*suppress(ls, lsup, supindex)
    return result
    
def get_alpha(mu, lmbda):
    '''
    Below formula (7) from https://arxiv.org/pdf/1602.08503.pdf
    '''
    return mu+lmbda



def create_lognormal_single_map(inputcl: np.ndarray, nside: int, lmax_gen: int, mu: float = 0.0, lamb: float = 0.0):
    alpha = get_alpha(mu, lamb)
    xisinput = cl2xi(inputcl)/alpha/alpha

    xigaussian = np.log(xisinput+1)
    clgaussian = xi2cl(xigaussian)

    vargauss = np.dot(np.arange(1, 2*len(clgaussian), 2), clgaussian)/(4*np.pi)

    lmax_gen = 2*nside-1 if lmax_gen is None else lmax_gen
    almgaussian = hp.synalm(clgaussian, lmax = lmax_gen) #GENERATE TO HIGH LMAX
    maps = hp.alm2map(almgaussian, nside = nside, pol = False)
    
    #vargauss = np.array([xigaussian[i, i][0] for i in range(Nfields)])
    #vargauss = np.array([np.var(m) for m in maps])

    expmu = (mu+lamb)*np.exp(-vargauss*0.5)
    maps = np.array(maps)
    maps = np.exp(maps) 
    maps *= expmu
    maps -= lamb
    return maps


#shifted log-normal distribution
def shifted_lognormal_zero_mean(x, sigmaG, lamb):
    #equation (22) of https://www.aanda.org/articles/aa/pdf/2011/12/aa17294-11.pdf
    return np.exp(-(np.log(x/lamb+1)+sigmaG**2/2)**2./(2.*sigmaG**2.))/(x+lamb)/sigmaG/np.sqrt(2.*np.pi)*(x>-lamb)

def get_out_quantities(outmap: np.ndarray):
    skewness = get_skew_from_map(outmap)
    variance = get_variance_from_map(outmap)
    mean = get_mean_from_map(outmap)
    lamb = get_lambda_from_skew(skewness, variance, mean)

    alpha = get_alpha(mean, lamb)
    sigmaG = get_sigma_gauss(alpha, variance)
    muG = get_mu_gauss(alpha, variance)
    return mean, variance, skewness, lamb, muG, sigmaG

def randomizing_fg(mappa: np.ndarray):
     f = lambda z: np.abs(z) * np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
     return f(mappa)
import numpy as np
import healpy as hp


'''
Copied from lenscarf, or D.lensalot
'''
def alm_copy(alm, mmaxin:int or None, lmaxout:int, mmaxout:int):
    """Copies the healpy alm array, with the option to change its lmax
        Parameters
        ----------
        alm :ndarray
            healpy alm array to copy.
        mmaxin: int or None
            mmax parameter of input array (can be set to None or negative for default)
        lmaxout : int
            new alm lmax
        mmaxout: int
            new alm mmax
    """
    lmaxin = hp.Alm.getlmax(alm.size, mmaxin)
    if mmaxin is None or mmaxin < 0: mmaxin = lmaxin
    if (lmaxin == lmaxout) and (mmaxin == mmaxout):
        ret = np.copy(alm)
    else:
        ret = np.zeros(hp.Alm.getsize(lmaxout, mmaxout), dtype = np.complex)
        lmax_min = min(lmaxout, lmaxin)
        for m in range(0, min(mmaxout, mmaxin) + 1):
            idx_in =  m * (2 * lmaxin + 1 - m) // 2 + m
            idx_out = m * (2 * lmaxout+ 1 - m) // 2 + m
            ret[idx_out: idx_out + lmax_min + 1 - m] = alm[idx_in: idx_in + lmax_min + 1 - m]
    return ret

import numpy as np




class SONoiseReader(object):
    def __init__(self, filename: str, nus = [27, 39, 93, 145, 225, 280], cross_nus: list = [(0, 1), (2, 3), (4, 5)]):
        
        #ell, nl27, nl39, nl93, nl145, nl225, nl280, nl27x39, nl93x145, nl225x280 = np.loadtxt(filename, unpack = True)
        data = np.loadtxt(filename).T
        self.ell = data[0, :]
        self.data = data[1:, :]
        self.nus = nus
        self.cross_nus = [tuple(cross) for cross in cross_nus]#not sure if yaml file allows tuples

    def get_noise(self, nu1: int, nu2: int = None):
        if nu2 is None:
            nu2 = nu1
        ind1 = self.nus.index(nu1)
        ind2 = self.nus.index(nu1) #I assume we usually only have <10 frequencies, so even if nu2 = nu1 do not have to make if statement for ind2 from ind1 in case of equality

        if ind1 != ind2:
            position = self.cross_nus.index((ind1, ind2))+len(self.nus)
        else:
            position = ind1

        return self.data[position, :]
        
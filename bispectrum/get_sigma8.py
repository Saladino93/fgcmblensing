import camb
import numpy as np


pars = camb.CAMBparams()

H0 = 67.1
Omegam = 0.315
ommh2 = Omegam*H0**2/100**2
Omegab = 0.049
ombh2 = Omegab*H0**2/100**2
omch2 = ommh2 - ombh2
Omegam = ommh2/(H0/100)**2
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(As = 2.215*1e-9, ns=0.968)

zm = np.logspace(-9, np.log(2000), 100)
zm = np.append(0, zm)
pars.set_matter_power(redshifts = zm, kmax = 1)

results = camb.get_results(pars)

s8 = np.array(results.get_sigma8())
np.savetxt('sigma8.txt', np.c_[zm, s8[::-1]])

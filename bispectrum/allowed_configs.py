import numpy as np

import pickle

from joblib import Parallel, delayed

#https://github.com/toshiyan/cmblensplus/blob/dcd212906da8039f63839d69e8bb45ebccd55d09/F90/src_utils/bstool.f90#L988
def W3j_approx(l1,l2,l3):
  #ind = np.where((l1+l2+l3)%2 != 0)
  if (l1+l2+l3)%2 != 0:
    result = 0
  else:
    Lh = (l1+l2+l3)*0.5
    a1 = ((Lh-l1+0.5)/(Lh-l1+1))**(Lh-l1+0.25)
    a2 = ((Lh-l2+0.5)/(Lh-l2+1))**(Lh-l2+0.25)
    a3 = ((Lh-l3+0.5)/(Lh-l3+1))**(Lh-l3+0.25)
    b = 1/((Lh-l1+1)*(Lh-l2+1)*(Lh-l3+1))**(0.25)
    result = (-1)**Lh/np.sqrt(2*np.pi) * np.exp(1.5)* (Lh+1)**(-0.25) * a1*a2*a3*b
  #result[ind] = 0
  return result

lmax = 1000 #4500
lmin = 2

ells = np.arange(lmin, lmax, 1)

batch_size = 100
n_jobs = 4
backend="loky"

def loop(l1):
    results = []
    for l2 in range(l1, lmax):
        for l3 in range(l2, lmax):
            if (l3>l1+l2) or (l3<abs(l1-l2)):
                continue
            if (l1>l2+l3) or (l1<abs(l2-l3)):
                continue
            if (l2>l3+l1) or (l2<abs(l3-l1)):
                continue
            if (((l1+l2+l3)%2)==1):
                continue             
            results += [(l1, l2, l3)] #=W3j_approx(l1, l2, l3)
    return results

import time

start = time.time()
results = Parallel(n_jobs = n_jobs, batch_size = batch_size, backend = backend, verbose = 0)(delayed(loop)(l1) for l1 in ells)
end = time.time()
print(end - start)

with open("allowed_configs", "wb") as fp: 
    pickle.dump(results, fp)
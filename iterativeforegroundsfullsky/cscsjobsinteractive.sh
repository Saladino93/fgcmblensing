#!/bin/sh -l
srun --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "postborn"
srun --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "postborngauss"
srun --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "postbornrand"
srun --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case ""
srun --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "rand"
srun --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "gauss"
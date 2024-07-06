#!/bin/sh -l

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate


srun -n 3 --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 3 -tol 7 -case "postborn"
srun -n 3  --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 3 -tol 7 -case "postborngauss"
srun -n 3  --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 3 -tol 7 -case "postbornrand"
srun -n 3  --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 3 -tol 7 -case ""
srun -n 3  --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 3 -tol 7 -case "rand"
srun -n 3  --cpus-per-task=8 python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 3 -tol 7 -case "gauss"


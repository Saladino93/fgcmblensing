#!/bin/sh -l
#SBATCH --job-name=lenscarfflipped
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --nodes=8
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1cd 
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=false 

srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 16 -itmax 10 -tol 7 -case "bornflipped"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case ""
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "rand"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "gauss"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "postborngauss"

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "rand" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "gauss" -v ""


#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 7 -itmax 10 -tol 7 -case "logdoubleskew" -v logprior
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 7 -itmax 10 -tol 7 -case "randlogdoubleskew" -v logprior
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 7 -itmax 10 -tol 7 -case "gausslogdoubleskew" -v logprior

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 7 -itmax 10 -tol 7 -case "logdoubleskew" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 7 -itmax 10 -tol 7 -case "randlogdoubleskew" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 7 -itmax 10 -tol 7 -case "gausslogdoubleskew" -v ""

#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "randlogdoubleskew"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "postlog"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "postlogrand"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "rand"
#run python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "gauss"


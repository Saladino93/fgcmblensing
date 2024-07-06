#!/bin/sh -l
#SBATCH --job-name=lenscarflog
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --nodes=16
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=6
export OMP_PLACES=threads
export OMP_PROC_BIND=false 

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 7 -tol 7 -case "log" 
srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 7 -tol 7 -case "randlog"

#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 15 -itmax 8 -tol 7 -case "log" -v logpriornew
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 15 -itmax 8 -tol 7 -case "randlog" -v logpriornew

#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "logdoubleskew"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "randlogdoubleskew"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "postlog"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 64 -itmax 2 -tol 7 -case "postlogrand"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "rand"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "gauss"
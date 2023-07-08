#!/bin/sh -l
#SBATCH --job-name=lenscarf
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=debug

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=false 

srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "postborn"
srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "postborngauss"
srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "postbornrand"
srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case ""
srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "rand"
srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 5 -tol 7 -case "gauss"

#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "log"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "randlog"
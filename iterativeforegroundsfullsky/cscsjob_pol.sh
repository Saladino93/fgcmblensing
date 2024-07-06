#!/bin/sh -l
#SBATCH --job-name=jobpol
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false


srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 1 -itmax 6 -tol 6 -case "" -v "lB10" 
srun python ./itfgs/params/S4n32_lB_200.py -k p_p -imin 0 -imax 1 -itmax 6 -tol 6 -case "" -v "lB200" 

srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 1 -itmax 6 -tol 8 -case "" -v "lB10"
srun python ./itfgs/params/S4n32_lB_200.py -k p -imin 0 -imax 1 -itmax 6 -tol 8 -case "" -v "lB200"

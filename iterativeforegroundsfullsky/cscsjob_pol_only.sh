#!/bin/sh -l
#SBATCH --job-name=lenscarf
#SBATCH --time=03:00:00
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

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=false 


#srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 64 -itmax 10 -tol 6 -case ""
#srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 64 -itmax 10 -tol 6 -case "rand"
#srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 64 -itmax 10 -tol 6 -case "gauss"

#srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 64 -itmax 10 -tol 6 -case "postborn"
srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 63 -itmax 10 -tol 6 -case "postbornrand"
srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 63 -itmax 10 -tol 6 -case "postborngauss"
#!/bin/sh -l
#SBATCH --job-name=lenscarflog
#SBATCH --time=00:55:00
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --constraint=gpu
#SBATCH --nodes=8
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=false 

srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 200 -imax 223 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "gauss" -sim_index 72


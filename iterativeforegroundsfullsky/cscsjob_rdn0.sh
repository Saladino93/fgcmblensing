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

srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 72
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 73
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 74
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 75
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 76
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 77
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 78
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 79
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 80
srun python ./itfgs/params/S4n32_rdn0.py -k ptt -imin 240 -imax 263 -itmax 0 -tol 7 -case "gauss" -v "rdn0" -case_specific "" -sim_index 81

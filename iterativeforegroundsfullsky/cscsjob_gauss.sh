#!/bin/sh -l
#SBATCH --job-name=lenscarf
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 0 -itmax 15 -tol 7 -case "gauss" -v ""
srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 0 -itmax 15 -tol 7 -case "gauss" -v bh_prova_aniso_filt_qe_start
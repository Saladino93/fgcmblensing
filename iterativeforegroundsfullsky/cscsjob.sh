#!/bin/sh
#SBATCH --job-name=lenscarf
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=mc
#SBATCH --nodes=1
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=s1203

module load cray-python/3.9.4.1 

source ~/lenscarf/bin/activate

export OMP_NUM_THREADS=8

export OMP_PLACES=threads

export OMP_PROC_BIND=false 

srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case ""
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "rand"
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "gauss"
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "log"
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "randlog"

srun -n 32 --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "postborn"


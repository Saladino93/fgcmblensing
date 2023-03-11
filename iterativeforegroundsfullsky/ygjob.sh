#!/bin/sh
#SBATCH --job-name=lenscarf
#SBATCH --time=11:00:00
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --output=/home/users/d/darwish/scratch/slurms/lenscarf-slurm-%J.out


ml GCC OpenMPI Python SciPy-bundle/2022.11 NVHPC

module load GCC

conda activate lenscarfn

export OMP_NUM_THREADS=8

export OMP_PLACES=threads

export OMP_PROC_BIND=false 

export SCRATCH=/home/users/d/darwish/scratch/

srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case ""
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "rand"
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "gauss"
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "log"
srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "randlog"

#module load GCC OpenMPI Anaconda3 Autotools/20220317 FFTW GSL
#module load NVHPC/20.7
#
#ml GCC OpenMPI Python SciPy-bundle/2022.11 NVHPC
#module load GCC
#conda activate lenscarfn
#srun -n 2 --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 32 -itmax 4 -tol 7

#ml GCC OpenMPI Python SciPy-bundle/2022.11 NVHPC
#module load GCC
#conda activate lenscarfn
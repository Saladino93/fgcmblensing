#!/bin/bash
#SBATCH -N 64
#SBATCH -A mp107
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=omar.darwish@unige.ch
#SBATCH -t 05:00:00
#SBATCH -o /pscratch/sd/o/omard/slurmouts/slurm-%j.out

export OMP_NUM_THREADS=8

export OMP_PLACES=threads

export OMP_PROC_BIND=false 

source /global/homes/o/omard/.conda/envs/cobaya/bin/activate lenscarf

srun -n 64 -c 8 --cpu_bind=cores python ./itfgs/params/SOGaussianOnly.py -imin 0 -imax 128 -itmax 6 -k ptt -v tol7 -tol 7

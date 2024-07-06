#!/bin/sh -l
#SBATCH --job-name=lenscarf
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --nodes=16
#SBATCH --output=/scratch/snx3000/odarwish/slurms/n32_delensing-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=false


srun python ./itfgs/params/S4n32_delensing_bias.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 8 -case "" -v "delensing_indep_pins_new"
srun python ./itfgs/params/S4n32_delensing_bias.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 8 -case "rand" -v "delensing_indep_pins_new"
srun python ./itfgs/scripts/analyzeresults.py -k "ptt" -imin 0 -imax 31 -itmax 1 -v "delensing_indep_pins_new" -s "born"

#srun python ./itfgs/params/S4n32_delensing_bias.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 8 -case "gauss" -v "delensing_fixed_noise"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "rand" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "gauss" -v ""

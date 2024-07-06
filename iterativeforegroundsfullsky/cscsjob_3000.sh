#!/bin/sh -l
#SBATCH --job-name=lenscarf
#SBATCH --time=08:10:00
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


#srun python ./itfgs/params/S4n32_3000.py -k ptt -imin 0 -imax 63 -itmax 5 -tol 7 -case "" -v "3000" 
#srun python ./itfgs/params/S4n32_3000.py -k ptt -imin 0 -imax 63 -itmax 5 -tol 7 -case "rand" -v "3000" 
#srun python ./itfgs/params/S4n32_3000.py -k ptt -imin 0 -imax 63 -itmax 5 -tol 7 -case "gauss" -v "3000" 
srun python ./itfgs/scripts/analyzeresults_3000.py -k "ptt" -imin 0 -imax 63 -itmax 5 -v "3000" -s "born"

srun python ./itfgs/params/S4n32_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postborn" -v "3000"
srun python ./itfgs/params/S4n32_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postbornrand" -v "3000"
srun python ./itfgs/params/S4n32_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postborngauss" -v "3000"
srun python ./itfgs/scripts/analyzeresults.py -k "p_p" -imin 0 -imax 63 -itmax 6 -v "3000" -s "postborn"


srun python ./itfgs/params/S4n32_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "postborn" -v "3000" 
srun python ./itfgs/params/S4n32_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "postbornrand" -v "3000" 
srun python ./itfgs/params/S4n32_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "postborngauss" -v "3000" 
srun python ./itfgs/scripts/analyzeresults.py -k "p" -imin 0 -imax 63 -itmax 3 -v "3000" -s "postborn"


srun python ./itfgs/params/S4n32_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 7  -case "" -v "3000"
srun python ./itfgs/params/S4n32_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 7 -case "rand" -v "3000"
srun python ./itfgs/params/S4n32_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 7 -case "gauss" -v "3000"
srun python ./itfgs/scripts/analyzeresults.py -k "p_p" -imin 0 -imax 63 -itmax 6 -v "3000" -s "born"


srun python ./itfgs/params/S4n32_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "" -v "3000"
srun python ./itfgs/params/S4n32_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "rand" -v "3000"
srun python ./itfgs/params/S4n32_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "gauss" -v "3000"
srun python ./itfgs/scripts/analyzeresults.py -k "p" -imin 0 -imax 63 -itmax 3 -v "3000" -s "born"





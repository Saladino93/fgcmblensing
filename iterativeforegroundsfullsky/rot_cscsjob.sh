#!/bin/sh -l
#SBATCH --job-name=jointrot
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=32
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false 

srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 6 -tol 8 -case "rot" -v ""
srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 6 -tol 8 -case "rotgauss" -v ""
srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 6 -tol 8 -case "postborn"

#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 31 -itmax 10 -tol 7 -case "rot" -v ""

#srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 10 -tol 5 -case "rotrand" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 10 -tol 5 -case "rot" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 10 -tol 5 -case "rotgauss" -v ""

#srun python ./itfgs/params/S4n32_rotation.py -k p_p -imin 0 -imax 31 -itmax 10 -tol 5 -case "rotrand" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k p_p -imin 0 -imax 31 -itmax 10 -tol 5 -case "rot" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k p_p -imin 0 -imax 31 -itmax 10 -tol 5 -case "rotgauss" -v ""

#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 31 -itmax 10 -tol 7 -case "rotrand" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 31 -itmax 10 -tol 7 -case "rot" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 31 -itmax 10 -tol 7 -case "rotgauss" -v ""

#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 16 -itmax 8 -tol 7 -case "rot" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 16 -itmax 8 -tol 7 -case "rotrand" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k ptt -imin 0 -imax 16 -itmax 8 -tol 7 -case "rotgauss" -v ""

#srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 8 -tol 7 -case "rot" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 8 -tol 7 -case "rand" -v ""
#srun python ./itfgs/params/S4n32_rotation.py -k p -imin 0 -imax 31 -itmax 8 -tol 7 -case "gauss" -v ""

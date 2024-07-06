#!/bin/sh -l
#SBATCH --job-name=flippedlenscarf
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

#srun python ./itfgs/params/S4n32_flipped_3000.py -k ptt -imin 0 -imax 63 -itmax 5 -tol 8 -case "" -v "flipped3000"
#srun python ./itfgs/params/S4n32_flipped_3000.py -k ptt -imin 0 -imax 63 -itmax 5 -tol 8 -case "rand" -v "flipped3000"
#srun python ./itfgs/params/S4n32_flipped_3000.py -k ptt -imin 0 -imax 63 -itmax 5 -tol 8 -case "gauss" -v "flipped3000"
srun python ./itfgs/scripts/analyzeresults_3000.py -k "ptt" -imin 0 -imax 63 -itmax 5 -v "flipped3000" -s "born"


srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postborn" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postbornrand" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postborngauss" -v "flipped3000"
srun python ./itfgs/scripts/analyzeresults_3000.py -k "p_p" -imin 0 -imax 63 -itmax 6 -v "flipped3000" -s "postborn"


srun python ./itfgs/params/S4n32_flipped_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "postborn" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "postbornrand" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "postborngauss" -v "flipped3000"
srun python ./itfgs/scripts/analyzeresults_3000.py -k "p" -imin 0 -imax 63 -itmax 3 -v "flipped3000" -s "postborn"


srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 7  -case "" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 7 -case "rand" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 7 -case "gauss" -v "flipped3000"
srun python ./itfgs/scripts/analyzeresults_3000.py -k "p_p" -imin 0 -imax 63 -itmax 6 -v "flipped3000" -s "born"


srun python ./itfgs/params/S4n32_flipped_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "rand" -v "flipped3000"
srun python ./itfgs/params/S4n32_flipped_3000.py -k p -imin 0 -imax 63 -itmax 3 -tol 8 -case "gauss" -v "flipped3000"
srun python ./itfgs/scripts/analyzeresults_3000.py -k "p" -imin 0 -imax 63 -itmax 3 -v "flipped3000" -s "born"


#srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postborn" -v "flipped"
#srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postbornrand" -v "flipped"
#srun python ./itfgs/params/S4n32_flipped_3000.py -k p_p -imin 0 -imax 63 -itmax 6 -tol 8 -case "postborngauss" -v "flipped"
#srun python ./itfgs/scripts/analyzeresults.py -k "p_p" -imin 0 -imax 63 -itmax 5 -v "flipped" -s "postborn"

#srun python ./itfgs/params/S4n32_flipped.py -k p -imin 0 -imax 63 -itmax 3 -tol 7 -case "" -v "flipped"
#srun python ./itfgs/params/S4n32_flipped.py -k p -imin 0 -imax 63 -itmax 3 -tol 7 -case "rand" -v "flipped"
#srun python ./itfgs/params/S4n32_flipped.py -k p -imin 0 -imax 63 -itmax 3 -tol 7 -case "gauss" -v "flipped"
#srun python ./itfgs/scripts/analyzeresults.py -k "p" -imin 0 -imax 63 -itmax 3 -v "flipped" -s "born"

#srun python ./itfgs/params/S4n32_flipped.py -k p -imin 0 -imax 63 -itmax 3 -tol 7 -case "postborn" -v "flipped"
#srun python ./itfgs/params/S4n32_flipped.py -k p -imin 0 -imax 63 -itmax 3 -tol 7 -case "postbornrand" -v "flipped"
#srun python ./itfgs/params/S4n32_flipped.py -k p -imin 0 -imax 63 -itmax 3 -tol 7 -case "postborngauss" -v "flipped"
#srun python ./itfgs/scripts/analyzeresults.py -k "p" -imin 0 -imax 63 -itmax 3 -v "flipped" -s "postborn"



#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "" -v "no_prior"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "rand" -v "no_prior"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "gauss" -v "no_prior"

#srun python ./itfgs/params/S4n32_true_phi.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "" -v "pin_WF_noisy"
#srun python ./itfgs/params/S4n32_true_phi.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "rand" -v "pin_WF_noisy"
#srun python ./itfgs/params/S4n32_true_phi.py -k ptt -imin 0 -imax 31 -itmax 1 -tol 7 -case "gauss" -v "pin_WF_noisy"

#srun python ./itfgs/scripts/analyzeresults.py -k "ptt" -imin 0 -imax 31 -itmax 1 -v "pin_WF_noisy" -s "born_pin"

#srun python ./itfgs/params/S4n32_mean_field_new.py -k ptt -imin 0 -imax 15 -itmax 4 -tol 8 -case "" -v "mf_new"

#srun python ./itfgs/params/S4n32_lmax_5000.py -k ptt -imin 0 -imax 0 -itmax 9 -tol 8 -case "rand"

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k p_p -imin 0 -imax 63 -itmax 10 -tol 5 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 5 -tol 4 -case "postborn"

#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 31 -itmax 7 -tol 8 -case "" -v logprior_tol8

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 7 -tol 8 -case ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 7 -tol 8 -case "rand"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 31 -itmax 7 -tol 8 -case "postborn" -v hightol8

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 15 -itmax 15 -tol 7 -case "" -v joint
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 15 -itmax 15 -tol 7 -case "rand" -v joint
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 15 -itmax 15 -tol 7 -case "gauss" -v joint

#srun python ./itfgs/params/S4n32.py -k p -imin 440 -imax 442 -itmax 7 -tol 6 -case "gauss"
#srun python ./itfgs/params/S4n32_mean_field_new.py -k ptt -imin 0 -imax 31 -itmax 5 -tol 7 -case "" -v "mf_new"
#srun python ./itfgs/params/S4n32_mean_field_new.py -k ptt -imin 0 -imax 31 -itmax 5 -tol 7 -case "rand" -v "mf_new"
#srun python ./itfgs/params/S4n32_mean_field_new.py -k ptt -imin 32 -imax 33 -itmax 8 -tol 7 -case "gauss" -v "mf_new"

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 63 -itmax 0 -tol 7 -case "bornflipped"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case ""
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "rand"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "postborngauss"

#srun python ./itfgs/params/S4n32.py -k ptt -imin 200 -imax 303 -itmax 0 -tol 6 -case "gauss"

#srun python ./itfgs/params/S4n32_mean_field.py -k ptt -imin 0 -imax 1 -itmax 8 -tol 7 -case "gauss"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "postborngauss"

#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 7 -itmax 8 -tol 7 -case "" -v logpriornew
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 7 -itmax 8 -tol 7 -case "rand" -v logpriornew
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 7 -itmax 8 -tol 7 -case "gauss" -v logpriornew

#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 32 -itmax 8 -tol 7 -case "rot" -v ""
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 32 -itmax 8 -tol 7 -case "rand" -v ""
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 32 -itmax 8 -tol 7 -case "gauss" -v ""

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 100 -itmax 10 -tol 7 -case "" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 100 -itmax 10 -tol 7 -case "rand" -v ""
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 100 -itmax 10 -tol 7 -case "gauss" -v ""

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 100 -itmax 10 -tol 7 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 100 -itmax 10 -tol 7 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 95 -itmax 10 -tol 7 -case "postborngauss"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case ""
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "rand"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "gauss"

#srun python ./itfgs/params/S4n32.py -k p_bh_s -imin 0 -imax 31 -itmax 0 -tol 6 -case ""
#srun python ./itfgs/params/S4n32.py -k p_bh_s -imin 0 -imax 31 -itmax 0 -tol 6 -case "rand"
#srun python ./itfgs/params/S4n32.py -k p_bh_s -imin 0 -imax 31 -itmax 0 -tol 6 -case "gauss"

#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "gauss"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case ""
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "rand"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "postborngauss"
#srun python ./itfgs/params/S4n32.py -k p -imin 0 -imax 31 -itmax 7 -tol 6 -case "postborn"

#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "" -v logpriornew
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "rand" -v logpriornew
#srun python ./itfgs/params/S4n32lnpri.py -k ptt -imin 0 -imax 63 -itmax 10 -tol 7 -case "gauss" -v logpriornew

#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 128 -itmax 2 -tol 7 -case "postborngauss"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case ""
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "rand"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 63 -itmax 0 -tol 7 -case "gauss"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 32 -itmax 0 -tol 7 -case "postborn"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 32 -itmax 0 -tol 7 -case "postbornrand"
#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 32 -itmax 0 -tol 7 -case "postborngauss"

#srun python ./itfgs/params/S4n32.py -k ptt_bh_s -imin 0 -imax 32 -itmax 0 -tol 7 -case ""

#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "logdoubleskew"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "randlogdoubleskew"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "postlog"
#srun --cpu_bind=cores python ./itfgs/params/S4n32.py -k ptt -imin 0 -imax 99 -itmax 2 -tol 7 -case "postlogrand"

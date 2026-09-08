#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
#SBATCH --job-name=mc2h
#SBATCH --account=desi
#SBATCH --array=0,1,2,4


source /global/common/software/desi/users/adematti/cosmodesi_environment.sh
export OMP_NUM_THREADS=1
cosmolist=(c000 c001 c002 c003 c004)
cosmo=${cosmolist[$SLURM_ARRAY_TASK_ID]}
srun -n 1 -c 8 --cpu-bind=cores python -u fit_abaucs_secondary.py --cosmo $cosmo --reparam_option "hybrid"

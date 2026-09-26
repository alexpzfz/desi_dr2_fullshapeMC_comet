#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
#SBATCH --job-name=mc2h
#SBATCH --account=desi
#SBATCH --array=0,1,2,4

# Resolve the repo root. Slurm runs a spool copy of this script, so under
# sbatch BASH_SOURCE points at the spool dir; ask Slurm for the original path.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _script=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
else
    _script="${BASH_SOURCE[0]}"
fi
REPO_ROOT="$(cd "$(dirname "$_script")/.." && pwd)"

# NERSC-only software environment.
if [ -n "${NERSC_HOST:-}" ]; then source /global/common/software/desi/users/adematti/cosmodesi_environment.sh; fi
export OMP_NUM_THREADS=1
cosmolist=(c000 c001 c002 c003 c004)
cosmo=${cosmolist[$SLURM_ARRAY_TASK_ID]}
srun -n 1 -c 8 --cpu-bind=cores python -u "$REPO_ROOT/src/fit_abacus_secondary.py" --cosmo $cosmo --reparam_option "hybrid"

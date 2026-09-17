#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=00:30:00
#SBATCH --ntasks=64
#SBATCH --constraint=cpu
#SBATCH --job-name=post_sc
#SBATCH --account=desi
#SBATCH --output=/global/u2/a/alexpzfz/desi_dr2_fullshapeMC_comet/outputs/logs/%x_%j.out

# Hardcoded (not derived from BASH_SOURCE): Slurm copies this script into its
# spool dir before running it, so BASH_SOURCE points at the spool copy on the
# compute node, not this file's real location.
REPO_ROOT="/global/u2/a/alexpzfz/desi_dr2_fullshapeMC_comet"

export OMP_NUM_THREADS=1

# Adding sigma8/Omega_m to every sample of every scale_config chain is too
# slow to run on a login node; this spreads the per-sample emulator/CLASS
# evaluations in postprocessing/post_scale_config.py across MPI ranks.
srun -n "$SLURM_NTASKS" python -u "$REPO_ROOT/postprocessing/post_scale_config.py" "$@"

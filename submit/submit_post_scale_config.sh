#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=00:30:00
#SBATCH --ntasks=64
#SBATCH --constraint=cpu
#SBATCH --job-name=post_sc
#SBATCH --account=desi
# --output is relative to where sbatch is run: submit from the repo root, or
# pass --output explicitly (the run_*.sh wrappers do).
#SBATCH --output=outputs/logs/%x_%j.out

# Resolve the repo root. Slurm runs a spool copy of this script, so under
# sbatch BASH_SOURCE points at the spool dir; ask Slurm for the original path.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _script=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
else
    _script="${BASH_SOURCE[0]}"
fi
REPO_ROOT="$(cd "$(dirname "$_script")/.." && pwd)"

export OMP_NUM_THREADS=1

# Adding sigma8/Omega_m to every sample of every scale_config chain is too
# slow to run on a login node; this spreads the per-sample emulator/CLASS
# evaluations in postprocessing/post_scale_config.py across MPI ranks.
srun -n "$SLURM_NTASKS" python -u "$REPO_ROOT/postprocessing/post_scale_config.py" "$@"

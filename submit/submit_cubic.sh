#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=cpu
#SBATCH --job-name=mc_cubic
#SBATCH --account=desi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Map each array index to one tracer.
tracers=("LRG" "ELG" "QSO")

# Power spectrum settings (fit_cubic.py convention)
kminP="0.02 0.02"
kmaxP="0.35 0.3"
ellP="0 2"

# Bispectrum settings (fit_cubic.py convention)
kminB="0.02 0.02"
kmaxB="0.20 0.15"
ellB="0,0,0 2,0,2"

# Other settings
hod='fiducial'
cosmo='c000'
freedom='interm'
de_model='lambda'
reparam='full'
counterterm_basis='DESIct'
outdir="$REPO_ROOT/outputs/chains"
n_live='2000'

print_usage() {
        script_name=$(basename "$0")
        cat <<EOF
Usage:
    sbatch [sbatch options] $script_name -- [fit_cubic.py options]

Important:
    Use '--' to separate sbatch options from python/script options.
    Without '--', sbatch may try to parse your overrides as its own options.

Modes:
    Array mode: Each array task fits one tracer independently
      - Must specify --array and --cpus-per-task in sbatch options
    Joint mode: Single job fits all tracers together (use --joint flag)
      - Defaults to 48 cpus-per-task for all 3 tracers

Examples:
    Array mode (fits one tracer per task):
      sbatch --array=0-2 --cpus-per-task=8 $script_name -- --kmaxP 0.30 0.25
      sbatch --array=1 --cpus-per-task=8 $script_name -- --bispec --kmaxB 0.18 0.15
    
    Joint mode (all tracers in one job):
      sbatch --cpus-per-task=48 $script_name -- --joint --kmaxP 0.30 0.25
      sbatch --cpus-per-task=48 $script_name -- --joint --bispec --reparam hybrid
EOF
}

for arg in "$@"; do
        if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
                print_usage
                exit 0
        fi
done

# sbatch forwards the separator "--" to the script; drop it before any further parsing.
user_args=("$@")
if [ "${#user_args[@]}" -gt 0 ] && [ "${user_args[0]}" = "--" ]; then
    user_args=("${user_args[@]:1}")
fi

# Check if joint fitting mode is requested
joint_mode=false
forwarded_user_args=()
for arg in "${user_args[@]}"; do
    if [ "$arg" = "--joint" ]; then
        joint_mode=true
        continue
    fi
    forwarded_user_args+=("$arg")
done

# Set default cpus based on mode if not specified by user
if [ "$joint_mode" = true ]; then
    default_cpus=48
else
    default_cpus=8
fi

# Get actual cpus-per-task from SLURM or use default
slurm_cpus=${SLURM_CPUS_PER_TASK:-$default_cpus}

# In joint mode, skip SLURM_ARRAY_TASK_ID checks
if [ "$joint_mode" = false ]; then
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
            echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with sbatch so the array index is defined."
            echo "Hint: pass model overrides after '--', e.g. sbatch $(basename "$0") -- --kmaxP 0.30 0.25"
            echo "Run '$(basename "$0") --help' for usage examples."
        exit 1
    fi

    if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#tracers[@]}" ]; then
        echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range [0, $((${#tracers[@]} - 1))]."
        exit 1
    fi

    tracer="${tracers[$SLURM_ARRAY_TASK_ID]}"
fi

# If bispectrum is requested at submit time, pass this script's default B settings.
bispec_defaults=()
has_bispec=false
for arg in "${user_args[@]}"; do
    if [ "$arg" = "--bispec" ]; then
        has_bispec=true
        break
    fi
done

if [ "$has_bispec" = true ]; then
    bispec_defaults=(
        --kminB $kminB
        --kmaxB $kmaxB
        --ellB $ellB
    )
fi


# Keep user args from leaking into the sourced environment script.
set --
#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh
export OMP_NUM_THREADS=1

# Build the command based on mode
if [ "$joint_mode" = true ]; then
    # Joint fit mode: pass all tracers as arguments to fit_cubic.py
    srun -n 1 -c "$slurm_cpus" --cpu-bind=cores python -u "$REPO_ROOT/src/fit_cubic.py" \
        --tracer "${tracers[@]}" \
        --hod "$hod" \
        --cosmo "$cosmo" \
        --kminP $kminP \
        --kmaxP $kmaxP \
        --ellP $ellP \
        --freedom "$freedom" \
        --de_model "$de_model" \
        --reparam "$reparam" \
        --counterterm_basis "$counterterm_basis" \
        --outdir "$outdir" \
        --n_live $n_live \
        ${bispec_defaults[@]} "${forwarded_user_args[@]}"
else
    # Array mode: single tracer per task
    srun -n 1 -c "$slurm_cpus" --cpu-bind=cores python -u "$REPO_ROOT/src/fit_cubic.py" \
        --tracer "$tracer" \
        --hod "$hod" \
        --cosmo "$cosmo" \
        --kminP $kminP \
        --kmaxP $kmaxP \
        --ellP $ellP \
        --freedom "$freedom" \
        --de_model "$de_model" \
        --reparam "$reparam" \
        --counterterm_basis "$counterterm_basis" \
        --outdir "$outdir" \
        --n_live $n_live \
        ${bispec_defaults[@]} "${forwarded_user_args[@]}"
fi

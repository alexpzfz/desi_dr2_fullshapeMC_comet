#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=cpu
#SBATCH --job-name=mc2f
#SBATCH --account=desi
#SBATCH --output=/global/u2/a/alexpzfz/desi_dr2_fullshapeMC_comet/outputs/logs/%x_%j.out

# Hardcoded (not derived from BASH_SOURCE): Slurm copies this script into its
# spool dir before running it, so BASH_SOURCE points at the spool copy on the
# compute node, not this file's real location -- that broke src/ resolution.
REPO_ROOT="/global/u2/a/alexpzfz/desi_dr2_fullshapeMC_comet"

# Map each array index to one tracer/z bin.
#zranges=("BGS 0.1 0.4" "LRG 0.4 0.6" "LRG 0.6 0.8" "LRG 0.8 1.1" "ELG_LOP 0.8 1.1" "ELG_LOP 1.1 1.6" "QSO 0.8 2.1")
tracer_labels=("BGS" "LRG1" "LRG2" "LRG3" "ELG1" "ELG2" "QSO")

# Power spectrum settings
dkP="0.005"
kminP="0.02 0.02"
kmaxP="0.35 0.25"
ellP="0 2"
ellwinP="0 2 4"
kwinminP="0.0015 0.0015 0.0015"
kwinmaxP="0.5 0.5 0.5"

# Bisectrum settings
dkB="0.005"
kminB="0.02 0.02"
kmaxB="0.2 0.15"
ellB="000 202"
#ellwinB="000 022 110 112 220 222"
#kwinminB="0.005 0.005 0.005 0.005 0.005 0.005"
#kwinmaxB="0.23 0.23 0.23 0.23 0.23 0.23"
ellwinB="000 022"
kwinminB="0.0015 0.0015"
kwinmaxB="0.3 0.3"


# Other settings
freedom='interm'
free_Mnu=''
region='GCcomb'
de_model='--de_model lambda'
mpc_h=''            # set to '--mpc_h' to run in Mpc/h units (switches reparam to sigma8) instead of Mpc
avirB_free=''        # set to '--avirB_free' to sample avirB freely instead of tying it to avir; only applies with --bispec

print_usage() {
        script_name=$(basename "$0")
        cat <<EOF
Usage:
    sbatch [sbatch options] $script_name -- [fit_cutsky_abacushf.py options]

Important:
    Use '--' to separate sbatch options from python/script options.
    Without '--', sbatch may try to parse your overrides as its own options.

Modes:
    Array mode: Each array task fits one tracer independently
      - Must specify --array and --cpus-per-task in sbatch options
      - tracer_labels=(${tracer_labels[@]}), so --array=0-$((${#tracer_labels[@]} - 1)) covers all of them
    Joint mode: Single job fits all tracers together (use --joint flag)
      - Defaults to $((8 * ${#tracer_labels[@]})) cpus-per-task for all ${#tracer_labels[@]} tracers

Examples:
    Array mode (fits one tracer per task):
      sbatch --array=0-6 --cpus-per-task=8 $script_name -- --kmaxP 0.30 0.20
      sbatch --array=3 --cpus-per-task=8 $script_name -- --bispec --kmaxB 0.18 0.14
      sbatch --array=0-6 --cpus-per-task=8 $script_name -- --mpc_h
      sbatch --array=3 --cpus-per-task=8 $script_name -- --bispec --avirB_free

    Joint mode (all tracers in one job):
      sbatch --cpus-per-task=56 $script_name -- --joint --kmaxP 0.30 0.20
      sbatch --cpus-per-task=56 $script_name -- --joint --bispec
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
    default_cpus=$((8 * ${#tracer_labels[@]}))
else
    default_cpus=8
fi

# Get actual cpus-per-task from SLURM or use default
slurm_cpus=${SLURM_CPUS_PER_TASK:-$default_cpus}

# In joint mode, skip SLURM_ARRAY_TASK_ID checks
if [ "$joint_mode" = false ]; then
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
            echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with sbatch so the array index is defined."
            echo "Hint: pass model overrides after '--', e.g. sbatch $(basename "$0") -- --kmaxP 0.30 0.20"
            echo "Run '$(basename "$0") --help' for usage examples."
        exit 1
    fi

    if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#tracer_labels[@]}" ]; then
        echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range [0, $((${#tracer_labels[@]} - 1))]."
        exit 1
    fi

    tracer_label="${tracer_labels[$SLURM_ARRAY_TASK_ID]}"
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
        --dkB $dkB
        --kminB $kminB
        --kmaxB $kmaxB
        --ellB $ellB
        --ellwinB $ellwinB
        --kwinminB $kwinminB
        --kwinmaxB $kwinmaxB
        $avirB_free
    )
fi


# Keep user args from leaking into the sourced environment script.
set --
#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=1

# Build the command based on mode
if [ "$joint_mode" = true ]; then
    # Joint fit mode: pass all tracers as arguments to single --tracer_label
    tracer_args=(--tracer_label "${tracer_labels[@]}")
    
    srun -n 1 -c "$slurm_cpus" --cpu-bind=cores python -u "$REPO_ROOT/src/fit_cutsky_abacushf.py" \
        "${tracer_args[@]}" \
        --region $region \
        --kminP $kminP \
        --kmaxP $kmaxP \
        --ellP $ellP \
        --ellwinP $ellwinP \
        --kwinminP $kwinminP \
        --kwinmaxP $kwinmaxP \
        --freedom "$freedom" \
        --dkP $dkP \
        $de_model $free_Mnu $mpc_h \
        ${bispec_defaults[@]} "${forwarded_user_args[@]}"
else
    # Array mode: single tracer per task
    srun -n 1 -c "$slurm_cpus" --cpu-bind=cores python -u "$REPO_ROOT/src/fit_cutsky_abacushf.py" \
        --tracer_label "$tracer_label" \
        --region $region \
        --kminP $kminP \
        --kmaxP $kmaxP \
        --ellP $ellP \
        --ellwinP $ellwinP \
        --kwinminP $kwinminP \
        --kwinmaxP $kwinmaxP \
        --freedom "$freedom" \
        --dkP $dkP \
        $de_model $free_Mnu $mpc_h \
        ${bispec_defaults[@]} "${forwarded_user_args[@]}"
fi

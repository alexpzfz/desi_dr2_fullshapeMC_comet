#!/usr/bin/bash
# Single-tracer P(k) fits of the blinded data produced by
# blinding_test/proof_of_concept.ipynb.
#
# The blinded data lives in $DATA_DIR under its own mocktype tag, with the
# covariance and window files symlinked in, so it is a drop-in --data_dir for
# fit_cutsky_abacushf.py -- no other change to the fit is needed.
#
# submit_cutsky_abacushf.sh's array indices are
#   0=BGS  1=LRG1  2=LRG2  3=LRG3  4=ELG1  5=ELG2  6=QSO
# and the blinded set covers 1-6 (the joint chain it came from had no BGS).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BLIND_TAG=blind1
MOCKTYPE="abacus-hf-dr2-v2-altmtl-${BLIND_TAG}"
DATA_DIR="$REPO_ROOT/blinding_test/blinded_data"
OUTDIR="$REPO_ROOT/outputs/chains/blinded_${BLIND_TAG}"

# Must match the settings the blinded files were generated with: the shift is a
# difference of two window-convolved models, so the refit has to convolve on the
# same theory grid. See the BLIND_* settings in the notebook.
dkP="0.005"
kminP="0.02 0.02"
kmaxP="0.35 0.25"
ellP="0 2"
ellwinP="0 2 4"
kwinminP="0.0015 0.0015 0.0015"
kwinmaxP="0.5 0.5 0.5"

print_usage() {
    cat <<USAGE
Usage: $(basename "$0") [-n|--dry-run] [array-spec]

  array-spec   sbatch array indices; default 1-6 (all blinded tracers).
               0=BGS 1=LRG1 2=LRG2 3=LRG3 4=ELG1 5=ELG2 6=QSO
  -n           print the sbatch command instead of submitting it

Examples:
  $(basename "$0")              # submit all six tracers
  $(basename "$0") 1,3          # submit LRG1 and LRG3 only
  $(basename "$0") -n           # show what would be submitted
USAGE
}

dry_run=false
array=""
for arg in "$@"; do
    case "$arg" in
        -h|--help) print_usage; exit 0 ;;
        -n|--dry-run) dry_run=true ;;
        -*) echo "Unknown option: $arg"; print_usage; exit 1 ;;
        *) array="$arg" ;;
    esac
done
array="${array:-1-6}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: no blinded data at $DATA_DIR."
    echo "Run the 'Saving the blinded data' section of blinding_test/proof_of_concept.ipynb first."
    exit 1
fi

mkdir -p "$REPO_ROOT/outputs/logs"
cmd=(sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --array="$array" --time=04:00:00 --cpus-per-task=32
     -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh"
     --mocktype "$MOCKTYPE"
     --data_dir "$DATA_DIR"
     --dkP $dkP
     --kminP $kminP
     --kmaxP $kmaxP
     --ellP $ellP
     --ellwinP $ellwinP
     --kwinminP $kwinminP
     --kwinmaxP $kwinmaxP
     --freedom interm
     --de_model lambda
     --outdir "$OUTDIR"
     --extra "$BLIND_TAG"
     --plot_contours --plot_dir "$OUTDIR")

if [ "$dry_run" = true ]; then
    printf '%q ' "${cmd[@]}"; echo
    exit 0
fi

mkdir -p "$OUTDIR"
"${cmd[@]}"

#!/usr/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO_ROOT/outputs/chains/scale_config"
mkdir -p "$REPO_ROOT/outputs/logs"
de_model='w0wa'
configs=("" "--mpc_h" "--mpc_h --sigma_kind sigma_12" "--mpc_h --reparam jeffreys")
export OMP_NUM_THREADS=1


for config in "${configs[@]}"; do
echo "Submitting job with config: $config"
        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --qos=shared --time=48:00:00 --cpus-per-task=128 -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
         --kmaxP 0.2 0.2 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR \
         --de_model $de_model --zeff_choice "zgeom" \
         --model "EFT" $config \
         --kmaxP_map '{"BGS": [0.15, 0.15]}' --kminP_map '{"BGS": [0.02, 0.05]}' \
         --n_live 8000
done

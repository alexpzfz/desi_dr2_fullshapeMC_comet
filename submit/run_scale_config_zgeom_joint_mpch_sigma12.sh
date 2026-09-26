#!/usr/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO_ROOT/outputs/chains/scale_config"
mkdir -p "$REPO_ROOT/outputs/logs"
de_models=('lambda' 'w0wa')
#units_ops=('' '--mpc_h')

#de_models=('w0wa')
#units_ops=('--mpc_h')
export OMP_NUM_THREADS=1
export COMET_THREADS_PER_WORKER=4

for de_model in "${de_models[@]}"; do
        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --qos=regular --time=48:00:00 --cpus-per-task=256 -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
        --bispec --kmaxP 0.35 0.3 --kmaxB 0.2 0.1 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR \
         --de_model $de_model --mpc_h --extra "bispec_ext" --zeff_choice "zgeom" \
         --kmaxP_map '{"BGS": [0.3, 0.2]}' --kminP_map '{"BGS": [0.02, 0.05]}' --kmaxB_map '{"BGS": [0.1, 0.05]}' \
         --sigma_kind "sigma_12" \
         --n_live 8000
done

export COMET_THREADS_PER_WORKER=1
for de_model in "${de_models[@]}"; do
        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --qos=shared --time=48:00:00 --cpus-per-task=64 -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
         --kmaxP 0.35 0.3 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR \
         --de_model $de_model --mpc_h --zeff_choice "zgeom" \
         --kmaxP_map '{"BGS": [0.3, 0.2]}' --kminP_map '{"BGS": [0.02, 0.05]}' \
         --sigma_kind "sigma_12" \
         --n_live 8000
done

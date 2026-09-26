#!/usr/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO_ROOT/outputs/chains/scale_config"
mkdir -p "$REPO_ROOT/outputs/logs"
de_models=('lambda' 'w0wa')
units_ops=('' '--mpc_h')

#de_models=('w0wa')
#units_ops=('--mpc_h')

for units_op in "${units_ops[@]}"; do
    for de_model in "${de_models[@]}"; do
        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --array=1,2-6 --time=05:00:00 --cpus-per-task=32 -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" --bispec --kmaxP 0.35 0.3 --kmaxB 0.2 0.1 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR --de_model $de_model ${units_op} --extra "bispec_ext" --zeff_choice "zgeom"
        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --array=0 --time=05:00:00 --cpus-per-task=32 -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" --bispec --kminP 0.02 0.05 --kmaxP 0.3 0.2 --kmaxB 0.2 0.1 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR --de_model $de_model ${units_op} --extra "bispec_ext" --zeff_choice "zgeom"
    done
done

#!/usr/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO_ROOT/outputs/chains/eft_config"
mkdir -p "$REPO_ROOT/outputs/logs"
de_model='w0wa'
configs=("" "--mpc_h" "--mpc_h --sigma_kind sigma_12")
eft_configs=("--kmaxP 0.2 0.2" "--kmaxP 0.2 0.2 --free_cnlo" "--kmaxP 0.3 0.25 --free_cnlo" "--kmaxP 0.35 0.3 --free_cnlo")


export OMP_NUM_THREADS=1
DATA_DIR="$REPO_ROOT/data/synth_EFT_cnlo"


for config in "${configs[@]}"; do
for eft_config in "${eft_configs[@]}"; do
echo "Submitting job with config: $config" "and eft_config: $eft_config"
        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --qos=shared --time=48:00:00 --cpus-per-task=40 -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
         $eft_config --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR \
         --de_model $de_model --zeff_choice "zgeom" \
         --model "EFT" $config \
         --mocktype "synth-bestfit" --data_dir "$DATA_DIR" \
         --extra synth_cnlo \
         --n_live 8000
done
done

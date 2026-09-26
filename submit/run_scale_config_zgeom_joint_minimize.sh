#!/usr/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO_ROOT/outputs/chains/scale_config"
mkdir -p "$REPO_ROOT/outputs/logs"
de_models=('w0wa')
units_ops=('' '--mpc_h')

#de_models=('w0wa')
#units_ops=('--mpc_h')

# Minuit is single-process, so there is no worker pool: the only parallelism is
# numba's thread pool inside the bispectrum kernels. Give it all the job's cores.
N_THREADS=8
hesse_op=''       # set to '--hesse' to also run HESSE after MIGRAD
export OMP_NUM_THREADS=1
export COMET_THREADS_PER_WORKER=$N_THREADS

for units_op in "${units_ops[@]}"; do
    for de_model in "${de_models[@]}"; do
        rotstr=''
        minimize_mode='am_then_full'
        if [[ $de_model == 'w0wa' ]]; then
            rotstr='--rotatew0wa'
            if [[ $units_op == '--mpc_h' ]]; then
                minimize_mode='simplex_full'
            fi
        fi
        echo "Submitting job for de_model=$de_model, units_op=$units_op, minimize_mode=$minimize_mode", $rotstr

        sbatch --output="$REPO_ROOT/outputs/logs/%x_%j.out" --qos=shared --time=48:00:00 --cpus-per-task=$N_THREADS --job-name=mc2f_min -- "$REPO_ROOT/submit/submit_cutsky_abacushf.sh" \
         --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
         --bispec --kmaxP 0.35 0.3 --kmaxB 0.2 0.1 --outdir $OUTDIR \
         --de_model $de_model ${units_op} --extra "bispec_ext" --zeff_choice "zgeom" \
         --kmaxP_map '{"BGS": [0.3, 0.2]}' --kminP_map '{"BGS": [0.02, 0.05]}' --kmaxB_map '{"BGS": [0.1, 0.05]}' \
         --minimize ${hesse_op} \
         --minimize_mode $minimize_mode ${rotstr}
    done
done

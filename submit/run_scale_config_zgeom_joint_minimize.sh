#!/usr/bin/bash

OUTDIR=/global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet/outputs/chains/scale_config
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

        sbatch --qos=shared --time=48:00:00 --cpus-per-task=$N_THREADS --job-name=mc2f_min -- submit_cutsky_abacushf.sh \
         --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
         --bispec --kmaxP 0.35 0.3 --kmaxB 0.2 0.1 --outdir $OUTDIR \
         --de_model $de_model ${units_op} --extra "bispec_ext" --zeff_choice "zgeom" \
         --kmaxP_map '{"BGS": [0.3, 0.2]}' --kminP_map '{"BGS": [0.02, 0.05]}' --kmaxB_map '{"BGS": [0.1, 0.05]}' \
         --minimize ${hesse_op} \
         --minimize_mode $minimize_mode ${rotstr}
    done
done

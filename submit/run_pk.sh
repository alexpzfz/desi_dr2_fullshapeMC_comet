#!/usr/bin/bash

OUTDIR=/global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet/outputs/chains/pk_only
#create the output directory if it doesn't exist
mkdir -p $OUTDIR
#de_models=('lambda' 'w0wa')
#units_ops=('' '--mpc_h')

de_models=('lambda')
units_ops=('')

for units_op in "${units_ops[@]}"; do
    for de_model in "${de_models[@]}"; do
        sbatch --array=1,2,3,5 --time=02:00:00 --cpus-per-task=32 -- submit_cutsky_abacushf.sh --kmaxP 0.35 0.25 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR --de_model $de_model ${units_op} --zeff_choice zgeom
        #sbatch --array=0 --time=05:00:00 --cpus-per-task=32 -- submit_cutsky_abacushf.sh --bispec --kminP 0.02 0.05 --kmaxP 0.35 0.3 --kmaxB 0.2 0.1 --outdir $OUTDIR --plot_contours --plot_dir $OUTDIR --de_model $de_model ${units_op}
    done
done

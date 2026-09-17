#!/usr/bin/bash
REPO_ROOT="/global/u2/a/alexpzfz/desi_dr2_fullshapeMC_comet"
OUTDIR=/global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet/outputs/chains/best_fits
LOGDIR=/global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet/outputs/logs/best_fits
#create log dir if it doesn't exist
mkdir -p $LOGDIR
#de_models=('lambda' 'w0wa')
#units_ops=('' '--mpc_h')

de_models=('lambda')
units_ops=('')
tracers=(BGS LRG1 LRG2 LRG3 ELG1 ELG2 QSO)
export OMP_NUM_THREADS=1

for tracer in "${tracers[@]}"; do
for units_op in "${units_ops[@]}"; do
    for de_model in "${de_models[@]}"; do
srun -n 1 --cpus-per-task=1 \
     -o "$LOGDIR/${tracer}_${de_model}.log" \
     -e "$LOGDIR/${tracer}_${de_model}.log" \
     python -u "$REPO_ROOT/src/fit_cutsky_abacushf.py" \
     --tracer_label $tracer \
     --bispec \
     --kmaxP 0.35 0.3 \
     --kmaxB 0.2 0.1 \
     --outdir $OUTDIR \
     --de_model $de_model \
     ${units_op} \
     --minimize &
    done
done
done
wait

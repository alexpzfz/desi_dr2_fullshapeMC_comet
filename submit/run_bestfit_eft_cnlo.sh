#!/usr/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO_ROOT/outputs/chains/best_fits_eft_cnlo"
LOGDIR="$REPO_ROOT/outputs/logs/best_fits_eft_cnlo"
#create log dir if it doesn't exist
mkdir -p $LOGDIR
#de_models=('lambda' 'w0wa')
#units_ops=('' '--mpc_h')

de_model='lambda'
tracers=(BGS LRG1 LRG2 LRG3 ELG1 ELG2 QSO)
export OMP_NUM_THREADS=1

kmaxP="0.35 0.3"


for tracer in "${tracers[@]}"; do
    if [[ $tracer == "BGS" ]]; then
        kminP="0.02 0.05"
    else
        kminP="0.02 0.02"
    fi
srun -n 1 --cpus-per-task=1 \
     -o "$LOGDIR/${tracer}_${de_model}.log" \
     -e "$LOGDIR/${tracer}_${de_model}.log" \
     python -u "$REPO_ROOT/src/fit_cutsky_abacushf.py" \
     --tracer_label $tracer \
     --kmaxP $kmaxP \
     --kminP $kminP \
     --outdir $OUTDIR \
     --de_model $de_model \
     --model EFT \
     --free_cnlo \
     --zeff_choice zgeom\
     --mpc_h \
     --fix_cosmo \
     --minimize &
    done
wait

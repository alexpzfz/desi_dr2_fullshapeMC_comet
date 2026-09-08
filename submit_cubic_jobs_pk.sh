#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=/global/homes/a/alexpzfz/comet-emu:$PYTHONPATH
export OMP_NUM_THREADS=1

array=0-2 #LRG, ELG, QSO
cpus_per_task=32
kmaxPcases=("0.35 0.3" "0.4 0.35")
# kmaxBcases=("0.2 0.15" "0.2 0.1")
hodcases=("fiducial" "alternative")
cosmocases=("c000" "c001" "c002" "c004")
demodelcases=("lambda" "w0wa")

for kmaxP in "${kmaxPcases[@]}"; do
    for de_model in "${demodelcases[@]}"; do
        for cosmo in "${cosmocases[@]}"; do
            if [[ "$cosmo" == "c000" ]]; then
                for hod in "${hodcases[@]}"; do
                    sbatch --array="$array" --cpus-per-task="$cpus_per_task" \
                            submit_cubic.sh -- --kmaxP $kmaxP --de_model "$de_model" --cosmo "$cosmo" --hod "$hod"
                done
            else
                sbatch --array="$array" --cpus-per-task="$cpus_per_task" \
                    submit_cubic.sh -- --kmaxP $kmaxP --de_model "$de_model" --cosmo "$cosmo"
            fi
        done
    done
done
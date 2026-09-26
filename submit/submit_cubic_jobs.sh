#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# NERSC-only software environment.
if [ -n "${NERSC_HOST:-}" ]; then source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main; fi
export OMP_NUM_THREADS=1

array=0-2 #LRG, ELG, QSO
cpus_per_task=16
kmaxPcases=("0.35 0.25" "0.35 0.3")
kmaxBcases=("0.2 0.15" "0.2 0.1")
hodcases=("fiducial" "alternative")
cosmocases=("c000" "c001" "c002" "c004")
demodelcases=("lambda" "w0wa")

for kmaxP in "${kmaxPcases[@]}"; do
    for kmaxB in "${kmaxBcases[@]}"; do
        for de_model in "${demodelcases[@]}"; do
            for cosmo in "${cosmocases[@]}"; do
                if [[ "$cosmo" == "c000" ]]; then
                    for hod in "${hodcases[@]}"; do
                        sbatch --array="$array" --cpus-per-task="$cpus_per_task" \
                            "$SCRIPT_DIR/submit_cubic.sh" -- --kmaxP $kmaxP --kmaxB $kmaxB --de_model "$de_model" --cosmo "$cosmo" --hod "$hod" --bispec
                    done
                else
                    sbatch --array="$array" --cpus-per-task="$cpus_per_task" \
                        "$SCRIPT_DIR/submit_cubic.sh" -- --kmaxP $kmaxP --kmaxB $kmaxB --de_model "$de_model" --cosmo "$cosmo" --bispec
                fi
            done
        done
    done
done
#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
#SBATCH --job-name=mc2f
#SBATCH --account=desi
#SBATCH --array=0-5

dks=(0.005 0.01)
regions=(SGC NGC GCcomb)


de_model='--de_model lambda'
force_nmocks=''


for arg in "$@"; do
    if [[ "$arg" == "--de_model" ]]; then
        shift
        de_model="--de_model $1"
    elif [[ "$arg" == "--force_nmocks" ]]; then
        force_nmocks='--force_nmocks 1000'  
        echo "Running with force_nmocks option"
    fi
done

j=${SLURM_ARRAY_TASK_ID}
dk=${dks[$((j/3))]}
region=${regions[$((j%3))]}

set --

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh
export OMP_NUM_THREADS=1
srun -n 1 -c 8 --cpu-bind=cores python -u fit_cutsky_secondgen.py \
    --tracer LRG \
    --zrange 0.4 0.6 \
    --region $region \
    --kminP 0.01 0.01 \
    --kmaxP 0.35 0.25 \
    --ellwinP 0 2 4 \
    --kwinPmin 0.0 0.0 0.0 \
    --kwinPmax 0.4 0.4 0.4 \
    --freedom 'adhoc' \
    --dkP $dk \
    $de_model $force_nmocks

#!/bin/bash --login
#SBATCH --job-name=reservoir_compute
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --mail-user=tlarsen@mathematics.byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=1

# Production defaults. Test submits override --time/--mem on the sbatch line.
# n=500 sits between the 20G (n=400) and 30G (n=600) comments below.
# 10G for 200, 20G for 400, 30G for 600, 40G for 800, 50G not enough for 1000

set -euo pipefail
mkdir -p logs

echo "Running SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} submit_dir=${SLURM_SUBMIT_DIR:-}"
echo "NETWORK_TYPE=$NETWORK_TYPE"
echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
echo "PARAM_SET=$PARAM_SET"
echo "PARAM_NAME=$PARAM_NAME"
echo "PARAM_VALUE=$PARAM_VALUE"
echo "TF_SECONDS=${TF_SECONDS:-7200} CHUNK_SIZE=${CHUNK_SIZE:-10}"

module load miniforge3
mamba activate reservoir

which python
python --version

CHUNK_SIZE=${CHUNK_SIZE:-10}
START=$((SLURM_ARRAY_TASK_ID))
END=$((START + CHUNK_SIZE - 1))

# Last array task can overshoot when the grid size is not a multiple of CHUNK_SIZE.
json_path="./utils/rho_p_thin_sets/${RHO_P_THIN_SET}.json"
n_cells=$(( $(jq '.rho | length' "$json_path") * $(jq '.p_thin | length' "$json_path") ))
if (( START >= n_cells )); then
    echo "Nothing to do: START=$START n_cells=$n_cells"
    exit 0
fi
if (( END >= n_cells )); then
    END=$((n_cells - 1))
fi

for (( i=START; i<=END; i++ )); do
    echo "ID_TO_PROCESS=$i"
    export ID_TO_PROCESS=$i
    python main.py -n_type "$NETWORK_TYPE" -r "$RHO_P_THIN_SET" -p "$PARAM_VALUE" -p_name "$PARAM_NAME" -p_set "$PARAM_SET"
done
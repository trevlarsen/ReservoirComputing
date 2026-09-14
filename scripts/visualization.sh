#!/bin/bash --login

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2048M
#SBATCH -J "Reservoir_Visualization"
#SBATCH --output=logs/visualization_%j.log
#SBATCH --mail-user=tlarsen@mathematics.byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

set -euo pipefail
mkdir -p logs

module load miniforge3
mamba activate reservoir

python utils/visualization.py -n_type "$NETWORK_TYPE" -r "$RHO_P_THIN_SET" -p "$PARAM_VALUE" -p_name "$PARAM_NAME" -p_set "$PARAM_SET"

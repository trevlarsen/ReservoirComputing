#!/bin/bash

# Usage:
#   bash scripts/submit.sh -s    # full search from scripts/vars.txt (~24h)
#   bash scripts/submit.sh -m    # medium search (~4h wall, coarser grid / shorter draws)
#   bash scripts/submit.sh -t    # short smoke test
#   bash scripts/submit.sh -v    # visualization; set VARS_FILE=... to match the campaign
#
# Run from the Thinned_Rescomp repo root so relative paths resolve.

submit_search() {
    local vars_file=$1
    local chunk=${CHUNK_SIZE:-10}
    shift || true

    echo "Reading campaigns from $vars_file (CHUNK_SIZE=$chunk TF_SECONDS=${TF_SECONDS:-7200})"
    while IFS=' ' read -r NETWORK_TYPE RHO_P_THIN_SET PARAM_SET PARAM_NAME PARAM_VALUE; do
        [[ -z "${NETWORK_TYPE:-}" || "$NETWORK_TYPE" =~ ^# ]] && continue

        echo "NETWORK_TYPE=$NETWORK_TYPE"
        echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
        echo "PARAM_SET=$PARAM_SET"
        echo "PARAM_NAME=$PARAM_NAME"
        echo "PARAM_VALUE=$PARAM_VALUE"
        echo "--------------------------"

        json_path="./utils/rho_p_thin_sets/$RHO_P_THIN_SET.json"
        if [[ ! -f "$json_path" ]]; then
            echo "Error: $RHO_P_THIN_SET json file not found at $json_path"
            continue
        fi

        num_rho=$(jq '.rho | length' "$json_path")
        num_pthin=$(jq '.p_thin | length' "$json_path")

        if [[ "$num_rho" -eq 0 || "$num_pthin" -eq 0 ]]; then
            echo "Error: rho or p_thin arrays are empty in $json_path"
            continue
        fi

        total_jobs=$((num_rho * num_pthin))

        echo "Found $num_rho rho values and $num_pthin p_thin values."
        echo "Submitting array 0-$((total_jobs - 1)):${chunk} ($total_jobs grid points, $chunk per Slurm task)."

        sbatch \
            --array=0-$((total_jobs - 1)):"$chunk" \
            --export=ALL,NETWORK_TYPE="$NETWORK_TYPE",RHO_P_THIN_SET="$RHO_P_THIN_SET",PARAM_SET="$PARAM_SET",PARAM_NAME="$PARAM_NAME",PARAM_VALUE="$PARAM_VALUE",CHUNK_SIZE="$chunk",TF_SECONDS="${TF_SECONDS:-7200}" \
            "$@" \
            scripts/simulations_array.sh

        if [[ "${SUBMIT_SLEEP:-60}" -gt 0 ]]; then
            sleep "$SUBMIT_SLEEP"
        fi
    done < "$vars_file"
}

while getopts "smtv" opt; do
    case $opt in
        s)
            echo "Running search ..."
            submit_search scripts/vars.txt
            ;;

        m)
            echo "Running medium search ..."
            # Same experiment as -s (directed_erdos, n=500, alpha=4e-9).
            # Grid: all 16 production rhos, p_thin every 0.05 plus 0.99 → 336 cells
            #   vs 1600 on comprehensive_set_IV.
            # Chunk 10 → 34 array tasks. 20 min of draws per cell × 10 ≈ 3.3h of
            # science per task; 4h walltime leaves headroom. Smoke test got ~10
            # draws in 5 min, so this is ~40 draws/cell vs ~240 on the 2h full run.
            export CHUNK_SIZE=10
            export TF_SECONDS=1200
            export SUBMIT_SLEEP=0
            mkdir -p logs
            submit_search scripts/vars_medium.txt --time=04:00:00 --mem=20G --job-name=rescomp_medium
            ;;

        t)
            echo "Running smoke test ..."
            # 10 (rho, p_thin) pairs, 2 per array task → 5 Slurm tasks.
            # Each pair runs draws for 5 minutes, so a task is ~10 min of science
            # plus conda startup. Walltime 1 hour leaves headroom.
            export CHUNK_SIZE=2
            export TF_SECONDS=300
            export SUBMIT_SLEEP=0
            mkdir -p logs
            # n=500 diversity metrics build (T, n, n) arrays ~8GB. 2G OOMs; 20G matches the original comments.
            submit_search scripts/vars_test.txt --time=00:20:00 --mem=20G --job-name=rescomp_test
            ;;

        v)
            echo "Running visualization ..."
            while IFS=' ' read -r NETWORK_TYPE RHO_P_THIN_SET PARAM_SET PARAM_NAME PARAM_VALUE; do
                echo "NETWORK_TYPE=$NETWORK_TYPE"
                echo "RHO_P_THIN_SET=$RHO_P_THIN_SET"
                echo "PARAM_SET=$PARAM_SET"
                echo "PARAM_NAME=$PARAM_NAME"
                echo "PARAM_VALUE=$PARAM_VALUE"
                echo "--------------------------"

                echo "Submitting job array"  
                sbatch --export=ALL,NETWORK_TYPE=$NETWORK_TYPE,RHO_P_THIN_SET=$RHO_P_THIN_SET,PARAM_SET=$PARAM_SET,PARAM_NAME=$PARAM_NAME,PARAM_VALUE=$PARAM_VALUE scripts/visualization.sh
                sleep "${SUBMIT_SLEEP:-0}"
            done < "${VARS_FILE:-scripts/vars.txt}"
            ;;

        *)
            echo "Usage: $0 [-s | -m | -t | -v]"
            exit 1
            ;;
    esac
done


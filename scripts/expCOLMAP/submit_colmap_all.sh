#!/bin/bash
# Submit one COLMAP sweep job per CO3D category (6 jobs total).
# All jobs run in parallel, each requesting 1 GPU on a100.
#
# Usage:
#   cd /home/asds/project_Hayk_Minasyan
#   bash scripts/expCOLMAP/submit_colmap_all.sh

cd /home/asds/project_Hayk_Minasyan
mkdir -p logs

declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

echo "Submitting COLMAP sweep jobs for 6 categories ..."
echo ""

for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"

    JOB_ID=$(CATEGORY="$CATEGORY" SEQ_ID="$SEQ_ID" GPU_IDX=0 \
        sbatch \
            --job-name="colmap_${CATEGORY}" \
            scripts/expCOLMAP/run_colmap_sweep.sh \
        | awk '{print $NF}')

    echo "  Submitted: ${CATEGORY}/${SEQ_ID}  →  job ${JOB_ID}"
done

echo ""
echo "Monitor:   squeue --me"
echo "Logs:      logs/colmap_colmap_<category>_<jobid>.log"
echo ""
echo "When all 6 jobs finish, generate the plot:"
echo "  cd /home/asds/project_Hayk_Minasyan"
echo "  source /home/asds/miniforge3/etc/profile.d/conda.sh"
echo "  conda activate co3d_env"
echo "  python scripts/expCOLMAP/plot_colmap_results.py"

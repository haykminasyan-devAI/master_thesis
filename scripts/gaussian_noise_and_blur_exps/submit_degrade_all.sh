#!/bin/bash
# Submit all blur + Gaussian-noise degradation experiments across 10 CO3D categories.
# 6 degradation settings × 10 sequences = 60 SLURM jobs.
#
# Run from the project root:
#   bash scripts/gaussian_noise_and_blur_exps/submit_degrade_all.sh

set -euo pipefail
cd /home/asds/project_Hayk_Minasyan

# ── 6 sequences (category / seq_id) ──────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

# ── 6 degradation settings ────────────────────────────────────────────────────
BLUR_SIGMAS=(1 3 5)
NOISE_STDS=(10 25 50)

TOTAL=0

for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"

    # ── blur experiments ──────────────────────────────────────────────────────
    for SIGMA in "${BLUR_SIGMAS[@]}"; do
        JOB=$(sbatch \
            --job-name="blur_s${SIGMA}_${CATEGORY:0:4}" \
            --export=ALL,DEGRADE_MODE=blur,DEGRADE_PARAM=${SIGMA},CATEGORY=${CATEGORY},SEQ_ID=${SEQ_ID} \
            scripts/gaussian_noise_and_blur_exps/run_exp_degrade.sh)
        echo "Submitted $JOB  →  blur σ=${SIGMA}  ${CATEGORY}/${SEQ_ID}"
        TOTAL=$((TOTAL + 1))
    done

    # ── noise experiments ─────────────────────────────────────────────────────
    for STD in "${NOISE_STDS[@]}"; do
        JOB=$(sbatch \
            --job-name="noise_s${STD}_${CATEGORY:0:4}" \
            --export=ALL,DEGRADE_MODE=noise,DEGRADE_PARAM=${STD},CATEGORY=${CATEGORY},SEQ_ID=${SEQ_ID} \
            scripts/gaussian_noise_and_blur_exps/run_exp_degrade.sh)
        echo "Submitted $JOB  →  noise σ=${STD}  ${CATEGORY}/${SEQ_ID}"
        TOTAL=$((TOTAL + 1))
    done
done

echo ""
echo "Submitted ${TOTAL} jobs total (6 categories × 6 settings)."
echo "Monitor with: squeue --me"
echo ""
echo "Once all jobs finish, generate the final averaged plots with:"
echo "  python scripts/gaussian_noise_and_blur_exps/plot_degrade_multi_seq.py \\"
echo "      --results_root outputs/dust3r/degrade_multi"

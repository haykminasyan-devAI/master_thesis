#!/usr/bin/env bash
# Gradio DUSt3R demo: one GPU via Slurm (ASDS).
#
# A) Batch (same as before):
#      cd /home/asds/project_Hayk_Minasyan
#      sbatch interactive_demo/submit_demo_slurm_asds.sh
#      tail -f logs/dust3r_demo_<JOBID>.log
#
# B) Interactive one GPU, then run the same Python command:
#      salloc --partition=a100 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=4:00:00
#      bash interactive_demo/run_demo_on_node.sh
#
# Browser: from your laptop, tunnel to the compute node on port 7860, e.g.:
#      ssh -L 7860:dgx:7860 asds@<login_host>
#      open http://127.0.0.1:7860

#SBATCH --job-name=dust3r_demo
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/dust3r_demo_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/dust3r_demo_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "=== DUSt3R demo job ${SLURM_JOB_ID:-?} ==="
echo "Node: $(hostname -s)  ($(hostname -f 2>/dev/null || true))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date -Is)"
echo ""

bash "${PROJECT_DIR}/interactive_demo/run_demo_on_node.sh"

echo "End: $(date -Is)"

#!/bin/bash
#SBATCH --job-name=noise_s10
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/noise_s10_%j.log
#SBATCH --error=logs/noise_s10_%j.err

# Run only σ=10 (degrade + DUSt3R) while the full noise sweep or other jobs run elsewhere.
export NOISE_STDS_EXPORT="10"
exec bash "$(dirname "$0")/run_noise_exp.sh"

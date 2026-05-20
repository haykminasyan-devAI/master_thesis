#!/usr/bin/env bash
# =============================================================================
# STEP 1 — Run this ON THE YSU CLUSTER (login node), from your project clone:
#   cd ~/project_Hayk_Minasyan
#   bash evaluation_blur_and_noise/check_eval6seq_ysu_prereqs.sh
#
# It prints what exists vs what the 6-seq blur+noise Chamfer eval needs, then
# suggests rsync commands for anything missing (same defaults as
# run_eval_6seq_blur_noise_chamfer_ysu.sh).
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat}"
NOISE_ROOT="${NOISE_ROOT:-/mnt/weka/hminasyan/outputs/noisy_frames_10cat}"
BLUR_ROOT="${BLUR_ROOT:-/mnt/weka/hminasyan/outputs/degraded_frames_10cat}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
BLUR_FINETUNED_CKPT="${BLUR_FINETUNED_CKPT:-/mnt/weka/hminasyan/finetune_blur_runs/deblurdinat_dust3r_asds_224_joint_5_10_20_30_from_dust3r/joint_sigmas_5_10_20_30/checkpoint-best-val.pth}"
NOISE_FINETUNED_CKPT="${NOISE_FINETUNED_CKPT:-/mnt/weka/hminasyan/finetune_noise_runs/uformer_dust3r_asds_224_recon_l05_lr2e4_wu3_wd02_random50/joint_sigmas_30_50_70/checkpoint-best-val.pth}"
DEBLURDINAT_REPO="${DEBLURDINAT_REPO:-${PROJECT_DIR}/DeblurDiNAT}"
DEBLURDINAT_PRETRAINED="${DEBLURDINAT_PRETRAINED:-${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
UFORMER_REPO="${UFORMER_REPO:-${PROJECT_DIR}/Uformer}"
UFORMER_PRETRAINED="${UFORMER_PRETRAINED:-${UFORMER_REPO}/logs/denoising/SIDD/Uformer_B/models/model_best.pth}"

SAMPLE_CAT=teddybear
SAMPLE_SEQ=101_11758_21048

# shellcheck disable=SC2206
MISSING=()

note_file() {
  local label="$1" path="$2"
  if [[ -f "$path" ]]; then
    local sz
    sz="$(du -h "$path" 2>/dev/null | cut -f1)"
    echo "  [OK]  $label"
    echo "        $path  ($sz)"
  else
    echo "  [!!]  MISSING  $label"
    echo "        $path"
    MISSING+=("file|$path|$label")
  fi
}

note_dir() {
  local label="$1" path="$2"
  if [[ -d "$path" ]]; then
    local sz
    sz="$(du -sh "$path" 2>/dev/null | cut -f1)"
    echo "  [OK]  $label"
    echo "        $path  ($sz)"
  else
    echo "  [!!]  MISSING  $label"
    echo "        $path"
    MISSING+=("dir|$path|$label")
  fi
}

echo ""
echo "=============================================================================="
echo "  YSU inventory — eval6seq blur + noise Chamfer"
echo "  $(date -u 2>/dev/null || date) | $(hostname)"
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "=============================================================================="
echo ""

echo "--- A) Code & launchers (under \$PROJECT_DIR) ---"
note_file "run_eval_6seq_blur_noise_chamfer.sh" "${SCRIPT_DIR}/run_eval_6seq_blur_noise_chamfer.sh"
note_file "run_eval_6seq_blur_noise_chamfer_ysu.sh" "${SCRIPT_DIR}/run_eval_6seq_blur_noise_chamfer_ysu.sh"
note_file "deblurdinat_run_inference.py" "${SCRIPT_DIR}/deblurdinat_run_inference.py"
note_file "uformer_run_inference.py" "${SCRIPT_DIR}/uformer_run_inference.py"
note_dir  "dust3r" "${PROJECT_DIR}/dust3r"
note_dir  "DeblurDiNAT" "${DEBLURDINAT_REPO}"
note_dir  "Uformer" "${UFORMER_REPO}"
echo ""

echo "--- B) Checkpoints (Weka + project paths used by run_eval_6seq_*_ysu.sh) ---"
note_file "DUSt3R ViT-L 224" "$DUST3R_CKPT"
note_file "DeblurDiNAT pretrained (SIDD-style)" "$DEBLURDINAT_PRETRAINED"
note_file "Uformer SIDD pretrained" "$UFORMER_PRETRAINED"
note_file "Deblur finetuned (joint sigmas 5–30)" "$BLUR_FINETUNED_CKPT"
note_file "Noise finetuned (Uformer+DUSt3R)" "$NOISE_FINETUNED_CKPT"
echo ""

echo "--- C) Data roots (large) ---"
note_dir "CO3D processed (GT pointclouds + layout)" "$CO3D_ROOT"
note_dir "Degraded frames (blur)" "$BLUR_ROOT"
note_dir "Noisy frames" "$NOISE_ROOT"
echo ""

echo "--- D) Spot-check (one sequence: ${SAMPLE_CAT} / ${SAMPLE_SEQ}) ---"
note_file "GT pointcloud" "${CO3D_ROOT}/${SAMPLE_CAT}/${SAMPLE_SEQ}/pointcloud.ply"
note_dir  "Blur dir sigma=5" "${BLUR_ROOT}/${SAMPLE_CAT}/${SAMPLE_SEQ}/blur_s5"
note_dir  "Noise dir sigma=30 (finetune sigmas)" "${NOISE_ROOT}/${SAMPLE_CAT}/${SAMPLE_SEQ}/noise_s30"
echo ""

echo "--- E) Python (Weka venv used by YSU launcher) ---"
if [[ -x "/mnt/weka/hminasyan/co3d_env/bin/python" ]]; then
  echo "  [OK]  co3d_env python"
  echo "        /mnt/weka/hminasyan/co3d_env/bin/python"
  "/mnt/weka/hminasyan/co3d_env/bin/python" -V 2>&1 | sed 's/^/        /'
else
  echo "  [!!]  MISSING  Weka co3d_env python"
  echo "        /mnt/weka/hminasyan/co3d_env/bin/python"
  MISSING+=("file|/mnt/weka/hminasyan/co3d_env/bin/python|co3d_env python")
fi
echo ""

echo "=============================================================================="
if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "  Summary: all checked paths are present."
  echo "  Next: sbatch evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer_ysu.sh"
else
  echo "  Summary: ${#MISSING[@]} item(s) missing — see transfer commands below."
fi
echo "=============================================================================="
echo ""

echo "------------------------------------------------------------------------------"
echo "  STEP 2 — If something is missing, transfer FROM your DGX (asds) machine"
echo "  using evaluation_blur_and_noise/transfer_eval6seq_asds_to_ysu.sh"
echo "  (run there with DRY_RUN=1 first), or run the rsync lines by hand."
echo "------------------------------------------------------------------------------"
echo ""

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "Missing items (paths):"
  for ent in "${MISSING[@]}"; do
    IFS='|' read -r _k path label <<<"$ent"
    echo "  - [$label] $path"
  done
  echo ""
  echo "Full bundle from DGX (edit SRC_ROOT / YSU_* in the script if needed):"
  echo "  On asds:  cd ~/project_Hayk_Minasyan"
  echo "            DRY_RUN=1 bash evaluation_blur_and_noise/transfer_eval6seq_asds_to_ysu.sh"
  echo "            bash evaluation_blur_and_noise/transfer_eval6seq_asds_to_ysu.sh"
  echo ""
  echo "That script copies: evaluation_blur_and_noise, dust3r, DeblurDiNAT, Uformer,"
  echo "checkpoints + finetune_*_runs + co3d_processed_10cat + degraded/noisy frames → YSU Weka paths."
fi

echo ""
echo "Done."
echo ""

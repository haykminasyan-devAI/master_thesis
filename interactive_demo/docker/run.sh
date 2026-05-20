#!/bin/bash
#
# Build & launch the interactive DUSt3R + finetuned demo in Docker.
#
# Required env vars OR command-line flags (flags win):
#   --project-dir       Path to the project_Hayk_Minasyan repo on the host
#                       (defaults to one level above this script).
#   --deblurdinat-dir   Path to the DeblurDiNAT repo on the host.
#   --checkpoints-dir   Host directory containing
#                         DUSt3R_ViTLarge_BaseDecoder_224_linear.pth
#                         joint_sigmas_5_10_20_30/checkpoint-best-val.pth
#                         joint_sigmas_5_10_20_30_50/checkpoint-best-val.pth
#   --port              Host port to expose (default 7860).
#   --image-size        224 (default) or 512.
#
# Example:
#   bash run.sh \
#     --project-dir /home/asds/project_Hayk_Minasyan \
#     --deblurdinat-dir /home/asds/DeblurDiNAT \
#     --checkpoints-dir /mnt/weka/hminasyan/demo_ckpts \
#     --port 7860

set -eu
cd "$(dirname "$0")"

ACTION="up"
if [[ "${1:-}" == "down" ]]; then
    ACTION="down"
    shift
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd .. && cd .. && pwd)}"
DEBLURDINAT_DIR="${DEBLURDINAT_DIR:-}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
HOST_PORT="${HOST_PORT:-7860}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir)      PROJECT_DIR="$2"; shift 2 ;;
        --deblurdinat-dir)  DEBLURDINAT_DIR="$2"; shift 2 ;;
        --checkpoints-dir)  CHECKPOINTS_DIR="$2"; shift 2 ;;
        --port)             HOST_PORT="$2"; shift 2 ;;
        --image-size)       IMAGE_SIZE="$2"; shift 2 ;;
        --hf-cache-dir)     HF_CACHE_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Same layout as training scripts / README: DeblurDiNAT under the project, demo ckpts under interactive_demo.
# Defaults apply to both `up` and `down` (compose needs non-empty volume sources for `down` too).
DEBLURDINAT_DIR="${DEBLURDINAT_DIR:-${PROJECT_DIR}/DeblurDiNAT}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_DIR}/interactive_demo/demo_ckpts}"

if [[ "${ACTION}" == "up" ]]; then
    if [[ ! -d "${DEBLURDINAT_DIR}" ]]; then
        echo "ERROR: DeblurDiNAT not found at: ${DEBLURDINAT_DIR}" >&2
        echo "       Clone it there or pass: --deblurdinat-dir /path/to/DeblurDiNAT" >&2
        exit 1
    fi
    if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
        echo "ERROR: checkpoints directory not found at: ${CHECKPOINTS_DIR}" >&2
        echo "       Put the .pth files there or pass: --checkpoints-dir /path/to/ckpts" >&2
        exit 1
    fi
fi

if command -v docker-compose &>/dev/null; then
    DCOMP="docker-compose"
elif docker compose version &>/dev/null 2>&1; then
    DCOMP="docker compose"
else
    echo "ERROR: docker-compose / 'docker compose' not found." >&2
    exit 1
fi

export PROJECT_DIR DEBLURDINAT_DIR CHECKPOINTS_DIR HOST_PORT IMAGE_SIZE HF_CACHE_DIR

echo "[run.sh] PROJECT_DIR     = ${PROJECT_DIR}"
echo "[run.sh] DEBLURDINAT_DIR = ${DEBLURDINAT_DIR}"
echo "[run.sh] CHECKPOINTS_DIR = ${CHECKPOINTS_DIR}"
echo "[run.sh] HOST_PORT       = ${HOST_PORT}"
echo "[run.sh] IMAGE_SIZE      = ${IMAGE_SIZE}"

mkdir -p "${HF_CACHE_DIR}"

if [[ "${ACTION}" == "down" ]]; then
    ${DCOMP} -f docker-compose-cuda.yml down
    exit 0
fi

${DCOMP} -f docker-compose-cuda.yml up --build

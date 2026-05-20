#!/bin/bash
set -eu

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/project_Hayk_Minasyan}
DEVICE=${DEVICE:-cuda}
SERVER_PORT=${SERVER_PORT:-7860}

DUST3R_CKPT=${DUST3R_CKPT:-/workspace/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
# KD Restormer student; fallback to legacy FT_5_10_20_30 env name
KD_RESTORMER_STUDENT_CKPT=${KD_RESTORMER_STUDENT_CKPT:-${FT_5_10_20_30:-/workspace/checkpoints/kd_restormer_student_best.pth}}
KD20_CKPT=${KD20_CKPT:-}
KD50_CKPT=${KD50_CKPT:-}
IMAGE_SIZE=${IMAGE_SIZE:-512}

cd "${PROJECT_ROOT}"

# Compile RoPE CUDA kernels once if not already built.
CUROPE=${PROJECT_ROOT}/dust3r/croco/models/curope
if [ -d "${CUROPE}" ] && ! ls "${CUROPE}"/curope*.so >/dev/null 2>&1; then
    echo "[entrypoint] compiling RoPE CUDA kernels..."
    (cd "${CUROPE}" && python setup.py build_ext --inplace) || \
        echo "[entrypoint] RoPE compile failed, will fall back to pure-pytorch path"
fi

ARGS=(
    --image_size "${IMAGE_SIZE}"
    --device "${DEVICE}"
    --server_port "${SERVER_PORT}"
    --local_network
)

[ -f "${DUST3R_CKPT}" ]                  && ARGS+=( --dust3r_ckpt "${DUST3R_CKPT}" )
[ -f "${KD_RESTORMER_STUDENT_CKPT}" ]   && ARGS+=( --kd_restormer_student_ckpt "${KD_RESTORMER_STUDENT_CKPT}" )
[ -f "${KD20_CKPT}" ]                   && ARGS+=( --kd20_ckpt "${KD20_CKPT}" )
[ -f "${KD50_CKPT}" ]                   && ARGS+=( --kd50_ckpt "${KD50_CKPT}" )

echo "[entrypoint] launching:"
echo "  python interactive_demo/demo_finetuned.py ${ARGS[*]} $*"

exec python interactive_demo/demo_finetuned.py "${ARGS[@]}" "$@"

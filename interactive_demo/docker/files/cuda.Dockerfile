FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL description="Interactive demo: DUSt3R + KD Restormer / encoder LoRA (CUDA)"
ARG DEBIAN_FRONTEND=noninteractive

ENV DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/workspace/project_Hayk_Minasyan \
    DUST3R_CKPT=/workspace/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
    KD_RESTORMER_STUDENT_CKPT=/workspace/checkpoints/kd_restormer_student_best.pth \
    KD20_CKPT= \
    KD50_CKPT= \
    SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1 \
        ca-certificates \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Project source is bind-mounted at runtime via docker-compose.
# We still install dependencies into the image so cold start is fast.
COPY requirements-demo.txt /tmp/requirements-demo.txt
# NATTEN publishes torch/CUDA-specific wheels on whl.natten.org; PyPI "natten" can resolve to
# a build that expects newer PyTorch APIs (e.g. torch.amp.custom_bwd) than the base image provides.
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements-demo.txt --find-links https://whl.natten.org

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]

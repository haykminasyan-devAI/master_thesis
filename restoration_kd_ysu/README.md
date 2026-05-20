# KD Front-End for DUSt3R (YSU)

This folder implements a lightweight restoration front-end trained by knowledge distillation from frozen Restormer.

## Objective

Robustness to:

- motion blur (linear horizontal PSF, 25x25)
- defocus blur (disk PSF radius 7)

without changing DUSt3R itself.

## Training design

- Teacher: frozen Restormer
  - motion branch: `motion_deblurring.pth`
  - defocus branch: `single_image_defocus_deblurring.pth`
- Student: lightweight encoder-decoder + 2 transformer bottleneck blocks
- Blur synthesis per frame:
  - 50% motion blur
  - 50% defocus blur
- Loss:
  - `L_total = 1.0 * L1(student, teacher) + 0.2 * (1 - SSIM(student, teacher))`
- Optimizer: AdamW
  - betas: `(0.9, 0.999)`
  - weight decay: `1e-4`
- LR: `3e-4` cosine to `1e-6`
- DDP defaults: 2 GPUs, per-GPU batch = 4 (total 8), full images (no patching)

## Files

- `train_kd_restormer_frontend.py` — training script
- `setup_restormer_teacher_ysu.sh` — one-time setup on YSU
- `submit_kd_restormer_2gpu.slurm.sh` — Slurm submission script

## ASDS -> YSU sync

From ASDS:

```bash
cd ~/project_Hayk_Minasyan
rsync -avz restoration_kd_ysu/ hminasyan@cluster.ysu.am:~/project_Hayk_Minasyan/restoration_kd_ysu/
```

## Run on YSU

```bash
cd ~/project_Hayk_Minasyan
bash restoration_kd_ysu/setup_restormer_teacher_ysu.sh
sbatch restoration_kd_ysu/submit_kd_restormer_2gpu.slurm.sh
```

## Monitor

```bash
squeue --me
tail -f ~/project_Hayk_Minasyan/logs/kd_restormer_2gpu_<JOBID>.log
tail -f ~/project_Hayk_Minasyan/logs/kd_restormer_2gpu_<JOBID>.err
```

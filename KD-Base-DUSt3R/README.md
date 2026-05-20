# KD-Base-DUSt3R

Stage 1 training engine for robust DUSt3R encoder under dark images:

- **Teacher**: Base DUSt3R encoder checkpoint (frozen)
- **Student**: DUSt3R encoder with LoRA (`r=16`, `alpha=32`)
- **Trainable params**: only LoRA layers inside student encoder
- **Siamese note**: DUSt3R has two encoder branches; both are supervised through the shared encoder LoRA path.
- **Frozen**:
  - teacher entirely
  - student decoder
  - student base encoder weights

## Loss

- `MSELoss(teacher_encoder_features(clean), student_encoder_features(dark))`

## Darkening augmentation (on-the-fly)

Gamma correction with random gamma from:

- `1.5`
- `2.2`

## Optimizer / Scheduler

- `AdamW`, `lr=1e-4`, `weight_decay=0.05`
- Warmup + cosine decay:
  - warmup: `5%` of total iterations
  - final lr: `1e-6`

## Dataset behavior

- Uses sequence-level split JSONs (`selected_seqs_train_10cat8.json`, `selected_seqs_val_10cat8.json`)
- No mixing train/val sequences.
- Frame subsampling with stride (`stride=5` by default), i.e. every 5th frame.

## Files

- `train_stage1_encoder_kd.py`: main training script
- `submit_stage1_kd_2gpu_ysu.sh`: 2-GPU Slurm launcher

## YSU Usage

```bash
cd ~/project_Hayk_Minasyan
sbatch "KD-Base-DUSt3R/submit_stage1_kd_2gpu_ysu.sh"
```

Logs:

- `/mnt/weka/hminasyan/logs/kd_base_dust3r_2gpu_<jobid>.log`
- `/mnt/weka/hminasyan/logs/kd_base_dust3r_2gpu_<jobid>.err`

Checkpoints:

- `student_lora_last.pth`
- `student_lora_best.pth`
- `log.txt`

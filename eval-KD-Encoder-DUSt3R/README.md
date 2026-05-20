# eval-KD-Encoder-DUSt3R

Fast GPU evaluation for KD-Encoder DUSt3R checkpoints with averaged Chamfer distance on:

- `bottle`
- `cup`
- `donut`
- `teddybear`
- `couch`
- `toytrain`

## Scenarios

The script evaluates and averages Chamfer over sequence IDs in the selected split for:

1. `dust3r_clean`
2. `dust3r_dark` (average over gamma 1.5 and 2.2)
3. `ourdust3r_kd20_dark` (average over gamma 1.5 and 2.2)
4. `ourdust3r_kd50_dark` (average over gamma 1.5 and 2.2)
5. `ourdust3r_kd50_clean`

## Defaults

- `n_frames=20`
- `image_size=224`
- `dark_gammas = [1.5, 2.2]`

## Paths used by default (YSU)

- Base DUSt3R ckpt:
  - `/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- KD 20-epoch ckpt:
  - `/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_base_dust3r_stage1_2gpu/student_lora_best.pth`
- KD 50-epoch ckpt:
  - `/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_base_dust3r_stage1_2gpu_stride2_ep50/student_lora_best.pth`

## Run on YSU

```bash
cd ~/project_Hayk_Minasyan
sbatch "eval-KD-Encoder-DUSt3R/submit_eval_kd_encoder_dust3r_ysu.sh"
```

Default dataset settings in submit script:

- `CO3D_PROC=/mnt/weka/hminasyan/data/co3d_processed`
- `CO3D_RAW=/mnt/weka/hminasyan/data/co3d`
- `SPLIT=test`

## Outputs

- Slurm log:
  - `/mnt/weka/hminasyan/logs/eval_kd_encoder_dust3r_<jobid>.log`
- Slurm err:
  - `/mnt/weka/hminasyan/logs/eval_kd_encoder_dust3r_<jobid>.err`
- JSON summary:
  - `/mnt/weka/hminasyan/outputs/eval_kd_encoder_dust3r/chamfer_eval_kd_encoder_<split>_n<n_frames>.json`


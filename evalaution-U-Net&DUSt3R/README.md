# Evaluation: U-Net + DUSt3R

This folder evaluates Chamfer distance for six categories:

- `bottle`, `cup`, `donut`, `teddybear`, `couch`, `toytrain`

with `n_frames=20` by default, comparing:

1. `dust3r_clean`
2. `dust3r_motion_blur`
3. `dust3r_defocus_blur`
4. `unet_dust3r_clean`
5. `unet_dust3r_motion_blur`
6. `unet_dust3r_defocus_blur`

where "unet" is the KD student from `student_best.pth`.

## Files

- `eval_unet_dust3r_chamfer.py`: main evaluator.
- `submit_eval_unet_dust3r_ysu.sh`: YSU Slurm launcher.

## Blur definitions

Blur generation matches `restoration_kd_ysu/train_kd_restormer_frontend.py`:

- Motion blur: linear horizontal PSF `25x25`, center row ones, normalized.
- Defocus blur: disk PSF radius `7`, kernel `15x15`, normalized.

## Run on YSU

```bash
cd ~/project_Hayk_Minasyan
sbatch "evalaution-U-Net&DUSt3R/submit_eval_unet_dust3r_ysu.sh"
```

Output JSON default:

`/mnt/weka/hminasyan/outputs/eval_unet_dust3r/chamfer_unet_dust3r_test_10cat8_n20.json`

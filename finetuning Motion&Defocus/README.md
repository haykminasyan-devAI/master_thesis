# DeblurDiNAT + DUSt3R (Motion & Defocus)

This folder mirrors the blur finetuning workflow, but replaces Gaussian-blur input with synthetic motion/defocus blur generated on-the-fly.

## What stays the same as blur finetuning

- DeblurDiNAT front-end + DUSt3R training loop.
- Freeze mode defaults to `deblurdinat_only` (DUSt3R frozen).
- DDP + AdamW + warmup+cosine schedule.
- Sample-level random split (`val_ratio`, `test_ratio`) created inside training script.
- Logging/checkpoint behavior (`log.txt`, `checkpoint-last.pth`, `checkpoint-best-val.pth`).

## What changes

- Blur type is synthetic and on-the-fly in model wrapper:
  - Motion blur: 25x25 kernel, center row = 1s, normalized.
  - Defocus blur: disk kernel radius 7 (15x15), normalized.
- For each sample, blur type is picked randomly:
  - `motion_prob` (default `0.5`) for motion,
  - `1 - motion_prob` for defocus.

## Files

- `deblurdinat/train_motion_defocus.py`: main training script.
- `deblurdinat/model_motion_defocus.py`: DeblurDiNAT + frozen DUSt3R wrapper with synthetic blur.
- `deblurdinat/submit_train_motion_defocus_research.slurm.sh`: YSU Slurm launcher.

## YSU paths (defaults in Slurm script)

- Dataset root: `/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed`
- DUSt3R ckpt: `/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`
- DeblurDiNAT weights: `${PROJECT_DIR}/DeblurDiNAT/results/DeblurDiNATL/models/DeblurDiNATL.pth`

## Notes on splitting

Even though your dataset root has `selected_seqs_train_10cat8.json` / `val` / `test`, this pipeline intentionally follows the same split behavior as old blur finetuning: it loads `Co3d(split="train")` and then does random sample-level split by `val_ratio`/`test_ratio`.

Default split in this folder is now:
- `train=80%`, `val=20%`, `test=0%`.

The training script resolves train split JSON robustly. Default is `split_train=train_10cat8`,
so it reads `selected_seqs_train_10cat8.json` in your YSU dataset root.

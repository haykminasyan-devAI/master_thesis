# KD-Zero-Reference: URetinex-Net (teacher) → Zero-DCE (student)

Knowledge distillation for **low-light enhancement** with CO3D frames + **on-the-fly synthetic low-light** (`synthetic_lowlight.py`). Teacher stays frozen (`eval`, no gradients).

References:
- [URetinex-Net](https://github.com/AndersonYong/URetinex-Net) (CVPR 2022)
- [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) (CVPR 2020)

Respect each repo’s **license / non-commercial** clauses.

## Loss (defaults in `train_kd_zerodce.py`)

| Term | λ default | Role |
|------|-----------|------|
| Charbonnier(student, teacher) | 1.0 | Robust pixel match to teacher |
| VGG perceptual | 0.5 | Structure / texture for downstream geometry |
| Spatial consistency (Zero-DCE style) | 1.0 | Neighbor gradients vs input |
| Exposure (`L_exp`) | 2.0 | Patch brightness anchor |
| TV on curve maps `A` | 200.0 | Smooth curves |

Teacher weights are **not** loaded into the student.

## One-time setup (clone + checkpoints)

From project root:

```bash
mkdir -p external
git clone https://github.com/AndersonYong/URetinex-Net.git external/URetinex-Net
git clone https://github.com/Li-Chongyi/Zero-DCE.git external/Zero-DCE
```

URetinex expects under `external/URetinex-Net/ckpt/`:

- `init_low.pth`
- `unfolding.pth`
- `L_adjust.pth`

(Usually bundled in the repo — verify [their `ckpt/` folder](https://github.com/AndersonYong/URetinex-Net/tree/main/ckpt).)

```bash
pip install -r KD-Zero-Reference/requirements.txt
```

## 7:1 train / val split JSONs (on cluster)

Run once where CO3D lives (writes **into** `co3d_root`):

```bash
python KD-Zero-Reference/scripts/build_split_7_1.py \
  --co3d_root /mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed \
  --src_split train_10cat8 \
  --out_train_suffix train_10cat8_7v1 \
  --out_val_suffix val_10cat8_7v1
```

Training expects:

- `selected_seqs_train_10cat8_7v1.json`
- `selected_seqs_val_10cat8_7v1.json`

## Rsync this folder to YSU

From your dev machine:

```bash
rsync -avz \
  /path/to/project_Hayk_Minasyan/KD-Zero-Reference \
  /path/to/project_Hayk_Minasyan/external/URetinex-Net \
  /path/to/project_Hayk_Minasyan/external/Zero-DCE \
  USER@cluster.ysu.am:/home/hminasyan/project_Hayk_Minasyan/
```

(Adjust paths if `external/` lives elsewhere.)

## Slurm (2× GPU)

```bash
cd /home/hminasyan/project_Hayk_Minasyan

export CO3D_ROOT=/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed
export OUTPUT_DIR=/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_zerodce_uretinex
export URETINEX_ROOT=/home/hminasyan/project_Hayk_Minasyan/external/URetinex-Net
export ZERODCE_ROOT=/home/hminasyan/project_Hayk_Minasyan/external/Zero-DCE

sbatch KD-Zero-Reference/submit_kd_zerodce_2gpu.slurm.sh
```

Override via env: `BATCH_SIZE`, `EPOCHS`, `LR`, `WEIGHT_DECAY`, `SPLIT_TRAIN`, `SPLIT_VAL`, `OUTPUT_DIR`, etc.

## Local dry-run (1 GPU)

```bash
export CUDA_VISIBLE_DEVICES=0
python KD-Zero-Reference/train_kd_zerodce.py \
  --co3d_root /path/to/co3d_processed_10cat8seq_fixed \
  --uretinex_root external/URetinex-Net \
  --zerodce_root external/Zero-DCE \
  --output_dir /tmp/kd_zero_test \
  --epochs 1 --batch_size 2 --num_workers 2
```

(Use `torchrun --nproc_per_node=1` the same way as Slurm for DDP tests.)

## Outputs

- `output_dir/args.json` — full hyperparameters
- `output_dir/log.txt` — one JSON line per epoch
- `output_dir/student_last.pth` and `student_epoch_*.pth` — Zero-DCE weights

## Notes

- **Image lists**: dataset walks all `frame*.jpg` under each sequence in the split JSON (10 categories by default).
- **URetinex** is heavy; first forward can be slow. If OOM, lower `--batch_size` (per GPU).
- **PyTorch** 2.x: VGG uses `torchvision` weights; old `pretrained=True` is a fallback in `losses.py`.

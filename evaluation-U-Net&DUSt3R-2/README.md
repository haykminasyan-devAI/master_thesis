# Evaluation: KD student + DUSt3R (v2)

Chamfer distance vs CO3D `pointcloud.ply` for **six categories**:

`bottle`, `cup`, `donut`, `teddybear`, `couch`, `toytrain`

**Scenarios** (same naming as v1; `unet_*` = Restormer-KD `StudentRestorationFrontEnd` + DUSt3R):

1. `dust3r_clean`
2. `dust3r_motion_blur`
3. `dust3r_defocus_blur`
4. `unet_dust3r_clean`
5. `unet_dust3r_motion_blur`
6. `unet_dust3r_defocus_blur`

**Default student checkpoint (50-epoch KD, 1 GPU):**  
`/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_restormer_frontend_1gpu/student_best.pth`

**Data root and split (important):**  
The six categories live under **`/mnt/weka/hminasyan/data/co3d_processed`** with split **`test`** (`selected_seqs_test.json`).  
Using **`test_10cat8`** on **`co3d_processed_10cat8seq_fixed`** will fail: that split lists the *other* ten classes (apple, banana, …), not bottle/cup/…

**Blur in this eval:** horizontal motion `25×25`; defocus disk radius `7` (`15×15` kernel)—see `eval_unet_dust3r_chamfer.py`.

## Sync to YSU

The `&` must be **quoted** on the **remote** path too, otherwise bash runs the part after `&` in the background (`DUSt3R-2/: No such file or directory`).

```bash
cd /path/to/project_Hayk_Minasyan
rsync -avz 'evaluation-U-Net&DUSt3R-2/' 'hminasyan@cluster.ysu.am:~/project_Hayk_Minasyan/evaluation-U-Net&DUSt3R-2/'
```

Or escape: `evaluation-U-Net\&DUSt3R-2/` on both source and destination.

## Submit on YSU

```bash
cd ~/project_Hayk_Minasyan
sbatch "evaluation-U-Net&DUSt3R-2/submit_eval_unet_dust3r_ysu.sh"
```

Override student or output path:

```bash
STUDENT_CKPT=/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_restormer_frontend_1gpu/student_best.pth \
OUT_JSON=/mnt/weka/hminasyan/outputs/eval_unet_dust3r2/my_run.json \
sbatch "evaluation-U-Net&DUSt3R-2/submit_eval_unet_dust3r_ysu.sh"
```

## Results

- **Averaged Chamfer** per scenario: `summary.<scenario>.mean_all` in the JSON; also printed at the end of the log.
- Per-category means: `summary.<scenario>.mean_by_category`.

Default JSON directory: `/mnt/weka/hminasyan/outputs/eval_unet_dust3r2/` (timestamp in filename).

## Logs

`/mnt/weka/hminasyan/logs/eval_unet_dust3r2_<JOBID>.log` and `.err`

Use `co3d_env` Python (has `roma` for DUSt3R `cloud_opt`).

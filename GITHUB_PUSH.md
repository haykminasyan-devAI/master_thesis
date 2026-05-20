# What to push to GitHub (`haykminasyan-devAI`)

Your working tree is ~**81 GB**; the GitHub repo should be **code + configs + docs** only (typically tens of MB).

Remote (update to SSH after Hayk key setup):

```bash
git remote set-url origin git@github.com-hayk:haykminasyan-devAI/master_thesis.git
```

---

## Do NOT push (ignored via `.gitignore`)

| Path / pattern | Size (approx.) | Why |
|----------------|----------------|-----|
| `data/` | ~7.7 GB | CO3D images on disk |
| `co3d/*/images/` | varies | Frame JPEGs under repo `co3d/` |
| `checkpoints/`, `*.pth` | ~4.7 GB+ | DUSt3R / Uformer weights |
| `interactive_demo/demo_ckpts/` | ~13 GB | Demo + joint finetune weights |
| `outputs/` | ~24 GB | Metrics, PLY, plots, `viz_selections/` |
| `finetune_blur_runs/` | ~24 GB | WandB, checkpoints, logs |
| `logs/`, `wandb/`, `.gradio/` | small–large | Run logs / UI cache |
| `inference_rest/Restormer/**/pretrained_models/` | ~400 MB | Restormer teachers (download) |
| `DeblurDiNAT/`, `IFAN/`, `Uformer/` | code clones | Separate git repos — clone on each machine |

Download weights on each cluster (document paths in README), e.g. DUSt3R Naver Labs URL, DeblurDiNAT GoPro weights, Restormer Google Drive per `restoration_kd_ysu/setup_restormer_teacher_ysu.sh`.

---

## SHOULD push (your project code)

### Core experiment code

| Folder | Purpose |
|--------|---------|
| `finetune_blur/` | DeblurDiNAT + DUSt3R joint training |
| `finetuning Motion&Defocus/` | Motion/defocus DeblurDiNAT variant |
| `restoration_kd_ysu/` | KD Restormer student frontend |
| `KD-Base-DUSt3R/` | Encoder LoRA KD |
| `KD-Zero-Reference/` | Dark-scene KD |
| `motion_blur_ysu/` | Motion LoRA experiments |
| `finetune_motion_blur/`, `finetune_defocus/`, `finetune_noise/` | Other front-end finetunes |
| `eval-KD-Encoder-DUSt3R/`, `evalaution-U-Net&DUSt3R/`, `evaluation-U-Net&DUSt3R-2/` | Chamfer eval |
| `EVAL-DeblurDinat-Motion-Defocus/`, `EVAL-kd_zerodce_uretinex/` | Eval scripts |
| `evaluation_blur_and_noise/` | Blur/noise eval utilities |
| `interactive_demo/` | Gradio demo (**without** `demo_ckpts/`) |
| `inference_rest/` | Restormer inference scripts (**not** `pretrained_models/`) |
| `scripts/` | Slurm, degradation, COLMAP, viz helpers |
| `experiments/` | Small experiment scripts (if not huge data inside) |

### `dust3r/` (modified fork)

- **Push:** Python changes (`dust3r/dust3r/*.py`, `croco/`, `requirements.txt`, etc.)
- **Do not push:** `dust3r/checkpoints/*.pth` (covered by `*.pth`)

You already have local edits on `dust3r`; those belong in the repo.

### Optional / case-by-case

| Item | Recommendation |
|------|----------------|
| `co3d/` Python package (`co3d/co3d/`, examples) | Push if you rely on it; keep `co3d/*/images/` ignored |
| `outputs/blur_exp_2/plot_blur_exp.py` only | Push scripts; ignore `outputs/` parent |
| Root `README.md` | **Add** a root README (clone deps, data paths, Slurm) — missing today |

---

## Third-party repos (clone, do not copy into git)

These directories have their own `.git` and are **gitignored**:

```bash
git clone <DeblurDiNAT-url> DeblurDiNAT
# IFAN, Uformer similarly if needed
```

Document URLs and commit hashes in your root README.

---

## Current git state (summary)

- **Tracked today:** ~151 files (mostly `dust3r/`, `scripts/`, small `outputs/` paths from an older commit).
- **Ahead of origin:** 3 commits not pushed.
- **Many untracked folders:** finetuning, KD, demo, eval — should be added **after** `.gitignore` is updated.
- **Origin:** `https://github.com/haykminasyan-devAI/master_thesis.git` → switch to `git@github.com-hayk:...` for SSH.

---

## Safe workflow before first big push

```bash
cd ~/project_Hayk_Minasyan

# 1) Remote for Hayk SSH key
git remote set-url origin git@github.com-hayk:haykminasyan-devAI/master_thesis.git

# 2) See what WOULD be added (no huge files)
git add -A
git status
git diff --cached --stat

# 3) If any file > 50MB appears, stop and fix .gitignore
git diff --cached --name-only | while read f; do
  test -f "$f" && du -h "$f"
done | sort -hr | head -20

# 4) Commit & push
git commit -m "Add experiment code, demos, eval; ignore data and checkpoints"
git push -u origin main
```

If push rejects a large file already in history, you need `git filter-repo` or a fresh repo — ask before force-pushing.

---

## Suggested root README sections

1. Title + short description (DUSt3R robustness: blur, KD, DeblurDiNAT).
2. **Dependencies:** `conda`, `co3d_env`, `pip install -r dust3r/requirements.txt`.
3. **External clones:** DeblurDiNAT, submodules if any.
4. **Data:** CO3D path on cluster; not in repo.
5. **Checkpoints:** where to download DUSt3R / finetuned `.pth`.
6. **Key commands:** train, eval, `interactive_demo/run_demo_on_node.sh`.

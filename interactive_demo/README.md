# Interactive demo — DUSt3R + finetuned DeblurDiNAT

A Gradio interface (same UX as the official `dust3r/demo.py`) with a **Pipeline**
dropdown that lets you switch between three models on the fly:

| Pipeline key                          | What it is                                                  |
| ------------------------------------- | ----------------------------------------------------------- |
| `dust3r`                              | Vanilla pretrained DUSt3R (ViT-L 224 linear)                |
| `deblur_finetuned_5_10_20_30`         | Joint DeblurDiNAT + DUSt3R finetuned on σ ∈ {5, 10, 20, 30} |
| `deblur_finetuned_5_10_20_30_50`      | Same, σ ∈ {5, 10, 20, 30, 50}                               |

Models are lazy-loaded on first selection and cached on the GPU, so switching is
instant after the first time.

---

## 1. Layout

```
interactive_demo/
├── demo_finetuned.py            # Gradio app (run directly OR inside Docker)
├── README.md
└── docker/
    ├── run.sh                   # Build + launch
    ├── docker-compose-cuda.yml
    └── files/
        ├── cuda.Dockerfile
        ├── requirements-demo.txt
        └── entrypoint.sh
```

---

## 2. Required assets on the host

You need three things accessible from the machine that runs the demo:

1. **This repo** — `project_Hayk_Minasyan/` with the `dust3r/` submodule populated
   (`git submodule update --init --recursive` if you cloned it shallow).
2. **The DeblurDiNAT repo** — same one you used during finetuning
   (e.g. `~/project_Hayk_Minasyan/DeblurDiNAT`).
3. **A checkpoints directory** containing:

   ```
   <CKPT_DIR>/
   ├── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth   # default base for vanilla + U-Net + LoRA in this repo
   ├── DUSt3R_ViTLarge_BaseDecoder_224_linear.pth  # optional fallback
   ├── student_best.pth                          # KD Restormer student (or under restoration_kd_ysu/...)
   ├── student_lora_best_20ep.pth / _50ep.pth    # encoder LoRA (optional)
   ├── joint_sigmas_5_10_20_30/checkpoint-best-val.pth
   └── joint_sigmas_5_10_20_30_50/checkpoint-best-val.pth
   ```

   (Override via env vars if paths differ — see [`docker/files/entrypoint.sh`](docker/files/entrypoint.sh).)

You can put any subset of those .pth files in place; the dropdown only lists
pipelines whose checkpoint is found.

---

## 3. Run with Docker (recommended)

Prerequisites on the host:
- Docker
- `docker compose` (v2) or `docker-compose` (v1)
- NVIDIA Container Toolkit (`nvidia-docker`) so the container sees the GPU.

Then:

```bash
cd interactive_demo/docker
bash run.sh \
  --project-dir      /home/asds/project_Hayk_Minasyan \
  --deblurdinat-dir  /home/asds/project_Hayk_Minasyan/DeblurDiNAT \
  --checkpoints-dir  /mnt/weka/hminasyan/demo_ckpts \
  --port             7860 \
  --image-size       512
```

This will:

1. Build the CUDA image (first run only).
2. Mount your project, DeblurDiNAT and checkpoints into `/workspace/...`.
3. Compile the RoPE CUDA kernels inside the container if missing.
4. Launch `demo_finetuned.py --local_network` on port `7860`.

Open `http://<host>:7860/` (or use SSH port-forwarding from your laptop:
`ssh -L 7860:localhost:7860 user@host`).

To stop: `Ctrl+C`, then `docker compose -f docker-compose-cuda.yml down`.

### Picking another host port
```bash
bash run.sh ... --port 7871
```

### Pointing at non-default checkpoint filenames
Set them via environment variables before calling `run.sh`:

```bash
export DUST3R_CKPT_IN_CONTAINER=/workspace/checkpoints/my_dust3r.pth
export FT_4_IN_CONTAINER=/workspace/checkpoints/my_5_10_20_30.pth
export FT_5_IN_CONTAINER=/workspace/checkpoints/my_5_10_20_30_50.pth
bash run.sh ...
```

---

## 4. Run without Docker (e.g. on the YSU cluster)

If Docker isn't available, run the same script directly inside your Conda env.

```bash
# 1. Activate the env you used for finetuning
source activate co3d_env   # or whatever yours is called

# 2. Make sure gradio + trimesh + plyfile are installed
pip install gradio==4.44.0 trimesh plyfile

# 3. Launch (defaults: 512 DPT + student_best if paths exist; override as needed)
cd ~/project_Hayk_Minasyan
python interactive_demo/demo_finetuned.py \
  --dust3r_ckpt              ~/project_Hayk_Minasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --kd_restormer_student_ckpt ~/project_Hayk_Minasyan/restoration_kd_ysu/outputs_from_ysu/kd_restormer_frontend_1gpu/student_best.pth \
  --kd20_ckpt                /path/to/student_lora_best_20ep.pth \
  --kd50_ckpt                /path/to/student_lora_best_50ep.pth \
  --image_size 512 \
  --local_network --server_port 7860
```

To run this on a YSU compute node, wrap it in `srun` or `sbatch` and tunnel the
port back through the login node, e.g.:

```bash
# On your laptop:
ssh -L 7860:<compute-node>:7860 hminasyan@cluster.ysu.am
# In a second terminal on YSU login node:
srun -p research --gres=gpu:1 --pty bash
python interactive_demo/demo_finetuned.py ...
```

---

## 5. Using the UI

1. Pick a **Pipeline** from the dropdown.
2. Drop **1 or more** images of the same scene into the file box.
3. (Optional) Tweak `min_conf_thr`, `cam_size`, scene-graph type, # iterations.
4. Click **Run**.

The 3D viewer below appears once global alignment finishes. Sliders below the
button can be moved without re-running inference (they only re-render the
exported `.glb`).

> Tip: with 1–2 images, DUSt3R uses `PairViewer` mode (no global alignment) and
> the result is near-instant. With more images, expect ~20–60 s on an A100.

---

## 6. How model loading works

`demo_finetuned.py` defines a `ModelRegistry` that loads on demand:

- **`dust3r`** → `dust3r.model.load_model(dust3r_ckpt)`
- **finetuned variants** → `finetune_blur.deblurdinat.model.build_model(...)`
  followed by `model.load_state_dict(ckpt["model"], strict=False)`. We then
  expose `model.patch_size` from the inner DUSt3R so `dust3r.utils.image.load_images`
  works unchanged.

Both objects are then passed straight to `dust3r.demo.get_reconstructed_scene`,
which is the same routine the official demo uses — so the rest of the pipeline
(global aligner, glb export, depth/conf rendering) is identical.

---

## 7. Troubleshooting

**`ERROR: No CUDA GPUs are available`** — you're on a login node. Get a GPU
session (`srun --gres=gpu:1 --pty bash`) or run inside Docker with the NVIDIA
toolkit.

**`ModuleNotFoundError: No module named 'finetune_blur'`** — you're running the
script from outside the project root. Either `cd` into the project root or set
`PYTHONPATH=/path/to/project_Hayk_Minasyan`.

**`Address already in use`** — change `--server_port` (or `--port` for Docker)
to a free one (7861, 7862, …).

**Browser shows a stale visualization** — hard-reload (`Cmd/Ctrl+Shift+R`) or
launch on a new port; Gradio sometimes re-uses the same WebSocket id.

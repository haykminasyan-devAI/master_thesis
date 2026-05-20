# Restormer Inference Quickstart

This folder contains a ready workflow to test [Restormer](https://github.com/swz30/Restormer) on your synthetic corrupted **teddybear** frames.

## What is prepared

- `inference_rest/Restormer/` (cloned repo)
- `inference_rest/prepare_teddybear_corruptions.py`
- `inference_rest/run_restormer_inference.sh`

## 1) Collect corrupted frames

Run from project root:

```bash
python inference_rest/prepare_teddybear_corruptions.py \
  --source_root /path/to/your/corrupted/frames/root \
  --target_root inference_rest/corrupted/teddybear \
  --limit_per_corruption 300 \
  --mode copy
```

This creates:

- `inference_rest/corrupted/teddybear/noise`
- `inference_rest/corrupted/teddybear/blur`
- `inference_rest/corrupted/teddybear/motion_blur`
- `inference_rest/corrupted/teddybear/raining`
- `inference_rest/corrupted/teddybear/defocus_blur`

## 2) Prepare Restormer dependencies

Inside `inference_rest/Restormer`, install dependencies (see upstream `INSTALL.md`), then get task checkpoints into:

- `Motion_Deblurring/pretrained_models/motion_deblurring.pth`
- `Deraining/pretrained_models/deraining.pth`
- `Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth`
- `Denoising/pretrained_models/gaussian_color_denoising_blind.pth`

## 3) Run inference

```bash
bash inference_rest/run_restormer_inference.sh
```

Optional tiled inference for large images:

```bash
TILE=720 TILE_OVERLAP=32 bash inference_rest/run_restormer_inference.sh
```

Outputs are saved under:

- `inference_rest/restored/teddybear/<RestormerTaskName>/`

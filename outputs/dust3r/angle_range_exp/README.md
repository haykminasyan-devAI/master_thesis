# Angle Range Experiment

**Question:** How does limiting the viewpoint coverage affect DUSt3R 3D reconstruction quality?

## Setup

- **Sequences:** 6 CO3D categories — teddybear, hydrant, cup, bottle, toybus, toytrain
- **Frames:** n = 2, 3, 4, 5, 6, 7, 8, 9, 10 (clean, no masking)
- **Metric:** Chamfer Distance (lower = better)

## Angle Ranges

| Tag | Angle Range | % of sequence used |
|---|---|---|
| `range_0_60deg` | 0° – 60° | first 16.67% of frames |
| `range_0_90deg` | 0° – 90° | first 25% of frames |
| `range_0_180deg` | 0° – 180° | first 50% of frames |

## Frame Selection

CO3D sequences are captured by a person walking around the object in a full 360° circle.
Frames are sorted by index, so frame index ≈ rotation angle.

For a given angle range and n_frames:

1. **Limit the pool** — take only the first X% of all frames in the sequence.
   - Example: teddybear has 202 frames. For 0–90°: pool = first 50 frames (frames 1–50).

2. **Sample evenly** — apply `np.linspace(0, pool_size-1, n_frames)` to pick n_frames evenly spaced from the pool.
   - Example: n_frames=5 from pool of 50 → indices [0, 12, 24, 37, 49] → frames 1, 13, 25, 38, 50.

This ensures frames always span the full selected angular range, regardless of n_frames.

## Files

| File | Purpose |
|---|---|
| `inference.py` | Runs DUSt3R inference with `--frame_pool_pct` parameter |
| `run_angle_range_exp.sh` | SLURM job script — dispatches all 162 runs across 4 GPUs |
| `plot_angle_range_exp.py` | Reads metrics and plots Chamfer Distance vs n_frames |
| `angle_range_compare.png` | Output plot (generated after experiment completes) |

## Output Structure

```
range_0_60deg/
    teddybear_101_11758_21048/
        frames_02/metrics.txt
        frames_03/metrics.txt
        ...
        frames_10/metrics.txt
    hydrant_106_12648_23157/
        ...
range_0_90deg/
    ...
range_0_180deg/
    ...
```

#!/usr/bin/env python3
"""LoRA fine-tuning: cross-view attention only, directional motion blur pairs."""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
# rsync without --delete can leave the pre-rename package; it shadows CroCo's top-level `models`.
_stale_models = os.path.join(PROJECT_DIR, "motion_blur_ysu", "models")
if os.path.isfile(os.path.join(_stale_models, "lora_cross_attn.py")):
    print(
        "Removing stale motion_blur_ysu/models/ (conflicts with dust3r/croco `models`).",
        file=sys.stderr,
    )
    shutil.rmtree(_stale_models)

DUST3R_DIR = os.path.join(PROJECT_DIR, "dust3r")
for p in (DUST3R_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from dust3r.inference import loss_of_one_batch
from dust3r.losses import ConfLoss, L21, Regr3D

from motion_blur_ysu.datasets.co3d_lora_motion import Co3dLoraMotion
from motion_blur_ysu.dust3r_lora.lora_cross_attn import build_lora_dust3r_cross_attn, count_trainable

TRAIN_CATEGORIES = [
    "apple",
    "banana",
    "baseballbat",
    "baseballglove",
    "bicycle",
    "broccoli",
    "bowl",
    "cake",
    "car",
    "carrot",
]
TEST_CATEGORIES = ["cup", "couch", "bottle", "teddybear", "donut", "toytrain"]
EVAL_CONDITIONS = ["clean-clean", "blur-blur", "clean-blur"]


def get_args():
    p = argparse.ArgumentParser("DUSt3R LoRA motion-blur (cross-attn only)")
    p.add_argument("--co3d_root", default="/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed")
    p.add_argument("--dust3r_ckpt", default="/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_train", default="train_10cat8")
    p.add_argument("--split_val", default="val_10cat8")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--symmetrize_batch", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--dist_url", default="env://")
    return p.parse_args()


def _require_cuda_for_distributed_job():
    """torchrun sets RANK/LOCAL_RANK; fail fast on login nodes without GPUs."""
    if "RANK" not in os.environ:
        return
    loc = int(os.environ.get("LOCAL_RANK", "0"))
    lw = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE",
            os.environ.get("WORLD_SIZE", "1"),
        )
    )
    if not torch.cuda.is_available():
        if loc == 0:
            print(
                "ERROR: No CUDA on this machine — you are likely on a CPU/login node.\n"
                "Multi-GPU training must run inside a GPU job (Slurm sbatch/salloc + srun), "
                "not from a plain SSH shell on the login node.\n"
                "Example: sbatch motion_blur_ysu/submit_train_lora_research.slurm.sh",
                file=sys.stderr,
            )
        raise SystemExit(1)
    if torch.cuda.device_count() < lw:
        if loc == 0:
            print(
                f"ERROR: Need {lw} GPU(s) on this node (LOCAL_WORLD_SIZE) but "
                f"torch.cuda.device_count() == {torch.cuda.device_count()}.\n"
                "Match --nproc_per_node to allocated GPUs, or request more GPUs in your job.",
                file=sys.stderr,
            )
        raise SystemExit(1)


def make_loader(ds, batch_size, num_workers, distributed, shuffle):
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=shuffle)
    else:
        sampler = (
            torch.utils.data.RandomSampler(ds)
            if shuffle
            else torch.utils.data.SequentialSampler(ds)
        )
    return DataLoader(
        ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, scaler, args, scheduler=None):
    model.train()
    logger = misc.MetricLogger("  ")
    logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch [{epoch}]"
    for batch in logger.log_every(loader, args.print_freq, header):
        out = loss_of_one_batch(
            batch,
            model,
            criterion,
            device,
            symmetrize_batch=bool(args.symmetrize_batch),
            use_amp=args.amp,
        )
        loss = out["loss"]
        if isinstance(loss, tuple):
            loss = loss[0]
        lv = float(loss.detach())
        if not math.isfinite(lv):
            continue
        optimizer.zero_grad()
        trainable = [p for p in model.parameters() if p.requires_grad]
        clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
        scaler(loss, optimizer, clip_grad=clip, parameters=trainable)
        if scheduler is not None:
            scheduler.step()
        logger.update(loss=lv)
        logger.update(lr=optimizer.param_groups[0]["lr"])
    logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, loader, device, args):
    model.eval()
    logger = misc.MetricLogger("  ")
    for batch in logger.log_every(loader, args.print_freq, "Val:"):
        out = loss_of_one_batch(
            batch,
            model,
            criterion,
            device,
            symmetrize_batch=False,
            use_amp=args.amp,
        )
        loss = out["loss"]
        if isinstance(loss, tuple):
            loss = loss[0]
        logger.update(loss=float(loss.detach()))
    logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in logger.meters.items()}


def main():
    args = get_args()
    _require_cuda_for_distributed_job()
    misc.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    overlap = set(TRAIN_CATEGORIES) & set(TEST_CATEGORIES)
    if overlap:
        raise ValueError(f"Train/test overlap: {overlap}")

    ds_tr = Co3dLoraMotion(
        ROOT=args.co3d_root,
        split=args.split_train,
        resolution=args.resolution,
        allowed_categories=TRAIN_CATEGORIES,
        train_mode=True,
        seed=args.seed,
        length_factor=256,
    )
    loaders_val = {}
    for cond in EVAL_CONDITIONS:
        ds_v = Co3dLoraMotion(
            ROOT=args.co3d_root,
            split=args.split_val,
            resolution=args.resolution,
            allowed_categories=TRAIN_CATEGORIES,
            train_mode=False,
            eval_condition=cond,
            seed=args.seed,
            length_factor=16,
        )
        loaders_val[cond] = make_loader(
            ds_v, args.batch_size, args.num_workers, args.distributed, shuffle=False
        )

    loader_tr = make_loader(
        ds_tr, args.batch_size, args.num_workers, args.distributed, shuffle=True
    )

    model = build_lora_dust3r_cross_attn(
        args.dust3r_ckpt,
        device="cpu",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    raw = model.module if hasattr(model, "module") else model

    n_t, n_a = count_trainable(raw)
    if misc.is_main_process():
        print(f"Trainable / total params: {n_t:,} / {n_a:,}")

    criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )

    steps_per_epoch = len(loader_tr)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
        return cos * (1.0 - args.eta_min / args.lr) + (args.eta_min / args.lr)

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = NativeScaler(enabled=bool(args.amp))

    best = float("inf")
    best_path = os.path.join(args.output_dir, "checkpoint_lora_best_val.pth")

    for epoch in range(args.epochs):
        if args.distributed and hasattr(loader_tr.sampler, "set_epoch"):
            loader_tr.sampler.set_epoch(epoch)
        stats = train_one_epoch(
            model,
            criterion,
            loader_tr,
            optimizer,
            device,
            epoch,
            scaler,
            args,
            scheduler=scheduler,
        )
        val_out = {}
        for cond, ld in loaders_val.items():
            val_out[cond] = evaluate(model, criterion, ld, device, args)

        if misc.is_main_process():
            row = {
                "epoch": epoch,
                "train_loss": stats["loss"],
                "lr": optimizer.param_groups[0]["lr"],
                **{f"val_{k.replace('-', '_')}_loss": v["loss"] for k, v in val_out.items()},
            }
            print(row)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(row) + "\n")

        key = val_out["clean-blur"]["loss"]
        if float(key) < best:
            best = float(key)
            ckpt = {
                "epoch": epoch,
                "model": raw.state_dict(),
                "args": vars(args),
                "best_val_clean_blur_loss": best,
            }
            if misc.is_main_process():
                torch.save(ckpt, best_path)
                torch.save(
                    {**ckpt, "note": "last"},
                    os.path.join(args.output_dir, "checkpoint_lora_last.pth"),
                )

    if misc.is_main_process():
        print(f"Done. best val clean-blur loss={best:.6f} -> {best_path}")


if __name__ == "__main__":
    main()

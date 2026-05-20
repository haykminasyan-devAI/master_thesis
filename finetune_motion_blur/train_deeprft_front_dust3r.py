#!/usr/bin/env python3
"""Fine-tune DeepRFT(frontend) + DUSt3R on motion-blurred CO3D (DDP-ready)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DUST3R_DIR = os.path.join(PROJECT_DIR, "dust3r")
for p in [DUST3R_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.distributed.elastic.multiprocessing.errors import record
from dust3r.losses import ConfLoss, Regr3D, L21
from dust3r.inference import loss_of_one_batch

from finetune_motion_blur.deeprft.model import build_model
from finetune_motion_blur.datasets.co3d_motion import Co3dMotion


def get_args_parser():
    p = argparse.ArgumentParser("DeepRFT + DUSt3R motion fine-tuning")
    p.add_argument("--co3d_root", required=True)
    p.add_argument("--motion_root", required=True)
    p.add_argument("--motion_tag", required=True)
    p.add_argument("--dust3r_ckpt", required=True, help="pretrained DUSt3R .pth")
    p.add_argument("--deeprft_repo", required=True, help="path to DeepRFT clone")
    p.add_argument("--deeprft_weights", default=None)
    p.add_argument("--output_dir", default="./output_deeprft_motion")
    p.add_argument("--freeze", default="deeprft_only",
                   choices=["deeprft_only", "deeprft_and_decoder", "all"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--grad_checkpoint", type=int, default=1, choices=[0, 1])
    p.add_argument("--frontend_checkpoint", type=int, default=0, choices=[0, 1])
    p.add_argument("--deeprft_num_res", type=int, default=8)
    p.add_argument("--symmetrize_batch", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--keep_freq", type=int, default=10)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p


def split_indices(n_items, val_ratio, test_ratio, seed):
    if val_ratio <= 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Require val_ratio > 0, test_ratio >= 0, and sum < 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = int(round(n_items * val_ratio))
    n_test = int(round(n_items * test_ratio))
    n_train = n_items - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split leaves no training samples")
    return idx[:n_train].tolist(), idx[n_train:n_train+n_val].tolist(), idx[n_train+n_val:].tolist()


def make_loader(subset, batch_size, num_workers, distributed, is_train):
    if distributed:
        sampler = DistributedSampler(subset, shuffle=is_train, drop_last=is_train)
    else:
        sampler = (
            torch.utils.data.RandomSampler(subset)
            if is_train else torch.utils.data.SequentialSampler(subset)
        )
    return DataLoader(
        subset, sampler=sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=is_train
    )


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp
        )
        loss = result["loss"]
        if isinstance(loss, tuple):
            loss = loss[0]
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            print(f"WARNING: non-finite loss ({loss_value}), skipping batch")
            continue

        optimizer.zero_grad()
        trainable = [p for p in model.parameters() if p.requires_grad]
        clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
        loss_scaler(loss, optimizer, clip_grad=clip, parameters=trainable)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args, split_name="val"):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"{split_name.capitalize()}:"
    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp
        )
        loss = result["loss"]
        if isinstance(loss, tuple):
            loss = loss[0]
        metric_logger.update(loss=float(loss.detach()))
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@record
def main():
    args = get_args_parser().parse_args()
    misc.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = Co3dMotion(
        motion_root=args.motion_root,
        motion_tag=args.motion_tag,
        split="train",
        ROOT=args.co3d_root,
        resolution=args.resolution,
        aug_crop=16,
    )
    print(f"Dataset: {len(dataset)} pairs (motion_tag={args.motion_tag})")

    train_idx, val_idx, test_idx = split_indices(
        len(dataset), args.val_ratio, args.test_ratio, seed=args.seed
    )
    ds_train = Subset(dataset, train_idx)
    ds_val = Subset(dataset, val_idx)
    ds_test = Subset(dataset, test_idx) if len(test_idx) > 0 else None
    print(f"Split: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test) if ds_test else 0}")

    loader_train = make_loader(ds_train, args.batch_size, args.num_workers, args.distributed, is_train=True)
    loader_val = make_loader(ds_val, args.batch_size, args.num_workers, args.distributed, is_train=False)
    loader_test = (
        make_loader(ds_test, args.batch_size, args.num_workers, args.distributed, is_train=False)
        if ds_test else None
    )

    model = build_model(
        dust3r_ckpt=args.dust3r_ckpt,
        deeprft_repo=args.deeprft_repo,
        deeprft_weights=args.deeprft_weights,
        device="cpu",
        freeze=args.freeze,
        use_grad_checkpoint=bool(args.grad_checkpoint),
        deeprft_num_res=args.deeprft_num_res,
        frontend_checkpoint=bool(args.frontend_checkpoint),
    ).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
        )
        model_without_ddp = model.module

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable (freeze={args.freeze})")

    criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    wu = int(args.warmup_epochs)
    if wu <= 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.eta_min)
    elif wu >= args.epochs:
        scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.epochs)
    else:
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=wu)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - wu), eta_min=args.eta_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[wu])

    loss_scaler = NativeScaler(enabled=bool(args.amp))

    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    best_ckpt = os.path.join(args.output_dir, "checkpoint-best-val.pth")
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.isfile(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as ex:
                print(f"Warning: scheduler not loaded ({ex})")
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from epoch {ckpt['epoch']}")

    use_wandb = bool(args.wandb_project) and misc.is_main_process()
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                resume="allow",
                dir=args.output_dir,
            )
            print(f"W&B run: {wandb.run.url}")
        except Exception as e:
            print(f"WARNING: W&B init failed ({e}). Continuing without W&B.")
            use_wandb = False

    print(f"Training for {args.epochs} epochs (from {start_epoch})")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, loader_train, optimizer, device, epoch, loss_scaler, args)
        val_stats = evaluate(model, criterion, loader_val, device, args, "val")
        test_stats = evaluate(model, criterion, loader_test, device, args, "test") if loader_test else {}
        scheduler.step()

        if misc.is_main_process():
            log_stats = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            print(f"Epoch {epoch}: {log_stats}")
            if use_wandb:
                import wandb
                wandb.log(log_stats, step=epoch)

            ckpt_data = {
                "epoch": epoch,
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "best_val_loss": best_val_loss,
            }
            if args.save_freq and (epoch + 1) % args.save_freq == 0:
                torch.save(ckpt_data, last_ckpt)
            if args.keep_freq and (epoch + 1) % args.keep_freq == 0:
                torch.save(ckpt_data, os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pth"))

            cur_val = float(val_stats.get("loss", float("inf")))
            if cur_val < best_val_loss:
                best_val_loss = cur_val
                ckpt_data["best_val_loss"] = best_val_loss
                torch.save(ckpt_data, best_ckpt)
                print(f"  -> New best val: {best_val_loss:.6f}")

    print(f"Training completed in {(time.time() - t0) / 3600:.1f}h")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

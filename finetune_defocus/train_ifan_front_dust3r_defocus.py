#!/usr/bin/env python3
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

from finetune_defocus.model_ifan_dust3r import build_model
from finetune_defocus.datasets.co3d_defocus import Co3dDefocus


def get_args_parser():
    p = argparse.ArgumentParser("IFAN(front) + DUSt3R defocus fine-tuning")
    p.add_argument("--co3d_root", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--defocus_radius", type=int, default=6)
    p.add_argument("--dust3r_ckpt", required=True)
    p.add_argument("--ifan_repo", required=True)
    p.add_argument("--ifan_ckpt", required=True)
    p.add_argument("--output_dir", default="./output_ifan_defocus")
    p.add_argument("--freeze", default="ifan_only", choices=["ifan_only", "all"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
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
    p.add_argument(
        "--split_strategy",
        default="random",
        choices=["random", "predefined"],
        help="random: legacy ratio split on one train json. predefined: train/val/test json names below.",
    )
    p.add_argument("--train_split", default="train", help="split name for CO3D json (predefined only)")
    p.add_argument("--val_split", default="val", help="split name for CO3D json (predefined only)")
    p.add_argument("--test_split", default="test", help="split name for CO3D json (predefined only)")
    p.add_argument(
        "--defocus_train_radius_min",
        type=int,
        default=None,
        help="With --defocus_train_radius_max: random integer train blur in [min,max]. Omit both for fixed train=--defocus_radius.",
    )
    p.add_argument("--defocus_train_radius_max", type=int, default=None)
    p.add_argument(
        "--finetune_from_ckpt",
        type=str,
        default=None,
        help="Load model weights only; epoch 0 and fresh optimizer. Ignored if checkpoint-last.pth exists in output_dir.",
    )
    return p


def split_indices(n_items, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = int(round(n_items * val_ratio))
    n_test = int(round(n_items * test_ratio))
    n_train = n_items - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split leaves no training samples")
    return idx[:n_train].tolist(), idx[n_train:n_train + n_val].tolist(), idx[n_train + n_val:].tolist()


def make_loader(subset, batch_size, num_workers, distributed, is_train):
    if distributed:
        sampler = DistributedSampler(subset, shuffle=is_train, drop_last=is_train)
    else:
        sampler = torch.utils.data.RandomSampler(subset) if is_train else torch.utils.data.SequentialSampler(subset)
    return DataLoader(subset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=is_train)


def build_scheduler(optimizer, epochs, warmup_epochs, eta_min):
    wu = int(warmup_epochs)
    if wu <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)), eta_min=eta_min)
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=wu)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, int(epochs) - wu), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[wu])


def _filter_dataset_categories(ds, categories):
    keep = set(categories)
    ds.scenes = {(obj, inst): views for (obj, inst), views in ds.scenes.items() if obj in keep}
    ds.scene_list = list(ds.scenes.keys())
    ds.invalidate = {scene: {} for scene in ds.scene_list}
    return ds


def _build_defocus_dataset(args, split_name):
    common = dict(split=split_name, ROOT=args.co3d_root, resolution=args.resolution, aug_crop=16)
    if args.defocus_train_radius_min is not None:
        return Co3dDefocus(
            args.defocus_radius,
            args.defocus_train_radius_min,
            args.defocus_train_radius_max,
            **common,
        )
    return Co3dDefocus(args.defocus_radius, **common)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        result = loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp)
        loss = result["loss"][0] if isinstance(result["loss"], tuple) else result["loss"]
        optimizer.zero_grad()
        trainable = [p for p in model.parameters() if p.requires_grad]
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=trainable)
        metric_logger.update(loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args, split_name="val"):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"{split_name.capitalize()}:"
    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        result = loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp)
        loss = result["loss"][0] if isinstance(result["loss"], tuple) else result["loss"]
        metric_logger.update(loss=float(loss.detach()))
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@record
def main():
    args = get_args_parser().parse_args()
    if (args.defocus_train_radius_min is None) ^ (args.defocus_train_radius_max is None):
        raise ValueError("Set both --defocus_train_radius_min and --defocus_train_radius_max, or neither.")
    if args.defocus_train_radius_min is not None:
        if args.defocus_train_radius_min < 1 or args.defocus_train_radius_max < 1:
            raise ValueError("defocus train radii must be >= 1")
    misc.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Dataset categories={args.categories}")
    if args.split_strategy == "predefined":
        ds_train = _filter_dataset_categories(_build_defocus_dataset(args, args.train_split), args.categories)
        ds_val = _filter_dataset_categories(_build_defocus_dataset(args, args.val_split), args.categories)
        try:
            ds_test = _filter_dataset_categories(_build_defocus_dataset(args, args.test_split), args.categories)
        except FileNotFoundError:
            ds_test = None
            print(
                f"Warning: selected_seqs_{args.test_split}.json not found. Proceeding without test split."
            )
        print(
            "Split strategy=predefined "
            f"(train={args.train_split}, val={args.val_split}, test={args.test_split})"
        )
    else:
        dataset = _build_defocus_dataset(args, "train")
        dataset = _filter_dataset_categories(dataset, args.categories)
        train_idx, val_idx, test_idx = split_indices(len(dataset), args.val_ratio, args.test_ratio, seed=args.seed)
        ds_train, ds_val = Subset(dataset, train_idx), Subset(dataset, val_idx)
        ds_test = Subset(dataset, test_idx) if len(test_idx) > 0 else None
        print("Split strategy=random (legacy pair-index split)")
    print(f"Split sizes: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test) if ds_test else 0}")

    loader_train = make_loader(ds_train, args.batch_size, args.num_workers, args.distributed, True)
    loader_val = make_loader(ds_val, args.batch_size, args.num_workers, args.distributed, False)
    loader_test = make_loader(ds_test, args.batch_size, args.num_workers, args.distributed, False) if ds_test else None

    model = build_model(
        dust3r_ckpt=args.dust3r_ckpt,
        ifan_repo=args.ifan_repo,
        ifan_ckpt=args.ifan_ckpt,
        device="cpu",
        freeze=args.freeze,
    ).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, gradient_as_bucket_view=False)
        model_without_ddp = model.module

    criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.eta_min)
    loss_scaler = NativeScaler(enabled=bool(args.amp))

    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    best_ckpt = os.path.join(args.output_dir, "checkpoint-best-val.pth")
    start_epoch, best_val = 0, float("inf")
    if os.path.isfile(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        print(f"Resumed from epoch {ckpt['epoch']}")
    elif args.finetune_from_ckpt:
        ckpt = torch.load(args.finetune_from_ckpt, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        print(
            f"Warm-start: loaded model from {args.finetune_from_ckpt} "
            f"(source epoch={ckpt.get('epoch', '?')}, source best_val_loss={ckpt.get('best_val_loss', '?')}). "
            "Training epochs 0..E-1 with fresh optimizer/scheduler in this output_dir."
        )

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
        except Exception as ex:
            print(f"WARNING: W&B init failed ({ex}). Continuing without W&B.")
            use_wandb = False

    for epoch in range(start_epoch, args.epochs):
        if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, loader_train, optimizer, device, epoch, loss_scaler, args)
        val_stats = evaluate(model, criterion, loader_val, device, args, "val")
        test_stats = evaluate(model, criterion, loader_test, device, args, "test") if loader_test else {}
        scheduler.step()
        if misc.is_main_process():
            log_stats = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}, **{f"test_{k}": v for k, v in test_stats.items()}}
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            print(log_stats)
            if use_wandb:
                import wandb
                wandb.log(log_stats, step=epoch)
            ckpt_data = {"epoch": epoch, "model": model_without_ddp.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "args": vars(args), "best_val_loss": best_val}
            torch.save(ckpt_data, last_ckpt)
            if float(val_stats.get("loss", float("inf"))) < best_val:
                best_val = float(val_stats["loss"])
                ckpt_data["best_val_loss"] = best_val
                torch.save(ckpt_data, best_ckpt)
                print(f"New best val: {best_val:.6f}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

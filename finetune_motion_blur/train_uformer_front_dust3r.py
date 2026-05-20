#!/usr/bin/env python3
"""Fine-tune Uformer(front-end) + DUSt3R on motion-blurred CO3D (DDP-ready)."""

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

from finetune_motion_blur.datasets.co3d_motion import Co3dMotion


def get_args_parser():
    p = argparse.ArgumentParser("Uformer + DUSt3R motion fine-tuning")
    p.add_argument("--co3d_root", required=True)
    p.add_argument("--motion_root", required=True)
    p.add_argument("--motion_tag", required=True)
    p.add_argument("--dust3r_ckpt", required=True)
    p.add_argument("--uformer_repo", required=True)
    p.add_argument("--uformer_weights", required=True)
    p.add_argument("--output_dir", default="./output_uformer_motion")
    p.add_argument("--freeze", default="uformer_only", choices=["uformer_only", "all"])
    p.add_argument(
        "--use_peft_lora",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=motion LoRA (DUSt3R encoder + Uformer Linears, decoder frozen). 0=legacy full Uformer front.",
    )
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--cosine_per_iteration",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=CosineAnnealingLR stepped each train batch; T_max=epochs*len(train_loader). 0=cosine per epoch.",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5, help="Default tuned for LoRA motion fine-tuning.")
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--warmup_epochs", type=int, default=0)
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
        help="random: ratio split on one train json. predefined: train/val/test CO3D json names.",
    )
    p.add_argument("--train_split", default="train", help="CO3D selected_seqs_<name>.json (predefined only)")
    p.add_argument("--val_split", default="val", help="predefined only")
    p.add_argument("--test_split", default="test", help="predefined only")
    p.add_argument(
        "--categories",
        nargs="+",
        default=[
            "apple",
            "banana",
            "baseballbat",
            "baseballglove",
            "bicycle",
            "bowl",
            "broccoli",
            "cake",
            "car",
            "carrot",
        ],
        help="Restrict scenes to these CO3D categories (after loading split json).",
    )
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
    return idx[:n_train].tolist(), idx[n_train:n_train + n_val].tolist(), idx[n_train + n_val:].tolist()


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


def build_scheduler(optimizer, epochs, warmup_epochs, eta_min):
    wu = int(warmup_epochs)
    if wu <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)), eta_min=eta_min)
    if wu >= int(epochs):
        return LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=int(epochs))
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=wu)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, int(epochs) - wu), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[wu])


def build_iteration_cosine_scheduler(optimizer, total_optimizer_steps, eta_min):
    """Cosine to eta_min after total_optimizer_steps (one .step() per train batch)."""
    return CosineAnnealingLR(optimizer, T_max=max(1, int(total_optimizer_steps)), eta_min=eta_min)


def _filter_dataset_categories(ds, categories):
    keep = set(categories)
    ds.scenes = {(obj, inst): views for (obj, inst), views in ds.scenes.items() if obj in keep}
    ds.scene_list = list(ds.scenes.keys())
    ds.invalidate = {scene: {} for scene in ds.scene_list}
    return ds


def _build_motion_dataset(args, split_name):
    return Co3dMotion(
        motion_root=args.motion_root,
        motion_tag=args.motion_tag,
        split=split_name,
        ROOT=args.co3d_root,
        resolution=args.resolution,
        aug_crop=16,
    )


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args, scheduler=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    step_each_batch = bool(args.cosine_per_iteration) and scheduler is not None
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
        if step_each_batch:
            scheduler.step()

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
    if args.cosine_per_iteration and int(args.warmup_epochs) > 0:
        print("NOTE: --warmup_epochs ignored when --cosine_per_iteration=1 (single cosine over all train steps).")
    misc.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"motion_tag={args.motion_tag}, categories={args.categories}")
    if args.split_strategy == "predefined":
        ds_train = _filter_dataset_categories(_build_motion_dataset(args, args.train_split), args.categories)
        ds_val = _filter_dataset_categories(_build_motion_dataset(args, args.val_split), args.categories)
        try:
            ds_test = _filter_dataset_categories(_build_motion_dataset(args, args.test_split), args.categories)
        except FileNotFoundError:
            ds_test = None
            print(
                f"Warning: selected_seqs_{args.test_split}.json not found. Proceeding without test split."
            )
        if ds_test is not None and len(ds_test) == 0:
            ds_test = None
            print("Warning: test split has no samples (empty JSON); skipping test evaluation.")
        print(
            "Split strategy=predefined "
            f"(train={args.train_split}, val={args.val_split}, test={args.test_split})"
        )
    else:
        dataset = _build_motion_dataset(args, "train")
        dataset = _filter_dataset_categories(dataset, args.categories)
        print(f"Dataset: {len(dataset)} pairs (motion_tag={args.motion_tag})")
        train_idx, val_idx, test_idx = split_indices(
            len(dataset), args.val_ratio, args.test_ratio, seed=args.seed
        )
        ds_train = Subset(dataset, train_idx)
        ds_val = Subset(dataset, val_idx)
        ds_test = Subset(dataset, test_idx) if len(test_idx) > 0 else None
        print("Split strategy=random (legacy pair-index split)")
    print(f"Split sizes: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test) if ds_test else 0}")

    loader_train = make_loader(ds_train, args.batch_size, args.num_workers, args.distributed, is_train=True)
    loader_val = make_loader(ds_val, args.batch_size, args.num_workers, args.distributed, is_train=False)
    loader_test = (
        make_loader(ds_test, args.batch_size, args.num_workers, args.distributed, is_train=False)
        if ds_test else None
    )

    if args.use_peft_lora:
        from finetune_motion_blur.model_uformer_motion_lora import build_model as build_motion_lora

        model = build_motion_lora(
            dust3r_ckpt=args.dust3r_ckpt,
            uformer_repo=args.uformer_repo,
            uformer_weights=args.uformer_weights,
            device="cpu",
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        ).to(device)
        print(f"Model: motion PEFT LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    else:
        from finetune_noise.model import build_model as build_legacy

        model = build_legacy(
            dust3r_ckpt=args.dust3r_ckpt,
            uformer_repo=args.uformer_repo,
            uformer_weights=args.uformer_weights,
            device="cpu",
            freeze=args.freeze,
        ).to(device)
        print(f"Model: legacy freeze={args.freeze}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=bool(args.use_peft_lora),
            gradient_as_bucket_view=False,
        )
        model_without_ddp = model.module

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable")

    criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters (check LoRA / freeze settings).")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(loader_train)
    total_train_steps = max(1, int(args.epochs) * steps_per_epoch)
    if args.cosine_per_iteration:
        scheduler = build_iteration_cosine_scheduler(optimizer, total_train_steps, args.eta_min)
        if misc.is_main_process():
            print(
                f"LR scheduler: CosineAnnealingLR per batch, T_max={total_train_steps} "
                f"(epochs={args.epochs} x steps/epoch={steps_per_epoch}), eta_min={args.eta_min}"
            )
    else:
        scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.eta_min)
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
                project=args.wandb_project, name=args.wandb_run_name,
                config=vars(args), resume="allow", dir=args.output_dir
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

        sched_for_batch = scheduler if args.cosine_per_iteration else None
        train_stats = train_one_epoch(
            model, criterion, loader_train, optimizer, device, epoch, loss_scaler, args, scheduler=sched_for_batch
        )
        val_stats = evaluate(model, criterion, loader_val, device, args, "val")
        test_stats = evaluate(model, criterion, loader_test, device, args, "test") if loader_test else {}
        if not args.cosine_per_iteration:
            scheduler.step()

        if misc.is_main_process():
            log_stats = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
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

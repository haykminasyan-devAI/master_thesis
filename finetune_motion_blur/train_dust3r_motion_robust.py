#!/usr/bin/env python3
"""
DUSt3R motion blur robustness fine-tuning on CO3D with category-level generalization.

Key points:
- No DUSt3R architecture modification
- On-the-fly linear motion blur augmentation (train)
- Validation by condition (clean-clean / blur-blur / clean-blur)
- Final test only once after training completes
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
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
from dust3r.inference import loss_of_one_batch
from dust3r.losses import ConfLoss, L21, Regr3D
from dust3r.model import load_model

from finetune_motion_blur.datasets.co3d_motion_robust import Co3dMotionRobust

VAL_CATEGORIES = ["laptop", "tv", "handbag", "bicycle", "car", "cake"]
TEST_CATEGORIES = ["cup", "couch", "bottle", "teddybear", "donut", "toytrain"]
EVAL_CONDITIONS = ["clean-clean", "blur-blur", "clean-blur"]


def get_args_parser():
    p = argparse.ArgumentParser("DUSt3R motion robustness fine-tuning")
    p.add_argument("--co3d_root", required=True)
    p.add_argument("--dust3r_ckpt", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--symmetrize_batch", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.05)

    p.add_argument("--lr_main", type=float, default=1e-5, help="LR for decoder/head + selected encoder blocks")
    p.add_argument("--lr_encoder_low", type=float, default=1e-6, help="LR for partially unfrozen encoder")
    p.add_argument("--encoder_blocks_train", type=int, default=4, help="How many last encoder blocks to train initially")
    p.add_argument("--unfreeze_full_encoder_epoch", type=int, default=-1, help="Epoch to unfreeze full encoder at lr_encoder_low")

    p.add_argument("--train_clean_only", type=int, default=0, choices=[0, 1],
                   help="1 = no blur augmentation in train (baseline).")
    p.add_argument("--kernel_min", type=int, default=3)
    p.add_argument("--kernel_max", type=int, default=9)

    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://")

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p


def _load_category_map(co3d_root: str, split_name: str = "train") -> dict:
    path = os.path.join(co3d_root, f"selected_seqs_{split_name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing split json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _available_categories(co3d_root: str) -> list:
    d = _load_category_map(co3d_root, split_name="train")
    return sorted([k for k, v in d.items() if isinstance(v, dict) and len(v) > 0])


def _filter_dataset_categories(ds, categories):
    keep = set(categories)
    ds.scenes = {(obj, inst): views for (obj, inst), views in ds.scenes.items() if obj in keep}
    ds.scene_list = list(ds.scenes.keys())
    ds.invalidate = {scene: {} for scene in ds.scene_list}
    return ds


def _build_dataset(args, split: str, categories: list, train_mode: bool, eval_condition: str):
    ds = Co3dMotionRobust(
        split=split,
        ROOT=args.co3d_root,
        resolution=args.resolution,
        aug_crop=16,
        train_mode=train_mode,
        eval_condition=eval_condition,
        kernel_min=args.kernel_min,
        kernel_max=args.kernel_max,
    )
    return _filter_dataset_categories(ds, categories)


def _make_loader(dataset, batch_size, num_workers, distributed, is_train):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train, drop_last=is_train)
    else:
        sampler = (
            torch.utils.data.RandomSampler(dataset)
            if is_train else torch.utils.data.SequentialSampler(dataset)
        )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def configure_trainable_params(model, encoder_blocks_train: int, full_encoder: bool):
    # Freeze everything first.
    _set_requires_grad(model, False)

    # Always train decoder + prediction heads.
    for m in [model.decoder_embed, model.dec_blocks, model.dec_blocks2, model.dec_norm, model.head1, model.head2]:
        _set_requires_grad(m, True)

    # Encoder strategy.
    if full_encoder:
        for m in [model.patch_embed, model.enc_blocks, model.enc_norm]:
            _set_requires_grad(m, True)
    else:
        n = max(0, int(encoder_blocks_train))
        if n > 0:
            for blk in list(model.enc_blocks)[-n:]:
                _set_requires_grad(blk, True)
            _set_requires_grad(model.enc_norm, True)


def build_optimizer(model, lr_main: float, lr_encoder_low: float):
    enc_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("enc_blocks") or name.startswith("patch_embed") or name.startswith("enc_norm"):
            enc_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr_main})
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_encoder_low})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.05)


def _mean_lr(optimizer):
    lrs = [g["lr"] for g in optimizer.param_groups]
    return float(sum(lrs) / max(1, len(lrs)))


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        out = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp
        )
        loss = out["loss"]
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
        metric_logger.update(lr=_mean_lr(optimizer))

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args, split_name: str):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"{split_name}:"

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        out = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp
        )
        loss = out["loss"]
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

    cats = _available_categories(args.co3d_root)
    val_set = set(VAL_CATEGORIES)
    test_set = set(TEST_CATEGORIES)
    train_categories = [c for c in cats if c not in val_set and c not in test_set]
    val_categories = [c for c in VAL_CATEGORIES if c in cats]
    test_categories = [c for c in TEST_CATEGORIES if c in cats]
    if not train_categories or not val_categories or not test_categories:
        raise RuntimeError(
            f"Category split invalid. train={train_categories}, val={val_categories}, test={test_categories}"
        )

    if misc.is_main_process():
        print("Category split:")
        print(f"  train ({len(train_categories)}): {train_categories}")
        print(f"  val   ({len(val_categories)}): {val_categories}")
        print(f"  test  ({len(test_categories)}): {test_categories}")

    train_eval_condition = "clean-clean" if bool(args.train_clean_only) else "train-policy"
    ds_train = _build_dataset(
        args, split="train", categories=train_categories,
        train_mode=(not bool(args.train_clean_only)), eval_condition="clean-clean"
    )
    if bool(args.train_clean_only):
        ds_train.train_mode = False
        ds_train.eval_condition = "clean-clean"
    loaders_val = {}
    for cond in EVAL_CONDITIONS:
        ds_val = _build_dataset(args, split="train", categories=val_categories, train_mode=False, eval_condition=cond)
        loaders_val[cond] = _make_loader(ds_val, args.batch_size, args.num_workers, args.distributed, is_train=False)

    # Final test split only; we evaluate once at the end.
    loaders_test = {}
    for cond in EVAL_CONDITIONS:
        ds_test = _build_dataset(args, split="train", categories=test_categories, train_mode=False, eval_condition=cond)
        loaders_test[cond] = _make_loader(ds_test, args.batch_size, args.num_workers, args.distributed, is_train=False)

    loader_train = _make_loader(ds_train, args.batch_size, args.num_workers, args.distributed, is_train=True)
    if misc.is_main_process():
        print(f"Train mode={train_eval_condition}, train pairs={len(ds_train)}")
        for cond in EVAL_CONDITIONS:
            print(f"Val {cond} pairs={len(loaders_val[cond].dataset)} | Test {cond} pairs={len(loaders_test[cond].dataset)}")

    model = load_model(args.dust3r_ckpt, device="cpu").to(device)
    configure_trainable_params(model, args.encoder_blocks_train, full_encoder=False)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False, gradient_as_bucket_view=False
        )
        model_without_ddp = model.module

    optimizer = build_optimizer(model_without_ddp, args.lr_main, args.lr_encoder_low)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs) * len(loader_train)), eta_min=1e-6)
    loss_scaler = NativeScaler(enabled=bool(args.amp))
    criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if misc.is_main_process():
        print(f"Parameters: {n_total:,} total, {n_trainable:,} trainable")

    best_val = float("inf")
    best_epoch = -1
    best_ckpt = os.path.join(args.output_dir, "checkpoint-best-val.pth")
    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    t0 = time.time()

    use_wandb = bool(args.wandb_project) and misc.is_main_process()
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), dir=args.output_dir)
        except Exception as ex:
            print(f"WARNING: wandb disabled ({ex})")
            use_wandb = False

    for epoch in range(args.epochs):
        if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        # Optional late encoder unfreeze.
        if args.unfreeze_full_encoder_epoch >= 0 and epoch == int(args.unfreeze_full_encoder_epoch):
            if misc.is_main_process():
                print(f"Unfreezing full encoder at epoch {epoch} (low LR={args.lr_encoder_low})")
            configure_trainable_params(model_without_ddp, args.encoder_blocks_train, full_encoder=True)
            optimizer = build_optimizer(model_without_ddp, args.lr_main, args.lr_encoder_low)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, (args.epochs - epoch) * len(loader_train)), eta_min=1e-6)

        train_stats = train_one_epoch(model, criterion, loader_train, optimizer, device, epoch, loss_scaler, args)
        val_cond_stats = {}
        for cond, loader in loaders_val.items():
            stats = evaluate(model, criterion, loader, device, args, split_name=f"val-{cond}")
            val_cond_stats[cond] = stats

        # Best model selection on val clean-blur (target robustness condition).
        cur_val = float(val_cond_stats["clean-blur"]["loss"])
        if cur_val < best_val:
            best_val = cur_val
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val,
                    "args": vars(args),
                },
                best_ckpt,
            )
            if misc.is_main_process():
                print(f"  -> New best val(clean-blur): {best_val:.6f} at epoch {epoch}")

        log_stats = {
            "epoch": epoch,
            "lr": _mean_lr(optimizer),
            "train_loss": train_stats["loss"],
            "val_clean_clean_loss": val_cond_stats["clean-clean"]["loss"],
            "val_blur_blur_loss": val_cond_stats["blur-blur"]["loss"],
            "val_clean_blur_loss": val_cond_stats["clean-blur"]["loss"],
            "best_val_clean_blur_loss": best_val,
        }
        if misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            print(f"Epoch {epoch}: {log_stats}")
            if use_wandb:
                import wandb
                wandb.log(log_stats, step=epoch)

        # Save last checkpoint every epoch.
        if misc.is_main_process():
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val,
                    "args": vars(args),
                },
                last_ckpt,
            )
        scheduler.step()

    # Final test protocol: evaluate once after training with best-val model.
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)

    test_cond_stats = {}
    for cond, loader in loaders_test.items():
        test_cond_stats[cond] = evaluate(model, criterion, loader, device, args, split_name=f"test-{cond}")

    final_stats = {
        "best_epoch": best_epoch,
        "best_val_clean_blur_loss": best_val,
        "test_clean_clean_loss": test_cond_stats["clean-clean"]["loss"],
        "test_blur_blur_loss": test_cond_stats["blur-blur"]["loss"],
        "test_clean_blur_loss": test_cond_stats["clean-blur"]["loss"],
        "elapsed_hours": (time.time() - t0) / 3600.0,
    }
    if misc.is_main_process():
        with open(os.path.join(args.output_dir, "final_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2)
        print("Final test metrics:", final_stats)
        if use_wandb:
            import wandb
            wandb.log(final_stats)
            wandb.finish()


if __name__ == "__main__":
    main()

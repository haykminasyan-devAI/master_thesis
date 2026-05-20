#!/usr/bin/env python3
"""
KD: frozen URetinex-Net (teacher) -> trainable Zero-DCE (student) on CO3D + synthetic low-light.
Distributed data parallel (torchrun).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from dataset_co3d import Co3dLowLightKD, collate_stack
from losses import L_TV, L_exp, L_spa, VGGPerceptualLoss, charbonnier
from student_zerodce import load_zerodce_student
from teacher_uretinex import load_uretinex_teacher

TEN_CATEGORIES = [
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


def init_dist():
    if "RANK" not in os.environ:
        return 0, 0, 1, False
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d_root", type=str, required=True)
    ap.add_argument("--split_train", type=str, default="train_10cat8_7v1")
    ap.add_argument("--split_val", type=str, default="val_10cat8_7v1")
    ap.add_argument("--uretinex_root", type=str, required=True)
    ap.add_argument("--zerodce_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--categories", type=str, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16, help="Per GPU")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--eta_min", type=float, default=1e-7)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--uretinex_ratio", type=float, default=5.0)
    ap.add_argument("--lambda_char", type=float, default=1.0)
    ap.add_argument("--lambda_vgg", type=float, default=0.5)
    ap.add_argument("--lambda_spa", type=float, default=1.0)
    ap.add_argument("--lambda_exp", type=float, default=2.0)
    ap.add_argument("--lambda_tv", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_freq", type=int, default=5)
    args = ap.parse_args()

    rank, local_rank, world, distributed = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    cats = args.categories or TEN_CATEGORIES

    ds_tr = Co3dLowLightKD(
        args.co3d_root,
        args.split_train,
        cats,
        crop_size=args.crop_size,
        is_train=True,
        seed=args.seed,
    )
    ds_va = Co3dLowLightKD(
        args.co3d_root,
        args.split_val,
        cats,
        crop_size=args.crop_size,
        is_train=False,
        seed=args.seed + 1,
    )

    if is_main:
        print(f"[data] train images={len(ds_tr)} val images={len(ds_va)}", flush=True)

    sampler_tr = DistributedSampler(ds_tr, shuffle=True, seed=args.seed) if distributed else None
    sampler_va = DistributedSampler(ds_va, shuffle=False) if distributed else None

    ld_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=(sampler_tr is None),
        sampler=sampler_tr,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,
    )
    ld_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler_va,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_stack,
    )

    student = load_zerodce_student(args.zerodce_root).to(device)
    teacher = load_uretinex_teacher(args.uretinex_root, device, ratio=args.uretinex_ratio)

    if distributed:
        student = DDP(student, device_ids=[local_rank])

    optim = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.eta_min)

    vgg_loss = VGGPerceptualLoss(device)
    l_spa = L_spa(device)
    l_exp = L_exp(16, 0.6, device)
    l_tv = L_TV(weight=1.0)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    def student_state_dict():
        return student.module.state_dict() if hasattr(student, "module") else student.state_dict()

    def run_epoch(loader, train_mode: bool):
        if train_mode:
            student.train()
        else:
            student.eval()
        tot = 0.0
        n = 0
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for batch in loader:
                low = batch.to(device, non_blocking=True)
                with torch.no_grad():
                    t_out = teacher(low)

                if train_mode:
                    optim.zero_grad(set_to_none=True)

                _, enh, A = student(low)

                lc = charbonnier(enh, t_out)
                lv = vgg_loss(enh, t_out)
                ls = torch.mean(l_spa(enh, low))
                le = l_exp(enh)
                lt = l_tv(A)

                loss = (
                    args.lambda_char * lc
                    + args.lambda_vgg * lv
                    + args.lambda_spa * ls
                    + args.lambda_exp * le
                    + args.lambda_tv * lt
                )

                if train_mode:
                    loss.backward()
                    optim.step()

                tot += float(loss.detach())
                n += 1

        return tot / max(1, n)

    for epoch in range(args.epochs):
        if sampler_tr is not None:
            sampler_tr.set_epoch(epoch)

        tr_loss = run_epoch(ld_tr, True)
        va_loss = run_epoch(ld_va, False)
        sched.step()

        if is_main:
            row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "lr": optim.param_groups[0]["lr"]}
            print(row, flush=True)
            with open(out_dir / "log.txt", "a") as f:
                f.write(json.dumps(row) + "\n")
            if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                ck = {
                    "epoch": epoch,
                    "state_dict": student_state_dict(),
                    "args": vars(args),
                }
                torch.save(ck, out_dir / f"student_epoch_{epoch:04d}.pth")
                torch.save(ck, out_dir / "student_last.pth")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

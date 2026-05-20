#!/usr/bin/env python3
"""Stage-1 KD: Base DUSt3R encoder (teacher) -> DUSt3R encoder LoRA (student) on dark images."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
try:
    import wandb
except Exception:  # noqa: BLE001
    wandb = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR / "dust3r", PROJECT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.model import load_model as load_dust3r_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def init_dist():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world)
    return True, rank, world, local_rank


def resolve_split_json(root: Path, split: str) -> Path:
    candidates = [root / f"selected_seqs_{split}.json"]
    if "_" in split:
        candidates.append(root / f"selected_seqs_{split.split('_', 1)[0]}.json")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Missing split json. Looked for:\n - " + "\n - ".join(map(str, candidates)))


class Co3dStrideFrameDataset(Dataset):
    """Sequence-level split by selected_seqs_*.json, with frame subsampling by stride."""

    def __init__(
        self,
        root: str,
        split: str,
        stride: int = 5,
        gamma_values=(1.5, 2.2),
        image_size: int = 224,
    ):
        self.root = Path(root)
        self.stride = max(1, int(stride))
        self.gamma_values = tuple(float(g) for g in gamma_values)
        self.image_size = int(image_size)
        split_json = resolve_split_json(self.root, split)
        data = json.loads(split_json.read_text())
        self.samples: list[str] = []
        for cat, seqs in data.items():
            if not isinstance(seqs, dict):
                continue
            for seq_id in sorted(seqs.keys()):
                img_dir = self.root / cat / seq_id / "images"
                if not img_dir.is_dir():
                    continue
                frames = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                if not frames:
                    continue
                self.samples.extend(str(p) for p in frames[:: self.stride])
        if not self.samples:
            raise RuntimeError(f"No frames found for split={split} stride={self.stride}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.image_size > 0:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            rgb = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to read image: {path}") from e

        gamma = random.choice(self.gamma_values)
        dark = np.power(rgb, gamma).astype(np.float32)

        clean_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        dark_t = torch.from_numpy(dark).permute(2, 0, 1).contiguous()
        return clean_t, dark_t


def collate_pad(batch):
    cleans, darks = zip(*batch)
    b = len(batch)
    c = 3
    h = max(x.shape[1] for x in cleans)
    w = max(x.shape[2] for x in cleans)
    h = int(math.ceil(h / 16.0) * 16)
    w = int(math.ceil(w / 16.0) * 16)
    clean_pad = torch.zeros(b, c, h, w, dtype=torch.float32)
    dark_pad = torch.zeros(b, c, h, w, dtype=torch.float32)
    true_shape = torch.zeros(b, 2, dtype=torch.long)
    for i, (cl, dk) in enumerate(zip(cleans, darks)):
        _, hh, ww = cl.shape
        clean_pad[i, :, :hh, :ww] = cl
        dark_pad[i, :, :hh, :ww] = dk
        true_shape[i] = torch.tensor([hh, ww], dtype=torch.long)
    return clean_pad, dark_pad, true_shape


def load_base_teacher(teacher_ckpt: str):
    if not Path(teacher_ckpt).is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
    teacher = load_dust3r_model(teacher_ckpt, device="cpu", verbose=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def get_encoder_features(model, img: torch.Tensor, true_shape: torch.Tensor):
    # DUSt3R-like models expose _encode_image, but signature varies across versions.
    try:
        out = model._encode_image(img, true_shape=true_shape, key_padding_mask=None)  # noqa: SLF001
    except TypeError:
        out = model._encode_image(img, true_shape=true_shape)  # noqa: SLF001
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats


def build_student_with_encoder_lora(dust3r_ckpt: str, lora_r: int = 16, lora_alpha: int = 32):
    student = load_dust3r_model(dust3r_ckpt, device="cpu", verbose=False)

    # Freeze everything first (decoder + base encoder included).
    for p in student.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Encoder self-attention in this DUSt3R family uses `attn.qkv`.
        # `projq/projk/projv` belong to decoder cross-attention blocks.
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
    )
    student = get_peft_model(student, cfg)

    # Keep only LoRA modules in encoder trainable.
    # PEFT naming can differ across wrappers, so match both raw and wrapped encoder names.
    for name, p in student.named_parameters():
        is_lora = "lora_" in name
        in_encoder = ("enc_blocks" in name) or ("module.model.enc_blocks" in name) or (".model.enc_blocks" in name)
        if is_lora and in_encoder:
            p.requires_grad = True
        else:
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if n_trainable == 0:
        # Fallback: if encoder name pattern changes, keep all LoRA params trainable
        # rather than crashing DDP with zero trainable parameters.
        for name, p in student.named_parameters():
            p.requires_grad = "lora_" in name
    return student


def enable_encoder_qkv_fallback(student) -> int:
    # Last-resort fallback when adapter path is disconnected from encoder features:
    # train encoder q/k/v projection weights directly.
    n_enabled = 0
    for name, p in student.named_parameters():
        in_enc = "enc_blocks" in name
        is_qkv = (".projq." in name) or (".projk." in name) or (".projv." in name)
        if in_enc and is_qkv:
            if not p.requires_grad:
                p.requires_grad = True
            n_enabled += p.numel()
    return n_enabled


def build_lr_lambda(total_steps: int, warmup_steps: int, min_factor: float):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(warmup_steps))
    min_factor = float(min_factor)

    def _fn(step: int):
        s = min(step, total_steps)
        if s < warmup_steps:
            return float(s + 1) / float(warmup_steps)
        prog = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_factor + (1.0 - min_factor) * cosine

    return _fn


def main():
    ap = argparse.ArgumentParser("Stage-1 encoder KD: Base DUSt3R teacher -> DUSt3R encoder LoRA student")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_train", default="train_10cat8")
    ap.add_argument("--split_val", default="val_10cat8")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)  # per GPU
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--wandb_entity", default="")
    ap.add_argument("--wandb_run_name", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    distributed, rank, world, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ds_tr = Co3dStrideFrameDataset(
        args.data_root, args.split_train, stride=args.stride, image_size=args.image_size
    )
    ds_va = Co3dStrideFrameDataset(
        args.data_root, args.split_val, stride=args.stride, image_size=args.image_size
    )
    smp_tr = DistributedSampler(ds_tr, shuffle=True) if distributed else None
    smp_va = DistributedSampler(ds_va, shuffle=False) if distributed else None

    ld_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        sampler=smp_tr,
        shuffle=(smp_tr is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_pad,
    )
    ld_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        sampler=smp_va,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_pad,
    )

    teacher = load_base_teacher(args.teacher_ckpt).to(device).eval()
    student = build_student_with_encoder_lora(args.dust3r_ckpt, args.lora_r, args.lora_alpha).to(device)
    student.train()

    # Probe whether encoder features are connected to trainable params.
    # Some PEFT/model combinations keep trainable params, but the specific call path
    # to _encode_image may bypass them.
    with torch.enable_grad():
        probe_img = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float32)
        probe_shape = torch.tensor([[224, 224]], device=device, dtype=torch.long)
        probe_feat = get_encoder_features(student, probe_img, probe_shape)
        if not probe_feat.requires_grad:
            enabled = enable_encoder_qkv_fallback(student)
            if rank == 0:
                print(
                    "[startup] warning: student encoder features had no grad path; "
                    f"enabled direct encoder qkv fallback params={enabled:,}",
                    flush=True,
                )

    if distributed:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)

    trainable = [p for p in student.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in student.parameters())
    if rank == 0:
        print(
            f"[startup] device={device} distributed={distributed} world={world} "
            f"train={len(ds_tr)} val={len(ds_va)} stride={args.stride}",
            flush=True,
        )
        print(f"[startup] student params total={n_total:,} trainable={n_trainable:,}", flush=True)

    use_wandb = rank == 0 and bool(args.wandb_project)
    if use_wandb and wandb is None:
        raise RuntimeError("W&B requested but wandb is not installed. Please install wandb in your env.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=(args.wandb_run_name or None),
            config=vars(args),
        )

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(ld_tr))
    warmup_steps = max(1, int(0.05 * total_steps))
    min_factor = args.lr_min / args.lr
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=build_lr_lambda(total_steps, warmup_steps, min_factor))
    mse = nn.MSELoss()

    step_global = 0
    best_val = float("inf")
    log_path = Path(args.output_dir) / "log.txt"
    ckpt_last = Path(args.output_dir) / "student_lora_last.pth"
    ckpt_best = Path(args.output_dir) / "student_lora_best.pth"

    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    for ep in range(args.epochs):
        if smp_tr is not None:
            smp_tr.set_epoch(ep)
        student.train()
        tr_loss = 0.0
        tr_n = 0
        for step, (img_clean, img_dark, true_shape) in enumerate(ld_tr):
            img_clean = img_clean.to(device, non_blocking=True)
            img_dark = img_dark.to(device, non_blocking=True)
            true_shape = true_shape.to(device, non_blocking=True)

            with torch.no_grad():
                t_feat = get_encoder_features(teacher, img_clean, true_shape)
            s_feat = get_encoder_features(_unwrap(student), img_dark, true_shape)

            loss = mse(s_feat, t_feat)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()

            lv = float(loss.detach())
            tr_loss += lv
            tr_n += 1
            step_global += 1
            if rank == 0 and (step_global % max(1, args.log_interval) == 0):
                print(
                    f"[train] ep={ep} step={step} gstep={step_global} "
                    f"loss={lv:.6f} lr={optim.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss_step": lv,
                            "train/lr": optim.param_groups[0]["lr"],
                            "epoch": ep,
                            "global_step": step_global,
                        },
                        step=step_global,
                    )

        # val
        student.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for vstep, (img_clean, img_dark, true_shape) in enumerate(ld_va):
                img_clean = img_clean.to(device, non_blocking=True)
                img_dark = img_dark.to(device, non_blocking=True)
                true_shape = true_shape.to(device, non_blocking=True)
                t_feat = get_encoder_features(teacher, img_clean, true_shape)
                s_feat = get_encoder_features(_unwrap(student), img_dark, true_shape)
                loss = mse(s_feat, t_feat)
                va_loss += float(loss.detach())
                va_n += 1

        tr_avg = tr_loss / max(1, tr_n)
        va_avg = va_loss / max(1, va_n)
        row = {"epoch": ep, "train_loss": tr_avg, "val_loss": va_avg, "lr": optim.param_groups[0]["lr"]}

        if rank == 0:
            print(row, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if use_wandb:
                wandb.log(
                    {
                        "train/loss_epoch": tr_avg,
                        "val/loss_epoch": va_avg,
                        "train/lr_epoch_end": optim.param_groups[0]["lr"],
                        "epoch": ep,
                    },
                    step=step_global,
                )

            state = _unwrap(student).state_dict()
            torch.save({"epoch": ep, "state_dict": state, "args": vars(args)}, ckpt_last)
            if va_avg < best_val:
                best_val = va_avg
                torch.save(
                    {"epoch": ep, "state_dict": state, "args": vars(args), "best_val": best_val},
                    ckpt_best,
                )
                print(f"[checkpoint] new best val={best_val:.6f} at epoch={ep}", flush=True)
                if use_wandb:
                    wandb.log({"val/best_loss": best_val, "best_epoch": ep}, step=step_global)

    if use_wandb:
        wandb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

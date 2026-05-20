#!/usr/bin/env python3
"""KD training for lightweight restoration front-end (motion/defocus) with frozen Restormer teacher."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from runpy import run_path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def motion_kernel_horizontal(size: int) -> np.ndarray:
    """Horizontal uniform motion PSF (size x size, stripe along middle row)."""
    s = int(size)
    if s % 2 == 0:
        s += 1
    k = np.zeros((s, s), dtype=np.float32)
    k[s // 2, :] = 1.0
    k /= k.sum()
    return k


def defocus_kernel_disk(radius: int) -> np.ndarray:
    r = int(radius)
    s = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y) <= (r * r)
    k = np.zeros((s, s), dtype=np.float32)
    k[mask] = 1.0
    k /= k.sum()
    return k


def resolve_split_json(root: Path, split: str) -> Path:
    candidates = [root / f"selected_seqs_{split}.json"]
    if "_" in split:
        candidates.append(root / f"selected_seqs_{split.split('_', 1)[0]}.json")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Missing split json. Looked for:\n - " + "\n - ".join(map(str, candidates)))


class Co3dFrameDataset(Dataset):
    """CO3D frames with a single corruption type per dataset (motion XOR defocus)."""

    def __init__(
        self,
        root: str,
        split: str,
        categories: list[str],
        blur_kind: str,
        motion_kernel_size: int = 35,
        defocus_radius: int = 9,
    ):
        if blur_kind not in ("motion", "defocus"):
            raise ValueError("blur_kind must be 'motion' or 'defocus'")
        self.blur_kind = blur_kind
        self.motion_k = motion_kernel_horizontal(motion_kernel_size)
        self.defocus_k = defocus_kernel_disk(defocus_radius)

        self.root = Path(root)
        split_json = resolve_split_json(self.root, split)
        data = json.loads(split_json.read_text())
        keep = set(categories)
        self.paths: list[str] = []
        for obj, seqs in data.items():
            if obj not in keep or not isinstance(seqs, dict):
                continue
            for sid in seqs.keys():
                img_dir = self.root / obj / sid / "images"
                if not img_dir.is_dir():
                    continue
                for p in sorted(img_dir.iterdir()):
                    if p.suffix.lower() in IMAGE_EXTS:
                        self.paths.append(str(p))
        if not self.paths:
            raise RuntimeError(f"No frames found in split {split_json}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        clean_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if clean_bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")

        if self.blur_kind == "motion":
            blur_type = 0
            corr_bgr = cv2.filter2D(clean_bgr, -1, self.motion_k)
        else:
            blur_type = 1
            corr_bgr = cv2.filter2D(clean_bgr, -1, self.defocus_k)

        clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        corr = cv2.cvtColor(corr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        clean_t = torch.from_numpy(clean).permute(2, 0, 1).contiguous()
        corr_t = torch.from_numpy(corr).permute(2, 0, 1).contiguous()
        return clean_t, corr_t, blur_type


def collate_full_images(batch):
    cleans, corrs, blur_types = zip(*batch)
    b = len(batch)
    c = 3
    h = max(x.shape[1] for x in corrs)
    w = max(x.shape[2] for x in corrs)
    h = int(math.ceil(h / 8.0) * 8)
    w = int(math.ceil(w / 8.0) * 8)

    clean_pad = torch.zeros(b, c, h, w, dtype=torch.float32)
    corr_pad = torch.zeros(b, c, h, w, dtype=torch.float32)
    mask = torch.zeros(b, 1, h, w, dtype=torch.float32)
    for i, (cl, co) in enumerate(zip(cleans, corrs)):
        _, hh, ww = cl.shape
        clean_pad[i, :, :hh, :ww] = cl
        corr_pad[i, :, :hh, :ww] = co
        mask[i, :, :hh, :ww] = 1.0
    blur_types = torch.tensor(blur_types, dtype=torch.long)
    return clean_pad, corr_pad, mask, blur_types


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


class SmallTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4, win: int = 8):
        super().__init__()
        self.channels = channels
        self.win = win
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x):
        # x: B,C,H,W
        b, c, h, w = x.shape
        win = self.win
        ph = (win - h % win) % win
        pw = (win - w % win) % win
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        b, c, hp, wp = x.shape

        xw = x.view(b, c, hp // win, win, wp // win, win).permute(0, 2, 4, 3, 5, 1).contiguous()
        xw = xw.view(-1, win * win, c)

        y = self.norm1(xw)
        y, _ = self.attn(y, y, y, need_weights=False)
        xw = xw + y
        xw = xw + self.ffn(self.norm2(xw))

        xw = xw.view(b, hp // win, wp // win, win, win, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = xw.view(b, c, hp, wp)
        if ph or pw:
            out = out[:, :, :h, :w]
        return out


class StudentRestorationFrontEnd(nn.Module):
    """3x3 stem -> 2-stage encoder -> 2 transformer bottleneck blocks -> 2-stage decoder."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 1, 1)

        self.enc1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)

        self.enc2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down2 = nn.Conv2d(64, 128, 3, 2, 1)

        self.bottleneck = nn.Sequential(
            SmallTransformerBlock(128, heads=4, win=8),
            SmallTransformerBlock(128, heads=4, win=8),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
        )
        self.dec2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, 64, 1),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),
        )
        self.dec1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 32, 1),
        )
        self.out = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.stem(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)
        return torch.clamp(out, 0.0, 1.0)


class RestormerTeacher(nn.Module):
    """Frozen Restormer teacher with motion/defocus branches."""

    def __init__(self, restormer_root: str):
        super().__init__()
        rr = Path(restormer_root)
        arch_path = rr / "basicsr/models/archs/restormer_arch.py"
        if not arch_path.is_file():
            raise FileNotFoundError(f"Restormer arch file not found: {arch_path}")
        arch = run_path(str(arch_path))["Restormer"]

        # Keep teacher architecture exactly aligned with Restormer demo.py defaults.
        # Motion_Deblurring and Single_Image_Defocus_Deblurring both use WithBias LN and 3-channel io.
        base_params = dict(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,
        )
        self.motion = arch(**base_params)
        self.defocus = arch(**base_params)

        m_path = rr / "Motion_Deblurring/pretrained_models/motion_deblurring.pth"
        d_path = rr / "Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth"
        if not m_path.is_file():
            raise FileNotFoundError(f"Missing motion teacher checkpoint: {m_path}")
        if not d_path.is_file():
            raise FileNotFoundError(f"Missing defocus teacher checkpoint: {d_path}")

        ck_m = torch.load(m_path, map_location="cpu")
        ck_d = torch.load(d_path, map_location="cpu")
        if "params" not in ck_m or "params" not in ck_d:
            raise KeyError("Restormer checkpoint missing 'params' key; expected official pretrained format.")
        self.motion.load_state_dict(ck_m["params"], strict=True)
        self.defocus.load_state_dict(ck_d["params"], strict=True)

        for m in (self.motion, self.defocus):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor, blur_type: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        idx_m = (blur_type == 0).nonzero(as_tuple=True)[0]
        idx_d = (blur_type == 1).nonzero(as_tuple=True)[0]
        if len(idx_m) > 0:
            out[idx_m] = torch.clamp(self.motion(x[idx_m]), 0.0, 1.0)
        if len(idx_d) > 0:
            out[idx_d] = torch.clamp(self.defocus(x[idx_d]), 0.0, 1.0)
        return out


def build_gaussian_window(channels: int, size: int = 11, sigma: float = 1.5, device="cpu"):
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(1)
    w2 = g @ g.t()
    w = w2.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return w


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor):
    c1 = 0.01**2
    c2 = 0.03**2
    p = window.shape[-1] // 2
    mx = F.conv2d(x, window, padding=p, groups=x.shape[1])
    my = F.conv2d(y, window, padding=p, groups=y.shape[1])
    mx2, my2, mxy = mx * mx, my * my, mx * my
    sx2 = F.conv2d(x * x, window, padding=p, groups=x.shape[1]) - mx2
    sy2 = F.conv2d(y * y, window, padding=p, groups=y.shape[1]) - my2
    sxy = F.conv2d(x * y, window, padding=p, groups=x.shape[1]) - mxy
    num = (2 * mxy + c1) * (2 * sxy + c2)
    den = (mx2 + my2 + c1) * (sx2 + sy2 + c2)
    return num / (den + 1e-8)


def masked_mean(x: torch.Tensor, mask: torch.Tensor):
    return (x * mask).sum() / (mask.sum() + 1e-8)


def train_one_epoch_phase(
    student,
    teacher,
    loader,
    optim,
    device,
    window,
    epoch: int,
    phase: str,
    rank: int,
    log_interval: int,
    max_train_steps: int,
):
    student.train()
    total = 0.0
    n = 0
    for step, (_clean, corr, mask, blur_type) in enumerate(loader):
        if max_train_steps > 0 and step >= max_train_steps:
            break
        corr = corr.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        blur_type = blur_type.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_out = teacher(corr, blur_type)

        pred = student(corr)
        l1 = masked_mean(torch.abs(pred - teacher_out).mean(1, keepdim=True), mask)
        ssim = ssim_map(pred, teacher_out, window).mean(1, keepdim=True)
        l_ssim = 1.0 - masked_mean(ssim, mask)
        loss = 1.0 * l1 + 0.2 * l_ssim

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total += float(loss.detach())
        n += 1
        if rank == 0 and (step % max(1, log_interval) == 0):
            print(
                f"[train] epoch={epoch} phase={phase} step={step} "
                f"loss={float(loss.detach()):.6f} "
                f"l1={float(l1.detach()):.6f} l_ssim={float(l_ssim.detach()):.6f}",
                flush=True,
            )
    return total, n


def validate_one_epoch_phase(
    student,
    teacher,
    loader,
    device,
    window,
    epoch: int,
    phase: str,
    rank: int,
    log_interval: int,
    max_val_steps: int,
):
    student.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for step, (_clean, corr, mask, blur_type) in enumerate(loader):
            if max_val_steps > 0 and step >= max_val_steps:
                break
            corr = corr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            blur_type = blur_type.to(device, non_blocking=True)

            teacher_out = teacher(corr, blur_type)
            pred = student(corr)
            l1 = masked_mean(torch.abs(pred - teacher_out).mean(1, keepdim=True), mask)
            ssim = ssim_map(pred, teacher_out, window).mean(1, keepdim=True)
            l_ssim = 1.0 - masked_mean(ssim, mask)
            loss = 1.0 * l1 + 0.2 * l_ssim
            total += float(loss.detach())
            n += 1
            if rank == 0 and (step % max(1, log_interval) == 0):
                print(
                    f"[val] epoch={epoch} phase={phase} step={step} "
                    f"loss={float(loss.detach()):.6f}",
                    flush=True,
                )
    return total, n


def init_dist():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world)
    return True, rank, world, local_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--restormer_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_train", default="train_10cat8")
    ap.add_argument("--split_val", default="val_10cat8")
    ap.add_argument("--epochs", type=int, default=50, help="Total epochs (cosine T_max matches this).")
    ap.add_argument("--batch_size", type=int, default=4)  # per GPU
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=50, help="Print every N train steps (rank0).")
    ap.add_argument("--max_train_steps", type=int, default=0, help="If >0, cap train steps per phase (debug).")
    ap.add_argument("--max_val_steps", type=int, default=0, help="If >0, cap val steps per phase (debug).")
    ap.add_argument(
        "--motion_kernel_size",
        type=int,
        default=35,
        help="Horizontal motion blur kernel size (odd; was 25).",
    )
    ap.add_argument(
        "--defocus_radius",
        type=int,
        default=9,
        help="Disk defocus radius in pixels (was 7).",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to student_last.pth / student_best.pth to continue training.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    distributed, rank, _, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(
            f"[startup] device={device} distributed={distributed} local_rank={local_rank} "
            f"epochs={args.epochs} batch_size={args.batch_size} workers={args.num_workers} "
            f"motion_kernel={args.motion_kernel_size} defocus_r={args.defocus_radius}",
            flush=True,
        )

    ds_tr_m = Co3dFrameDataset(
        args.data_root,
        args.split_train,
        TRAIN_CATEGORIES,
        "motion",
        motion_kernel_size=args.motion_kernel_size,
        defocus_radius=args.defocus_radius,
    )
    ds_tr_d = Co3dFrameDataset(
        args.data_root,
        args.split_train,
        TRAIN_CATEGORIES,
        "defocus",
        motion_kernel_size=args.motion_kernel_size,
        defocus_radius=args.defocus_radius,
    )
    ds_va_m = Co3dFrameDataset(
        args.data_root,
        args.split_val,
        TRAIN_CATEGORIES,
        "motion",
        motion_kernel_size=args.motion_kernel_size,
        defocus_radius=args.defocus_radius,
    )
    ds_va_d = Co3dFrameDataset(
        args.data_root,
        args.split_val,
        TRAIN_CATEGORIES,
        "defocus",
        motion_kernel_size=args.motion_kernel_size,
        defocus_radius=args.defocus_radius,
    )
    if rank == 0:
        print(
            f"[startup] train len motion={len(ds_tr_m)} defocus={len(ds_tr_d)} | "
            f"val len motion={len(ds_va_m)} defocus={len(ds_va_d)} "
            f"split_train={args.split_train} split_val={args.split_val}",
            flush=True,
        )

    smp_tr_m = DistributedSampler(ds_tr_m, shuffle=True) if distributed else None
    smp_tr_d = DistributedSampler(ds_tr_d, shuffle=True) if distributed else None
    smp_va_m = DistributedSampler(ds_va_m, shuffle=False) if distributed else None
    smp_va_d = DistributedSampler(ds_va_d, shuffle=False) if distributed else None

    ld_tr_m = DataLoader(
        ds_tr_m,
        batch_size=args.batch_size,
        shuffle=(smp_tr_m is None),
        sampler=smp_tr_m,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_full_images,
    )
    ld_tr_d = DataLoader(
        ds_tr_d,
        batch_size=args.batch_size,
        shuffle=(smp_tr_d is None),
        sampler=smp_tr_d,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_full_images,
    )
    ld_va_m = DataLoader(
        ds_va_m,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=smp_va_m,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_full_images,
    )
    ld_va_d = DataLoader(
        ds_va_d,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=smp_va_d,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_full_images,
    )

    student = StudentRestorationFrontEnd().to(device)
    resume_ck = None
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        rpath = Path(args.resume).expanduser()
        if not rpath.is_file():
            raise FileNotFoundError(f"--resume not found: {rpath}")
        try:
            resume_ck = torch.load(rpath, map_location="cpu", weights_only=False)
        except TypeError:
            resume_ck = torch.load(rpath, map_location="cpu")
        student.load_state_dict(resume_ck["state_dict"], strict=True)
        start_epoch = int(resume_ck.get("epoch", -1)) + 1
        best_val = float(resume_ck.get("best_val", float("inf")))
        if rank == 0:
            print(
                f"[resume] loaded {rpath} start_epoch={start_epoch} best_val={best_val}",
                flush=True,
            )

    teacher = RestormerTeacher(args.restormer_root).to(device)
    if rank == 0:
        print("[startup] teacher loaded (motion + defocus), frozen=True", flush=True)

    if distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[local_rank])

    optim = AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    if resume_ck is not None and resume_ck.get("optimizer"):
        try:
            optim.load_state_dict(resume_ck["optimizer"])
        except Exception as ex:
            if rank == 0:
                print(f"[resume] warning: could not load optimizer state: {ex}", flush=True)

    # PyTorch CosineAnnealingLR with last_epoch >= 0 requires initial_lr on each group.
    # Old checkpoints often have weights only; set initial_lr from current lr.
    for g in optim.param_groups:
        g.setdefault("initial_lr", float(g["lr"]))

    # Cosine over full training horizon; last_epoch aligns with completed epochs.
    sched_last = start_epoch - 1
    if resume_ck is not None and resume_ck.get("scheduler") and start_epoch > 0:
        try:
            sd = resume_ck["scheduler"]
            if sd.get("T_max") == args.epochs:
                sched = CosineAnnealingLR(
                    optim, T_max=args.epochs, eta_min=args.lr_min, last_epoch=-1
                )
                sched.load_state_dict(sd)
            else:
                raise ValueError("T_max mismatch")
        except Exception as ex:
            if rank == 0:
                print(f"[resume] rebuilding cosine scheduler (scheduler load failed: {ex})", flush=True)
            sched = CosineAnnealingLR(
                optim,
                T_max=args.epochs,
                eta_min=args.lr_min,
                last_epoch=sched_last,
            )
    else:
        sched = CosineAnnealingLR(
            optim,
            T_max=args.epochs,
            eta_min=args.lr_min,
            last_epoch=sched_last,
        )

    window = build_gaussian_window(3, size=11, sigma=1.5, device=device)
    best_path = os.path.join(args.output_dir, "student_best.pth")
    last_path = os.path.join(args.output_dir, "student_last.pth")

    if start_epoch >= args.epochs:
        if rank == 0:
            print(f"[done] start_epoch={start_epoch} >= epochs={args.epochs}, nothing to run.", flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, args.epochs):
        if smp_tr_m is not None:
            smp_tr_m.set_epoch(epoch)
            smp_tr_d.set_epoch(epoch + 10007)
        if smp_va_m is not None:
            smp_va_m.set_epoch(epoch)
            smp_va_d.set_epoch(epoch + 10007)

        tr_m, n_m = train_one_epoch_phase(
            student,
            teacher,
            ld_tr_m,
            optim,
            device,
            window,
            epoch,
            "motion",
            rank,
            args.log_interval,
            args.max_train_steps,
        )
        tr_d, n_d = train_one_epoch_phase(
            student,
            teacher,
            ld_tr_d,
            optim,
            device,
            window,
            epoch,
            "defocus",
            rank,
            args.log_interval,
            args.max_train_steps,
        )
        tr_loss = tr_m + tr_d
        tr_n = n_m + n_d

        vm, nm = validate_one_epoch_phase(
            student,
            teacher,
            ld_va_m,
            device,
            window,
            epoch,
            "motion",
            rank,
            args.log_interval,
            args.max_val_steps,
        )
        vd, nd = validate_one_epoch_phase(
            student,
            teacher,
            ld_va_d,
            device,
            window,
            epoch,
            "defocus",
            rank,
            args.log_interval,
            args.max_val_steps,
        )
        sched.step()
        tr_avg = tr_loss / max(1, tr_n)
        va_m_avg = vm / max(1, nm)
        va_d_avg = vd / max(1, nd)
        va_avg = 0.5 * (va_m_avg + va_d_avg)

        if rank == 0:
            row = {
                "epoch": epoch,
                "train_loss": tr_avg,
                "train_loss_motion": tr_m / max(1, n_m),
                "train_loss_defocus": tr_d / max(1, n_d),
                "val_loss": va_avg,
                "val_loss_motion": va_m_avg,
                "val_loss_defocus": va_d_avg,
                "lr": optim.param_groups[0]["lr"],
            }
            print(row, flush=True)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(row) + "\n")

            state = student.module.state_dict() if hasattr(student, "module") else student.state_dict()
            payload = {
                "epoch": epoch,
                "state_dict": state,
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "args": vars(args),
                "best_val": best_val,
            }
            torch.save(payload, last_path)
            if va_avg < best_val:
                best_val = va_avg
                payload["best_val"] = best_val
                torch.save(payload, best_path)
                print(f"  -> New best val (mean motion+defocus): {best_val:.6f}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

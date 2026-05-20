"""Distillation + Zero-DCE-style stability losses (device-agnostic)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((a - b) ** 2 + eps * eps))


class VGGPerceptualLoss(nn.Module):
    """Lightweight feature matching on conv3_3 (before relu4)."""

    def __init__(self, device: torch.device):
        super().__init__()
        try:
            w = models.VGG16_Weights.IMAGENET1K_FEATURES
            vgg = models.vgg16(weights=w).features[:16].to(device)
        except Exception:
            vgg = models.vgg16(pretrained=True).features[:16].to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach()
        fp = self.vgg(self._norm(pred))
        ft = self.vgg(self._norm(target))
        return F.l1_loss(fp, ft)


class L_exp(nn.Module):
    def __init__(self, patch_size: int, mean_val: float, device: torch.device):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
        self.register_buffer("target", torch.tensor([mean_val], device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=1, keepdim=True)
        mean = self.pool(x)
        return torch.mean((mean - self.target) ** 2)


class L_spa(nn.Module):
    """Spatial consistency (Zero-DCE); call as L_spa(enhanced, low_input) — same order as official train loop."""

    def __init__(self, device: torch.device):
        super().__init__()
        kl = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32, device=device)
        kr = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32, device=device)
        ku = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=device)
        kd = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32, device=device)
        self.register_buffer("weight_left", kl.view(1, 1, 3, 3))
        self.register_buffer("weight_right", kr.view(1, 1, 3, 3))
        self.register_buffer("weight_up", ku.view(1, 1, 3, 3))
        self.register_buffer("weight_down", kd.view(1, 1, 3, 3))
        self.pool = nn.AvgPool2d(4)

    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        org_mean = torch.mean(org, dim=1, keepdim=True)
        enhance_mean = torch.mean(enhance, dim=1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enh_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enh_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enh_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enh_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = (D_org_left - D_enh_left) ** 2
        D_right = (D_org_right - D_enh_right) ** 2
        D_up = (D_org_up - D_enh_up) ** 2
        D_down = (D_org_down - D_enh_down) ** 2
        return D_left + D_right + D_up + D_down


class L_TV(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.w = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h <= 1 or w <= 1:
            return x.new_zeros(())
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        count_h = (h - 1) * w
        count_w = h * (w - 1)
        return self.w * 2.0 * (h_tv / count_h + w_tv / count_w) / float(b)

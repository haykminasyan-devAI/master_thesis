#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DUST3R_DIR = os.path.join(PROJECT_DIR, "dust3r")
for p in [DUST3R_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from dust3r.inference import loss_of_one_batch
from dust3r.losses import ConfLoss, L21, Regr3D

from finetune_defocus.datasets.co3d_defocus import Co3dDefocus
from finetune_defocus.model_ifan_dust3r import build_model


def get_args():
    p = argparse.ArgumentParser("Evaluate IFAN+DUSt3R on defocus categories")
    p.add_argument("--co3d_root", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--defocus_radius", type=int, default=6)
    p.add_argument("--dust3r_ckpt", required=True)
    p.add_argument("--ifan_repo", required=True)
    p.add_argument("--ifan_ckpt", required=True)
    p.add_argument("--finetuned_ckpt", required=True)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _filter_dataset_categories(ds, categories):
    keep = set(categories)
    ds.scenes = {(obj, inst): views for (obj, inst), views in ds.scenes.items() if obj in keep}
    ds.scene_list = list(ds.scenes.keys())
    ds.invalidate = {scene: {} for scene in ds.scene_list}
    return ds


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = Co3dDefocus(
        defocus_radius=args.defocus_radius,
        split="train",
        ROOT=args.co3d_root,
        resolution=args.resolution,
        aug_crop=16,
    )
    dataset = _filter_dataset_categories(dataset, args.categories)
    subset = Subset(dataset, list(range(len(dataset))))
    loader = DataLoader(
        subset,
        sampler=torch.utils.data.SequentialSampler(subset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(
        dust3r_ckpt=args.dust3r_ckpt,
        ifan_repo=args.ifan_repo,
        ifan_ckpt=args.ifan_ckpt,
        device="cpu",
        freeze="ifan_only",
    ).to(device)
    ckpt = torch.load(args.finetuned_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
    model.eval()

    criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2).to(device)
    metric_logger = misc.MetricLogger(delimiter="  ")
    for batch in metric_logger.log_every(loader, 20, "Eval:"):
        res = loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=args.amp)
        loss = res["loss"][0] if isinstance(res["loss"], tuple) else res["loss"]
        metric_logger.update(loss=float(loss.detach()))
    metric_logger.synchronize_between_processes()

    result = {
        "categories": args.categories,
        "num_pairs": len(subset),
        "loss": metric_logger.meters["loss"].global_avg if "loss" in metric_logger.meters else None,
        "finetuned_ckpt": args.finetuned_ckpt,
    }
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

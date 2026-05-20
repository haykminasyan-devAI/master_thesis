#!/usr/bin/env python3
"""Fine-tune Uformer + DUSt3R on CO3D with noisy images (DDP-ready)."""

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
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DUST3R_DIR = os.path.join(PROJECT_DIR, 'dust3r')
for p in [DUST3R_DIR, PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.distributed.elastic.multiprocessing.errors import record
from dust3r.losses import ConfLoss, Regr3D, L21
from dust3r.inference import loss_of_one_batch
from dust3r.model import load_model

from finetune_noise.model import build_model
from finetune_noise.datasets.co3d_noise import Co3dNoise


def _attach_optional_attn_key_padding_mask(view, model):
    """If batch contains per-sample patch masks, attach key-padding mask for DUSt3R attention."""
    if 'attn_mask_hw' not in view:
        return
    m = view['attn_mask_hw']
    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m)
    if isinstance(m, list):
        m = torch.stack([
            torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in m
        ], dim=0)
    if not torch.is_tensor(m):
        return
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim != 3:
        return
    B, nh, nw = m.shape
    p = int(getattr(model, 'patch_size', 16))
    H, W = view['img'].shape[-2], view['img'].shape[-1]
    if nh * nw != (H // p) * (W // p):
        return
    view['attn_key_padding_mask'] = m.reshape(B, -1).to(device=view['img'].device, dtype=torch.bool)


def get_args_parser():
    p = argparse.ArgumentParser('Uformer + DUSt3R fine-tuning (noise)')
    p.add_argument('--co3d_root', required=True)
    p.add_argument('--noise_root', required=True)
    p.add_argument('--noise_sigmas', type=int, nargs='+', default=[30],
                   help='one or more noise sigmas to train jointly (e.g. 30 50 70)')
    p.add_argument('--train_categories', nargs='+', default=None,
                   help='optional explicit train categories (e.g. apple banana ...)')
    p.add_argument('--val_categories', nargs='+', default=None,
                   help='optional explicit val categories (e.g. car carrot)')
    p.add_argument('--val_source_split', default='train', choices=['train', 'test'],
                   help='dataset split used for category-based validation set')
    p.add_argument('--random_split', action='store_true',
                   help='use random train/val split (--val_ratio / --test_ratio) over all categories; '
                        'ignores --train_categories and --val_categories')
    p.add_argument('--dust3r_ckpt', required=True, help='pretrained DUSt3R .pth')
    p.add_argument('--dust3r_only', action='store_true',
                   help='train DUSt3R directly on noisy images (no Uformer front-end)')
    p.add_argument('--uformer_repo', default=None, help='path to cloned Uformer repo')
    p.add_argument('--uformer_weights', default=None, help='path to pretrained Uformer .pth')
    p.add_argument('--output_dir', default='./output_uformer_dust3r')
    p.add_argument('--freeze', default='uformer_only', choices=['uformer_only', 'all'])
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=0,
                   help='linear LR warmup before cosine (0 = cosine only)')
    p.add_argument('--eta_min', type=float, default=1e-6, help='cosine LR floor')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='max grad norm; 0 disables')
    p.add_argument('--resolution', type=int, default=224)
    p.add_argument('--amp', type=int, default=1, choices=[0, 1],
                   help='mixed precision')
    p.add_argument('--lambda_recon', type=float, default=0.0,
                   help='weight for L1(Uformer(noisy), clean) reconstruction loss')
    p.add_argument('--symmetrize_batch', type=int, default=0, choices=[0, 1],
                   help='1=symmetric pair loss (2x memory); 0 saves VRAM')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--print_freq', type=int, default=20)
    p.add_argument('--val_ratio', type=float, default=0.10)
    p.add_argument('--test_ratio', type=float, default=0.10)
    p.add_argument('--save_freq', type=int, default=1)
    p.add_argument('--keep_freq', type=int, default=10)
    p.add_argument('--val_every', type=int, default=1,
                   help='run validation every N epochs (1 = every epoch)')
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='stop after this many non-improving validations (0 disables)')
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_url', default='env://')
    p.add_argument('--wandb_project', default=None,
                   help='W&B project name. Leave unset to disable W&B logging.')
    p.add_argument('--wandb_run_name', default=None,
                   help='W&B run name (auto-generated if not set).')
    return p


def build_lr_scheduler(optimizer, epochs, warmup_epochs, eta_min):
    """Schedule shape must match the checkpoint when resuming, or load_state_dict fails."""
    wu = int(warmup_epochs)
    epochs = int(epochs)
    if wu <= 0:
        return CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=eta_min)
    if wu >= epochs:
        return LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=epochs)
    warmup = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=wu)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - wu), eta_min=eta_min)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[wu])


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        model_ref = model.module if hasattr(model, 'module') else model
        _attach_optional_attn_key_padding_mask(batch[0], model_ref)
        _attach_optional_attn_key_padding_mask(batch[1], model_ref)
        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp)
        loss3d = result['loss']
        if isinstance(loss3d, tuple):
            loss3d, _details = loss3d

        recon = None
        if args.lambda_recon and args.lambda_recon > 0:
            v1, v2 = result['view1'], result['view2']
            b1, b2 = batch
            rest1 = v1.get('img_restored', v1.get('img', None))
            rest2 = v2.get('img_restored', v2.get('img', None))
            clean1 = v1.get('img_clean', b1.get('img_clean', None))
            clean2 = v2.get('img_clean', b2.get('img_clean', None))
            if (rest1 is not None) and (rest2 is not None) and (clean1 is not None) and (clean2 is not None):
                recon = (rest1 - clean1).abs().mean()
                recon = recon + (rest2 - clean2).abs().mean()
                recon = 0.5 * recon
            else:
                if misc.is_main_process():
                    print('WARNING: recon tensors unavailable for this batch; skipping recon term.')

        loss = loss3d if recon is None else (loss3d + (args.lambda_recon * recon))
        loss_value = float(loss.detach())

        if not torch.isfinite(torch.tensor(loss_value)):
            print(f'WARNING: non-finite loss ({loss_value}), skipping batch')
            continue

        optimizer.zero_grad()
        trainable = [p for p in model.parameters() if p.requires_grad]
        clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
        loss_scaler(loss, optimizer, clip_grad=clip, parameters=trainable)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss3d=float(loss3d.detach()))
        if recon is not None:
            metric_logger.update(recon=float(recon.detach()))
            metric_logger.update(lambda_recon=float(args.lambda_recon))
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args, split_name='val'):
    if len(data_loader) == 0:
        if misc.is_main_process():
            print(f'WARNING: {split_name} loader is empty; returning inf losses.')
        return {'loss': float('inf'), 'loss3d': float('inf')}

    model.eval()
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = f'{split_name.capitalize()}:'

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        model_ref = model.module if hasattr(model, 'module') else model
        _attach_optional_attn_key_padding_mask(batch[0], model_ref)
        _attach_optional_attn_key_padding_mask(batch[1], model_ref)
        result = loss_of_one_batch(
            batch, model, criterion, device,
            symmetrize_batch=bool(args.symmetrize_batch), use_amp=args.amp)
        loss3d = result['loss']
        if isinstance(loss3d, tuple):
            loss3d, _details = loss3d

        recon = None
        if args.lambda_recon and args.lambda_recon > 0:
            v1, v2 = result['view1'], result['view2']
            b1, b2 = batch
            rest1 = v1.get('img_restored', v1.get('img', None))
            rest2 = v2.get('img_restored', v2.get('img', None))
            clean1 = v1.get('img_clean', b1.get('img_clean', None))
            clean2 = v2.get('img_clean', b2.get('img_clean', None))
            if (rest1 is not None) and (rest2 is not None) and (clean1 is not None) and (clean2 is not None):
                recon = (rest1 - clean1).abs().mean()
                recon = recon + (rest2 - clean2).abs().mean()
                recon = 0.5 * recon

        loss = loss3d if recon is None else (loss3d + (args.lambda_recon * recon))
        metric_logger.update(loss=float(loss.detach()))
        metric_logger.update(loss3d=float(loss3d.detach()))
        if recon is not None:
            metric_logger.update(recon=float(recon.detach()))

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def split_indices(n_items, val_ratio, test_ratio, seed):
    if val_ratio <= 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError('Require val_ratio > 0, test_ratio >= 0, and sum < 1')
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = int(round(n_items * val_ratio))
    n_test = int(round(n_items * test_ratio))
    n_train = n_items - n_val - n_test
    if n_train <= 0:
        raise ValueError('Split leaves no training samples')
    return idx[:n_train].tolist(), idx[n_train:n_train+n_val].tolist(), idx[n_train+n_val:].tolist()


def _filter_dataset_categories(ds, categories):
    """Filter Co3d/Co3dNoise dataset in-place by object category name."""
    keep = set(categories)
    ds.scenes = {(obj, inst): views for (obj, inst), views in ds.scenes.items() if obj in keep}
    ds.scene_list = list(ds.scenes.keys())
    ds.invalidate = {scene: {} for scene in ds.scene_list}
    return ds


def build_noise_dataset_for_categories(sigmas, noise_root, co3d_root, resolution, categories, split='train'):
    per_sigma = []
    for sigma in sigmas:
        ds = Co3dNoise(
            noise_sigma=sigma,
            noise_root=noise_root,
            split=split,
            ROOT=co3d_root,
            resolution=resolution,
            aug_crop=16,
        )
        ds = _filter_dataset_categories(ds, categories)
        per_sigma.append(ds)
    return ConcatDataset(per_sigma) if len(per_sigma) > 1 else per_sigma[0]


def make_loader(subset, batch_size, num_workers, distributed, is_train):
    if distributed:
        sampler = DistributedSampler(subset, shuffle=is_train, drop_last=is_train)
    else:
        sampler = (torch.utils.data.RandomSampler(subset) if is_train
                   else torch.utils.data.SequentialSampler(subset))
    return DataLoader(subset, sampler=sampler, batch_size=batch_size,
                      num_workers=num_workers, pin_memory=True, drop_last=is_train)


@record
def main():
    args = get_args_parser().parse_args()
    misc.init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sigmas = sorted(set(args.noise_sigmas))
    using_category_split = (
        (not args.random_split)
        and bool(args.train_categories)
        and bool(args.val_categories))
    if args.random_split and misc.is_main_process():
        print('Random split enabled (ignoring category lists).')
    if using_category_split:
        overlap = set(args.train_categories).intersection(set(args.val_categories))
        if overlap:
            raise ValueError(f'Category overlap between train/val: {sorted(overlap)}')
        print(f'Category split enabled.')
        print(f'  train_categories={args.train_categories}')
        print(f'  val_categories={args.val_categories}')
        ds_train = build_noise_dataset_for_categories(
            sigmas, args.noise_root, args.co3d_root, args.resolution, args.train_categories)
        ds_val = build_noise_dataset_for_categories(
            sigmas, args.noise_root, args.co3d_root, args.resolution, args.val_categories, split=args.val_source_split)
        ds_test = None
        print(f'Dataset (category split): train={len(ds_train)}, val={len(ds_val)}, test=0 '
              f'(noise_sigmas={sigmas})')
    else:
        per_sigma = [
            Co3dNoise(
                noise_sigma=sigma,
                noise_root=args.noise_root,
                split='train',
                ROOT=args.co3d_root,
                resolution=args.resolution,
                aug_crop=16,
            )
            for sigma in sigmas
        ]
        dataset = ConcatDataset(per_sigma) if len(per_sigma) > 1 else per_sigma[0]
        print(f'Dataset: {len(dataset)} pairs (noise_sigmas={sigmas})')

        train_idx, val_idx, test_idx = split_indices(
            len(dataset), args.val_ratio, args.test_ratio, seed=args.seed)
        ds_train = Subset(dataset, train_idx)
        ds_val = Subset(dataset, val_idx)
        ds_test = Subset(dataset, test_idx) if len(test_idx) > 0 else None
        print(f'Split: train={len(ds_train)}, val={len(ds_val)}, '
              f'test={len(ds_test) if ds_test else 0}')

    if len(ds_train) == 0:
        raise ValueError('Training dataset is empty after filtering/splitting.')
    if len(ds_val) == 0:
        raise ValueError(
            f'Validation dataset is empty. For category split, try --val_source_split test. '
            f'train_categories={args.train_categories}, val_categories={args.val_categories}'
        )

    loader_train = make_loader(ds_train, args.batch_size, args.num_workers,
                               args.distributed, is_train=True)
    loader_val = make_loader(ds_val, args.batch_size, args.num_workers,
                             args.distributed, is_train=False)
    loader_test = (make_loader(ds_test, args.batch_size, args.num_workers,
                               args.distributed, is_train=False)
                   if ds_test else None)

    model = build_model(
        dust3r_ckpt=args.dust3r_ckpt,
        uformer_repo=args.uformer_repo,
        uformer_weights=args.uformer_weights,
        device='cpu',
        freeze=args.freeze,
    ).to(device) if not args.dust3r_only else load_model(args.dust3r_ckpt, device='cpu').to(device)

    if args.dust3r_only and (args.lambda_recon and args.lambda_recon > 0):
        if misc.is_main_process():
            print('DUSt3R-only mode ignores reconstruction branch; forcing lambda_recon=0.0')
        args.lambda_recon = 0.0

    model_without_ddp = model
    if args.distributed:
        ddp_find_unused = bool(args.dust3r_only)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=ddp_find_unused,
            gradient_as_bucket_view=False,
        )
        if misc.is_main_process():
            print(f'DDP config: find_unused_parameters={ddp_find_unused}')
        model_without_ddp = model.module

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_total:,} total, {n_trainable:,} trainable (freeze={args.freeze})')

    criterion = ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    loss_scaler = NativeScaler(enabled=bool(args.amp))

    last_ckpt = os.path.join(args.output_dir, 'checkpoint-last.pth')
    best_ckpt = os.path.join(args.output_dir, 'checkpoint-best-val.pth')
    start_epoch = 0
    best_val_loss = float('inf')

    # Resume: load optimizer first, then build the scheduler using the *saved* schedule
    # geometry (epochs / warmup / eta_min). If we rebuild with a new --epochs (e.g. 50)
    # while the checkpoint was trained with 30, T_max no longer matches and
    # load_state_dict fails — warmup/cosine restart and LR can rise again.
    sched_epochs = int(args.epochs)
    sched_wu = int(args.warmup_epochs)
    sched_eta = float(args.eta_min)
    resume_ckpt = None
    if os.path.isfile(last_ckpt):
        resume_ckpt = torch.load(last_ckpt, map_location='cpu')
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        if isinstance(resume_ckpt.get('args'), dict):
            sa = resume_ckpt['args']
            sched_epochs = int(sa.get('epochs', args.epochs))
            sched_wu = int(sa.get('warmup_epochs', args.warmup_epochs))
            sched_eta = float(sa.get('eta_min', args.eta_min))
        elif misc.is_main_process():
            print(
                'Warning: checkpoint has no args metadata; using CLI for LR schedule shape. '
                'If scheduler load fails, use the same --epochs/--warmup_epochs as the original run.'
            )
        if misc.is_main_process():
            print(
                f'Resume: LR schedule matches checkpoint: '
                f'total_epochs={sched_epochs}, warmup_epochs={sched_wu}, eta_min={sched_eta}. '
                f'This invocation --epochs={args.epochs}.'
            )

    scheduler = build_lr_scheduler(optimizer, sched_epochs, sched_wu, sched_eta)

    if resume_ckpt is not None:
        model_without_ddp.load_state_dict(resume_ckpt['model'])
        if 'scheduler' in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt['scheduler'])
                if misc.is_main_process():
                    print('Loaded LR scheduler state from checkpoint (continues previous job).')
            except Exception as ex:
                print(f'Warning: scheduler not loaded ({ex})')
        start_epoch = resume_ckpt['epoch'] + 1
        best_val_loss = float(resume_ckpt.get('best_val_loss', best_val_loss))
        print(f'Resumed from epoch {resume_ckpt["epoch"]}')
        if int(args.epochs) > sched_epochs and misc.is_main_process():
            print(
                f'Note: CLI --epochs={args.epochs} is greater than the schedule used in the '
                f'checkpoint (epochs={sched_epochs}). Training will continue for '
                f'epochs {start_epoch}..{args.epochs - 1}; LR is still stepped with the '
                f'loaded scheduler (cosine may already be at or near eta_min).'
            )

    print(f'Training for {args.epochs} epochs (from {start_epoch})')

    use_wandb = bool(args.wandb_project) and misc.is_main_process()
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                resume='allow',
                dir=args.output_dir,
            )
            print(f'W&B run: {wandb.run.url}')
        except Exception as e:
            print(f'WARNING: W&B init failed ({e}). Continuing without W&B.')
            use_wandb = False

    t0 = time.time()

    val_every = max(1, int(args.val_every))
    patience = max(0, int(args.early_stop_patience))
    no_improve_count = 0

    for epoch in range(start_epoch, args.epochs):
        if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, loader_train, optimizer,
                                      device, epoch, loss_scaler, args)
        do_val = ((epoch + 1) % val_every == 0) or (epoch + 1 == args.epochs)
        val_stats = {}
        test_stats = {}
        if do_val:
            val_stats = evaluate(model, criterion, loader_val, device, args, 'val')
            if loader_test is not None:
                test_stats = evaluate(model, criterion, loader_test, device, args, 'test')
        elif misc.is_main_process():
            print(f'Skipping validation at epoch {epoch} (val_every={val_every})')

        scheduler.step()

        if misc.is_main_process():
            log_stats = {
                'epoch': epoch,
                'lr': scheduler.get_last_lr()[0],
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
            print(f'Epoch {epoch}: {log_stats}')

            if use_wandb:
                import wandb
                wandb.log(log_stats, step=epoch)

            ckpt_data = {
                'epoch': epoch,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': vars(args),
                'best_val_loss': best_val_loss,
            }
            if args.save_freq and (epoch + 1) % args.save_freq == 0:
                torch.save(ckpt_data, last_ckpt)
            if args.keep_freq and (epoch + 1) % args.keep_freq == 0:
                torch.save(ckpt_data,
                           os.path.join(args.output_dir, f'checkpoint-{epoch:04d}.pth'))

            if do_val:
                cur_val = float(val_stats.get('loss', float('inf')))
                if cur_val < best_val_loss:
                    best_val_loss = cur_val
                    ckpt_data['best_val_loss'] = best_val_loss
                    torch.save(ckpt_data, best_ckpt)
                    no_improve_count = 0
                    print(f'  -> New best val: {best_val_loss:.6f}')
                else:
                    no_improve_count += 1
                    print(f'  -> No val improvement ({no_improve_count}/{patience})')
                    if patience > 0 and no_improve_count >= patience:
                        print(f'Early stopping at epoch {epoch} (patience={patience})')
                        break

    print(f'Training completed in {(time.time() - t0) / 3600:.1f}h')
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()

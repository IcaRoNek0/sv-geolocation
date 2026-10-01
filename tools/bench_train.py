#!/usr/bin/env python
"""Synthetic GPU/DDP benchmark; no image I/O, downloads or checkpoint writes.

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/bench_train.py --config configs/round3.yaml --mode packed
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from data.prepare import ViewConfig, choose_view_count
from models.env_model import build_model
from models.losses import MultiTaskLoss
from train import autocast_ctx, pick_amp_dtype


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--mode', choices=['packed', 'dense'], default='packed')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=10)
    args = ap.parse_args()
    if args.steps < 1 or args.warmup < 0:
        ap.error('steps must be positive and warmup nonnegative')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; run this benchmark on Kaggle')
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get('task', 'environment') != 'environment':
        ap.error('This benchmark uses environment heads; use configs/round3.yaml')
    cfg.update(pretrained=False, pack_views=args.mode == 'packed')
    torch.set_num_threads(cfg.get('cpu_threads', 1))
    torch.backends.cudnn.benchmark = False
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    if world > 1:
        torch.distributed.init_process_group('nccl')
    torch.manual_seed(2026)
    raw = build_model(2604, 350, 35, cfg).to(device).train()
    model = (torch.nn.parallel.DistributedDataParallel(
        raw, device_ids=[rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        if world > 1 else raw)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 3e-4))
    amp = pick_amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=amp is torch.float16)
    criterion = MultiTaskLoss(np.eye(2604, dtype=np.float32), **cfg.get('loss', {})).to(device)
    vc = ViewConfig(**cfg.get('views', {}))
    b = args.batch_size if args.batch_size is not None else cfg.get('batch_size', 8)
    accum = cfg.get('accum', 1)
    if b < 1 or accum < 1:
        ap.error('batch size and accumulation must be positive')
    views = torch.randint(0, 256, (b, vc.n_max, 3, vc.size, vc.size),
                          dtype=torch.uint8, device=device)
    targets = {'county': torch.arange(b, device=device) % 2604,
               'city': torch.arange(b, device=device) % 350,
               'prov': torch.arange(b, device=device) % 35,
               'coord': torch.zeros(b, 2, device=device)}
    rng = np.random.default_rng(2026 + rank)
    timings = []
    opt.zero_grad(set_to_none=True)
    # Round warmup to an accumulation boundary so measurement starts cleanly.
    warmup = ((args.warmup + accum - 1) // accum) * accum
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + args.steps):
        counts = [choose_view_count(rng, vc) for _ in range(b)]
        mask = torch.tensor(np.arange(vc.n_max)[None, :] < np.array(counts)[:, None], device=device)
        targets['vmask'] = mask
        update = (step + 1) % accum == 0
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        sync = model.no_sync() if world > 1 and not update else contextlib.nullcontext()
        with sync:
            with autocast_ctx(device, amp):
                out = model(views, mask, return_view_logits=bool(cfg.get('loss', {}).get('w_view', 0)))
                loss, _ = criterion(out, targets, collect_parts=False)
            scaler.scale(loss / accum).backward()
        if update:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 5))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - start
        if step >= warmup:
            timings.append(dt)
        if step == 0 or (step + 1) % 10 == 0:
            print(f'rank{rank} {args.mode} step={step+1} valid_views={sum(counts)} seconds={dt:.3f}', flush=True)
    report = {'rank': rank, 'mode': args.mode, 'world': world, 'batch_per_gpu': b,
              'mean_seconds': float(np.mean(timings)), 'median_seconds': float(np.median(timings)),
              'p90_seconds': float(np.percentile(timings, 90)),
              'peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9}
    print(json.dumps(report), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

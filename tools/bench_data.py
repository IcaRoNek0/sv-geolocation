#!/usr/bin/env python
"""量数据管线的各段耗时，在目标机器上跑。

训练的瓶颈通常在 CPU 而不是 GPU（日志里"等数据 / 计算"的比值就是证据）。
但瓶颈具体落在解码、视图切分还是增强上，因机器而异——手机和 Colab 的
比例并不一样。所以不要猜，在**将要训练的那台机器上**跑一遍。

用法：
    python tools/bench_data.py --data /content/data
    python tools/bench_data.py --data /content/data --workers 1 2 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prepare import ViewConfig, augment, make_sample  # noqa: E402
from data.shards import ShardIndex, decode_jpeg  # noqa: E402
from utils.views import extract_views, surround_headings  # noqa: E402


def bench(fn, n):
    fn()
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="按这份配置的 views 段来测，否则量的是默认值不是真实设置")
    ap.add_argument("--data", type=Path, default=Path("data/shards"))
    ap.add_argument("--points", type=Path, default=Path("data/pool/county_points.npz"))
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--workers", type=int, nargs="*", default=[],
                    help="额外测这些 worker 数下的 DataLoader 吞吐")
    args = ap.parse_args()

    import os
    from data.labels import build_classes
    from data.shards import load_samples, load_split
    from utils.geo_utils import CountyPoints

    n_cpu = os.cpu_count() or 1
    print(f"CPU {n_cpu} 核")
    try:                       # 没有 torch 也能测各段的纯 CPU 耗时
        import torch
        print(f"torch {torch.__version__}   CUDA {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("未安装 torch —— 只测各段耗时，跳过 DataLoader 吞吐")

    shards = sorted(args.data.glob("*.tar"))
    t = time.time()
    idx = ShardIndex(shards)
    print(f"分片 {len(shards)} 个 / {len(idx):,} 条，索引扫描 {time.time()-t:.1f}s")

    samples = load_samples(args.data / "samples.jsonl")
    split, _ = load_split(args.data / "split.json")
    cp = CountyPoints(args.points)
    adcodes, ci = build_classes(list(samples.values()), cp)

    # 必须按真实配置测：用 ViewConfig() 的默认值量出来的是另一套数字，
    # 会让人对着错的瓶颈优化。
    try:
        import yaml
        full = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        views_cfg = full.get("views", {})
        print(f"视图配置来自 {args.config}")
    except ImportError:
        views_cfg = {}
        print(f"⚠ 没有 pyyaml，退回 ViewConfig 默认值——"
              f"这与训练用的配置可能不同，数字仅供参考")
    cfg = ViewConfig(**views_cfg)
    # 取一批真实的训练样本，覆盖两种尺寸（四川 2048 宽、全国 1024 宽）
    keys = [k for k, v in split.items() if v == "train"][: args.iterations]
    blobs = {k: idx.read(k) for k in keys}

    print(f"\n视图配置：n_max={cfg.n_max} size={cfg.size} fov={cfg.fov}")
    print(f"每次测 {args.iterations} 条\n")
    print(f"{'阶段':<26}{'每样本':>10}")

    print(f"{'读分片 + 解码':<24}"
          f"{bench(lambda: decode_jpeg(next(iter(blobs.values())), min_width=4*cfg.size), 40):>9.1f}ms")

    panos = {k: decode_jpeg(b, min_width=4 * cfg.size) for k, b in blobs.items()}
    first = next(iter(panos.values()))
    print(f"{'  解码后尺寸':<24}{str(first.shape[1])+'x'+str(first.shape[0]):>10}")

    print(f"{'视图切分 ({})'.format(cfg.n_max):<24}"
          f"{bench(lambda: extract_views(first, surround_headings(cfg.n_max), size=cfg.size), 40):>9.1f}ms")

    views = extract_views(first, surround_headings(cfg.n_max), size=cfg.size)
    rng = np.random.default_rng(0)
    print(f"{'增强':<24}{bench(lambda: augment(views.copy(), rng, cfg), 40):>9.1f}ms")

    # 完整一条样本：轮换不同的全景，避免缓存带来的乐观偏差
    seq = list(panos.values())
    state = {"i": 0}

    def one():
        p = seq[state["i"] % len(seq)]
        state["i"] += 1
        return make_sample(p, rng, cfg)

    per = bench(one, args.iterations)
    print(f"\n{'合计（不含解码）':<22}{per:>9.1f}ms")
    per_total = per + 5.0
    print(f"{'合计（含解码，估）':<22}{per_total:>9.1f}ms")
    print(f"{'单 worker 吞吐':<24}{1000/per_total:>9.1f} 条/秒")

    if args.workers:
        from data.dataset import PanoramaDataset, make_loader
        print("\nDataLoader 实测吞吐（含进程开销与传输）：")
        ds = PanoramaDataset(shards, split, samples, ci,
                             {k: (0.0, 0.0) for k in samples},
                             {k: 0 for k in samples}, {k: 0 for k in samples},
                             split="train", view_cfg=cfg, augment=True)
        ds._idx = idx
        for w in args.workers:
            loader = make_loader(ds, 8, True, num_workers=w)
            it = iter(loader)
            for _ in range(3):        # 预热，跳过 worker 启动
                next(it)
            n, t0 = 0, time.time()
            for _ in range(12):
                b = next(it)
                n += b["views"].shape[0]
            dt = time.time() - t0
            print(f"  workers={w}  {n/dt:>6.1f} 条/秒   "
                  f"（{dt/12*1000:.0f} ms/batch，batch=8）")
            del loader

    print("\n怎么看："
          "若'合计'远大于'视图切分+增强'，瓶颈在解码或进程传输；"
          "若某一段独占大头，就优化那一段。")
    print("换更大模型不会更快——GPU 本来就闲着，限制的是这里。")


if __name__ == "__main__":
    main()

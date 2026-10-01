#!/usr/bin/env python
"""训练主程序。

    python train.py --config configs/default.yaml --data <分片目录> --out <checkpoint 目录>
    python train.py ... --resume          # 断点续跑
    python train.py ... --overfit 64      # 冒烟：64 条样本能否过拟合

Colab 约束：T4 只支持 fp16（自动判断）；会话 4-6 小时被回收，故按时间存
checkpoint 且必须写 Drive；分片顺序固定 seed，否则续跑等于换训练集。
"""
import argparse
import contextlib
import json
import os
import math
import hashlib
from datetime import timedelta
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import PanoramaDataset, make_loader
from data.labels import (build_classes, centroids, city_of_adcode,
                         province_of_adcode, soft_targets)
from data.prepare import ViewConfig
from data.shards import load_samples, load_split, open_index
from models.env_model import build_model
from models.losses import MultiTaskLoss
from utils.geo_utils import CountyPoints, denormalize_coords, normalize_coords
from utils.metrics import summarize


_T0 = time.time()
_IS_MAIN = True          # 分布式下只有 rank 0 打印与存盘


def stage(msg):
    """打印启动阶段，带累计耗时。

    必须 flush：Colab 的 stdout 走管道默认块缓冲，不刷的话会静默十几秒。
    """
    if _IS_MAIN:
        print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def _fmt_lr(opt):
    """把各参数组的学习率都列出来。

    只打 param_groups[0] 会误导：主干用的是 lr×backbone_lr_scale，正好是
    第 0 组，于是日志里显示 3.0e-05 而配置写的是 3.0e-04，看起来像配置没生效。
    """
    vals = [g["lr"] for g in opt.param_groups]
    if len(set(f"{v:.1e}" for v in vals)) == 1:
        return f"{vals[0]:.1e}"
    return "/".join(f"{v:.1e}" for v in vals)


def autocast_ctx(device, amp_dtype):
    """autocast 上下文。dtype=None 在 CPU 上会报错，所以显式走空上下文。"""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def pick_amp_dtype(device):
    """T4（sm_75）不支持 bf16，只能 fp16；sm≥8 用 bf16 更稳。"""
    if device.type != "cuda":
        return None
    cc = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if cc[0] >= 8 else torch.float16


def stratified_subset(keys, samples, n, seed=0):
    """按县级类别轮转抽 N 条。

    直接取前 N 条会高度聚集（panoid 含城市码），多类基线能到 0.6。
    """
    by_class = defaultdict(list)
    for k in keys:
        by_class[samples[k]["adcode"]].append(k)
    rng = random.Random(seed)
    for v in by_class.values():
        rng.shuffle(v)
    classes = sorted(by_class)
    out, i = [], 0
    while len(out) < n:
        added = False
        for c in classes:
            if i < len(by_class[c]):
                out.append(by_class[c][i])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        i += 1
    return out


def apply_freeze_policy(model, epoch, cfg):
    """按 epoch 幂等地设定主干可训练性。

    不能写成"到某个 epoch 解冻一次"：续跑时 start_epoch 已越过解冻点，
    条件永不成立，主干会一直冻着。
    """
    if not cfg.get("freeze_backbone", False):
        model.set_backbone_trainable(True)
        return
    at = cfg.get("unfreeze_after", 0)
    model.set_backbone_trainable(bool(at) and epoch >= at)


def build_indices(samples, class_index):
    """构造各头的标签索引与坐标回归目标。"""
    county_index, coord_index, city_index, prov_index = {}, {}, {}, {}
    cities, provs = set(), set()
    for pid, s in samples.items():
        county_index[pid] = class_index.get(s["adcode"], -1)
        x, y = normalize_coords(s["lng"], s["lat"])
        coord_index[pid] = (float(x), float(y))
        city_index[pid] = city_of_adcode(s["adcode"])
        prov_index[pid] = province_of_adcode(s["adcode"])
        cities.add(city_index[pid])
        provs.add(prov_index[pid])
    city_list = sorted(cities)
    prov_list = sorted(provs)
    city_map = {c: i for i, c in enumerate(city_list)}
    prov_map = {p: i for i, p in enumerate(prov_list)}
    return (
        county_index,
        coord_index,
        {k: city_map[v] for k, v in city_index.items()},
        {k: prov_map[v] for k, v in prov_index.items()},
        city_list,
        prov_list,
    )


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, n_classes, cent, loss_fn=None):
    model.eval()
    all_scores, all_targets, all_true_coord = [], [], []
    losses = []
    auxiliary = np.zeros(4, dtype=np.int64)
    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        vmask = batch["vmask"].to(device, non_blocking=True)
        with autocast_ctx(device, amp_dtype):
            out = model(views, vmask)
        if loss_fn is not None:
            b = {k: v.to(device) for k, v in batch.items()
                 if k in ("county", "city", "prov", "coord")}
            _, parts = loss_fn(out, b)
            losses.append(parts)
        for j, head in enumerate(("city", "prov")):
            target = batch[head].numpy()
            valid_aux = target >= 0
            pred = out[head].argmax(dim=-1).cpu().numpy()
            auxiliary[2*j] += ((pred == target) & valid_aux).sum()
            auxiliary[2*j+1] += valid_aux.sum()
        scores = out["county"].float().cpu().numpy()
        all_scores.append(np.argsort(-scores, axis=1)[:, :min(10, n_classes)])
        all_targets.append(batch["county"].numpy())
        # 必须 stack 成 (N, 2)，不能留着元组列表去 concatenate：
        # np.concatenate 会把每个 (lon, lat) 元组当一维序列拼起来，
        # 得到长度 2N 的平铺数组，后面的布尔索引会静默取错行。
        lon, lat = denormalize_coords(batch["coord"][:, 0].numpy(),
                                      batch["coord"][:, 1].numpy())
        all_true_coord.append(np.stack([lon, lat], axis=1))
    model.train()

    local = {
        "auxiliary": auxiliary,
        "order": np.concatenate(all_scores) if all_scores else np.empty((0, min(10, n_classes)), dtype=np.int64),
        "targets": np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.int64),
        "coords": np.concatenate(all_true_coord) if all_true_coord else np.empty((0, 2)),
    }
    gathered = [local]
    if torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        # Only transmit ranked indices, not the full N x 2604 logit matrix.
        torch.distributed.all_gather_object(gathered, local)
    order = np.concatenate([part["order"] for part in gathered])
    targets = np.concatenate([part["targets"] for part in gathered])
    coords = np.concatenate([part["coords"] for part in gathered])
    scores = np.full((len(targets), n_classes), -np.inf, dtype=np.float32)
    scores[np.arange(len(targets))[:, None], order] = -np.arange(order.shape[1])

    valid = targets >= 0
    scores, targets = scores[valid], targets[valid]
    if len(targets) == 0:
        return {"known_fraction": 0.0, "n": 0}

    m = summarize(scores, targets, n_classes,
                  class_centroids=cent if len(cent) == n_classes else None)

    # 距离误差只在两条线索融合选点后才有意义，这里先用 top-1 县的质心近似
    pred_loc = np.stack([
        cent[scores[i].argmax()] if len(cent) > scores[i].argmax() else (np.nan, np.nan)
        for i in range(len(scores))
    ])
    true_loc = coords[valid]
    ok = np.isfinite(pred_loc).all(axis=1)
    if ok.any():
        from utils.metrics import distance_summary
        m.update({f"top1_{k}": v for k, v in
                  distance_summary(pred_loc[ok], true_loc[ok]).items()})
    m["known_fraction"] = float(valid.mean())
    m["all_top1"] = m["top1"] * m["known_fraction"]
    m["all_top5"] = m["top5"] * m["known_fraction"]
    auxiliary = sum((part["auxiliary"] for part in gathered), np.zeros(4, dtype=np.int64))
    m["city_or_year_top1"] = float(auxiliary[0] / max(1, auxiliary[1]))
    m["province_or_prefix_top1"] = float(auxiliary[2] / max(1, auxiliary[3]))
    if losses:
        m["loss"] = float(np.mean([l["county"] for l in losses]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/shards"),
                    help="分片与 samples.jsonl / split.json 所在目录")
    ap.add_argument("--points", type=Path, default=Path("data/pool/county_points.npz"))
    ap.add_argument("--out", type=Path, required=True, help="checkpoint 与日志目录")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", type=Path, help="Warm-start compatible .pt weights with a fresh optimizer")
    ap.add_argument("--epochs", type=int, help="Override epoch budget, including overfit mode")
    ap.add_argument("--overfit", type=int, default=0,
                    help="只用 N 条样本训练，验证能否过拟合（M0 冒烟）")
    ap.add_argument("--workers", type=int, default=None,
                    help="覆盖配置里的 DataLoader worker 数。核数少的机器上"
                         "开太多会与主进程抢 CPU，值得实测")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="覆盖配置里的 batch_size（不动 accum）")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="每 N 轮验证一次。冒烟测试不需要每轮都验，"
                         "验证集上的前向和训练一样贵")
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    if args.resume and args.init:
        raise ValueError("Use --init for a new experiment or --resume, not both")

    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        cfg["epochs"] = args.epochs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 分布式是可选的：torchrun --nproc_per_node=2 启动即启用，
    # 直接 python train.py 则走原来的单卡路径，行为不变。
    world = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = torch.distributed.get_rank()
    else:
        local_rank, rank = 0, 0
    global _IS_MAIN
    _IS_MAIN = rank == 0

    device = (torch.device("cuda", local_rank) if torch.cuda.is_available()
              else torch.device("cpu"))
    amp_dtype = pick_amp_dtype(device)
    if device.type == "cuda":
        # Packed views change convolution batch shapes on every step.
        # Autotuning those shapes can dominate useful GPU work.
        packed = cfg.get("pack_views", bool(cfg.get("loss", {}).get("w_view", 0)))
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False)) and not packed
        stage(f"cuDNN benchmark={torch.backends.cudnn.benchmark}, pack_views={packed}")
    torch.set_num_threads(cfg.get("cpu_threads", 1))
    if _IS_MAIN:
        print(f"设备 {device}  精度 {amp_dtype}  "
              f"{'DDP ×' + str(world) if distributed else '单进程'}")
    if device.type == "cuda":
        print(f"  rank{rank} {torch.cuda.get_device_name(local_rank)}  "
              f"显存 {torch.cuda.get_device_properties(local_rank).total_memory/1e9:.1f} GB",
              flush=True)
        if _IS_MAIN and amp_dtype is torch.float16:
            print("提示：Turing 及更早架构只能 fp16；换到 L4/A100 会自动切 bf16")

    args.out.mkdir(parents=True, exist_ok=True)
    stage(f"设备 {device}  精度 {amp_dtype}")
    if not (list(args.data.glob("*.tar"))
            or [p for p in args.data.iterdir() if p.is_dir()]):
        raise SystemExit(f"{args.data} 下既没有 .tar 分片也没有解包目录——第 ⑥ 格跑了吗？")
    samples = load_samples(args.data / "samples.jsonl")
    stage(f"元数据 {len(samples):,} 条，数据目录 {args.data}")
    split, split_payload = load_split(args.data / "split.json")
    task = cfg.get("task", "environment")
    if task == "vehicle":
        from data.vehicle import build_vehicle_indices
        (adcodes, city_list, prov_list, county_index, city_index, prov_index,
         coord_index) = build_vehicle_indices(samples, split,
             cfg.get("min_vehicle_samples", 20), cfg.get("min_vehicle_dates", 2))
        soft = np.eye(len(adcodes), dtype=np.float32)
        cent = np.empty((0, 2))
        stage(f"Vehicle classes {len(adcodes)}, years {len(city_list)}")
    elif task == "environment":
        cp = CountyPoints(args.points)
        adcodes, class_index = build_classes(list(samples.values()), cp)
        cent = centroids(adcodes, cp)
        soft = soft_targets(cent, half_km=cfg.get("soft_half_km", 50.0))
        (county_index, coord_index, city_index, prov_index,
         city_list, prov_list) = build_indices(samples, class_index)
        stage(f"Classes {len(adcodes)} counties / {len(city_list)} cities / {len(prov_list)} provinces")
    else:
        raise ValueError(f"Unknown task: {task}")

    # 类别表随 checkpoint 一起存：推理端靠它把 logit 下标映回县码。
    # 不这样做的话，推理时重建类别表一旦与训练时不一致，预测会被静默地
    # 解释成别的县——而且完全看不出来。
    run_meta = {
        "adcodes": adcodes, "cities": city_list, "provinces": prov_list,
        "backbone": cfg.get("backbone", "convnext_tiny"),
        "views": cfg.get("views", {}), "soft_half_km": cfg.get("soft_half_km", 50.0),
        "seed": args.seed, "task": cfg.get("task", "environment"),
        "loss": cfg.get("loss", {}), "format_version": 2,
        "samples_sha256": hashlib.sha256((args.data / "samples.jsonl").read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256((args.data / "split.json").read_bytes()).hexdigest(),
    }
    meta_path = args.out / "classes.json"
    if args.resume:
        if not (args.out / "last.pt").exists() or not meta_path.exists():
            raise ValueError("--resume requires last.pt and classes.json")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in run_meta if old_meta.get("format_version") == 2 else ("adcodes", "cities", "provinces", "backbone", "views", "seed"):
            if old_meta.get(key) != run_meta[key]:
                raise ValueError(f"Resume metadata mismatch: {key}; use a new directory")
    elif (args.out / "last.pt").exists() or (args.out / "best.pt").exists():
        raise ValueError("Output contains checkpoints; use --resume or a new directory")
    if _IS_MAIN:
        meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 数据集 ──────────────────────────────────────────────────────
    vcfg = ViewConfig(**cfg.get("views", {}))
    if args.overfit:
        # 冒烟要验证管线而非增强鲁棒性；全开增强时 64 条样本 230 轮才到
        # top1 0.70，判断变得含糊。
        vcfg.crop_scale = None
        vcfg.blur_prob = 0.0
        vcfg.brightness = vcfg.contrast = vcfg.saturation = 0.1
        vcfg.channel_gain = 0.0
    idx_cache = {}          # 各划分共用一份索引，避免重复扫 tar
    # Command-line loader overrides must apply to validation and training alike.
    import os as _os
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    def dataset(split_name, augment, eval_mode="panorama"):
        ds = PanoramaDataset(
            args.data, split, samples, county_index, coord_index,
            city_index, prov_index, split=split_name, view_cfg=vcfg,
            augment=augment, seed=args.seed, eval_mode=eval_mode, task=task)
        # 索引只在父进程建一次（扫 tar 头要几秒）。fd 跨 fork 共享是安全的，
        # 因为 read() 用 os.pread——它在偏移量处读、不移动文件位置。
        if "idx" not in idx_cache:
            idx_cache["idx"] = open_index(
                args.data,
                progress=lambda i, n, name: stage(f"扫描分片 {i+1}/{n} {name}"))
        ds._idx = idx_cache["idx"]
        if task == "vehicle" and augment:
            ds.keys = [k for k in ds.keys if county_index[k] >= 0]
        if args.overfit:
            subset_labels = ({k: {"adcode": str(county_index[k])} for k in ds.keys}
                             if task == "vehicle" else samples)
            ds.keys = stratified_subset(ds.keys, subset_labels, args.overfit, args.seed)
        return ds

    # 各类样本量分布：长尾有多长直接决定县级能做到什么程度
    from collections import Counter
    cnt = Counter(samples[k]["adcode"] for k, v in split.items() if v == "train")
    dist = Counter(cnt.values())
    print(f"训练集类别分布：{len(cnt)} 个县有样本；"
          f"≥100 条的 {sum(1 for v in cnt.values() if v >= 100)} 个，"
          f"<10 条的 {sum(1 for v in cnt.values() if v < 10)} 个")

    train_ds = dataset("train", augment=True)
    # Exact strided validation shards: no padding duplicates and no dropped tail.
    val_loaders = {}
    for name in ("val_same", "val_national", "test_county"):
        if not any(v == name and k in samples for k, v in split.items()):
            continue
        for mode in cfg.get("eval_modes", ["panorama"]):
            key = name if mode == "panorama" else f"{name}_{mode}"
            ds = dataset(name, False, mode)
            val_loaders[key] = make_loader(ds, cfg.get("eval_batch", 16), False,
                num_workers=cfg.get("workers", 2),
                sampler=range(rank, len(ds), world) if distributed else None)
    if cfg.get("selection_split", "val_same") not in val_loaders:
        raise ValueError("Configured selection_split has no validation samples")
    batch_size = cfg.get("batch_size", 8)
    n_cpu = _os.cpu_count() or 1
    workers = cfg.get("workers", 2)
    stage(f"CPU {n_cpu} 核，DataLoader {workers} 个 worker"
          + ("（worker 数超过核数会互相抢 CPU）" if workers > n_cpu else ""))
    # Keep shuffle RNG separate from worker seeding, including mid-epoch resume.
    train_sampler = torch.utils.data.RandomSampler(train_ds,
                        generator=torch.Generator().manual_seed(args.seed))
    if distributed:
        # drop_last 必须开：各进程批数不一致会让集合通信互相等待到死锁
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, shuffle=True, drop_last=True, seed=args.seed)
    train_loader = make_loader(train_ds, batch_size, True, sampler=train_sampler,
                               num_workers=cfg.get("workers", 2), seed=args.seed,
                               distributed=distributed)
    if not len(train_loader):
        raise ValueError("No training batches; reduce batch size")
    stage(f"就绪：训练 {len(train_ds):,} 条，验证 "
          f"{ {k: len(v.dataset) for k, v in val_loaders.items()} }")
    print(f"训练样本 {len(train_ds):,}  验证 {[f'{k}:{len(v.dataset)}' for k, v in val_loaders.items()]}")

    # ── 模型 ────────────────────────────────────────────────────────
    stage("构建模型…")
    model_cfg = dict(cfg)
    if args.init or args.resume:
        model_cfg["pretrained"] = False
    raw_model = build_model(len(adcodes), len(city_list), len(prov_list), model_cfg).to(device)
    if args.init:
        initial = torch.load(args.init, map_location="cpu", weights_only=False)
        initial_meta = initial.get("run_meta") or json.loads(
            (args.init.parent / "classes.json").read_text(encoding="utf-8"))
        for key in ("adcodes", "cities", "provinces", "backbone"):
            if initial_meta[key] != run_meta[key]:
                raise ValueError(f"Warm-start class/model mismatch: {key}")
        if initial_meta.get("task", "environment") != task:
            raise ValueError("Warm-start task mismatch")
        raw_model.load_state_dict(initial["model"])
        stage(f"Initialized from {args.init}; optimizer and scheduler start fresh")
    model = raw_model
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    stage(f"模型就绪：{n_par:.1f}M 参数，其中可训练 {n_tr:.1f}M")
    loss_fn = MultiTaskLoss(soft, **cfg.get("loss", {})).to(device)
    # 主干用低学习率微调而非冻结（冻结时 ImageNet 特征不足以分辨县级）。
    # 收全部参数：冻结参数 grad 为 None 会被自动跳过，但漏收的话之后解冻
    # 的参数就永远进不了优化器。
    backbone_scale = cfg.get("backbone_lr_scale", 0.1)
    backbone_ids = {id(q) for q in raw_model.backbone.parameters()}
    groups = [
        {"params": [q for q in model.parameters() if id(q) in backbone_ids],
         "lr": cfg.get("lr", 3e-4) * backbone_scale},
        {"params": [q for q in model.parameters() if id(q) not in backbone_ids],
         "lr": cfg.get("lr", 3e-4)},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.get("lr", 3e-4),
                            weight_decay=cfg.get("weight_decay", 0.05))
    # DDP 包装放在优化器之后：DistributedDataParallel 不转发属性访问，
    # 包上之后 model.backbone 就没了。它也不复制参数，优化器持有的仍是
    # 同一批对象，所以先后顺序对更新无影响。
    if distributed and cfg.get("freeze_backbone", False) and cfg.get("unfreeze_after", 0):
        raise ValueError("DDP staged unfreezing unsupported; use freeze_backbone: false")
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            raw_model, device_ids=[local_rank], find_unused_parameters=False,
            gradient_as_bucket_view=True, broadcast_buffers=False)
    epochs = cfg.get("epochs", 40)
    if args.overfit:
        # 原有轮数只有 160 次优化步，测的是"步数不够"而非"管线通不通"。
        # 预热 10% 在 3200 步下等于前 40 轮全在热身，一并压缩。
        epochs = max(epochs, cfg.get("overfit_epochs", 400)) if args.epochs is None else args.epochs
        cfg["accum"] = 1
        cfg["pct_start"] = 0.02
    steps_per_epoch = math.ceil(len(train_loader) / cfg.get("accum", 1))
    total_steps = epochs * steps_per_epoch
    pct_start = cfg.get("pct_start", 0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=total_steps, pct_start=pct_start)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    start_epoch, best, resume_batch = 0, -1.0, 0
    ckpt_path = args.out / "last.pt"
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # 用未包装的模型：存盘时已去掉 module. 前缀（见 save），
        # 在 DDP 包装体上加载会因键名不匹配而失败
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state.get("best", -1.0)
        if not state.get("epoch_complete", True):
            start_epoch, resume_batch = state["epoch"], state["next_batch"]
        if state.get("run_meta", run_meta) != run_meta:
            raise ValueError("Checkpoint metadata/config mismatch")
        expected = {"batches": len(train_loader), "world": world, "accum": cfg.get("accum", 1),
                    "total_steps": total_steps, "batch_size": batch_size}
        if state.get("loader_contract", expected) != expected:
            raise ValueError("Resume loader/scheduler changed; start a new run")
        apply_freeze_policy(raw_model, start_epoch, cfg)
        print(f"从 epoch {start_epoch} 续跑（当前最好 {best:.4f}，"
              f"主干{'可训练' if not raw_model.freeze_backbone else '冻结'}）")

    # ── 训练 ────────────────────────────────────────────────────────
    log_fh = (args.out / "log.jsonl").open("a", encoding="utf-8") if _IS_MAIN else None
    ckpt_minutes = cfg.get("ckpt_minutes", 30)
    last_ckpt = time.time()
    accum = cfg.get("accum", 1)
    eval_every = (args.eval_every if args.eval_every is not None
                  else cfg.get("eval_every", 1))
    if eval_every > 1:
        print(f"每 {eval_every} 轮验证一次")

    checkpoint_contract = {"batches": len(train_loader), "world": world, "accum": accum,
                           "total_steps": total_steps, "batch_size": batch_size}
    if _IS_MAIN:
        (args.out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    stage(f"开始训练：{epochs} 轮 × {steps_per_epoch} 步 = {total_steps} 次优化步")
    for epoch in range(start_epoch, epochs):
        # 不设 set_epoch 的话每个 epoch 的洗牌顺序完全相同，
        # 等于永远只看同一批数据
        if distributed:
            train_sampler.set_epoch(epoch)
        else:
            train_sampler.generator.manual_seed(args.seed + epoch)
        train_ds.set_epoch(0 if args.overfit else epoch)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(args.seed + epoch)
        model.train()
        was_frozen = raw_model.freeze_backbone
        apply_freeze_policy(raw_model, epoch, cfg)
        if was_frozen and not raw_model.freeze_backbone:
            n_tr = sum(p.numel() for p in model.parameters()
                       if p.requires_grad) / 1e6
            print(f"  [{time.time() - _T0:7.1f}s] epoch {epoch}：解冻主干，"
                  f"可训练参数 {n_tr:.1f}M", flush=True)

        do_eval = (eval_every <= 1 or epoch % eval_every == 0
                   or epoch == epochs - 1)
        t0, running, seen = time.time(), 0.0, 0
        t_data = t_compute = 0.0
        recent = deque(maxlen=20)
        hit_num = hit_den = 0
        opt.zero_grad(set_to_none=True)

        # 手动迭代才能把"等数据"与"算梯度"分开计时——数据是瓶颈时，
        # 这个比值是唯一能说明问题的证据。
        loader_it = iter(train_loader)
        n_batches = len(train_loader)
        last_log = time.time()
        for step in range(n_batches):
            t_batch = time.time()
            batch = next(loader_it)
            if epoch == start_epoch and step < resume_batch:
                continue
            data_sec = time.time() - t_batch
            t_data += data_sec
            t_work = time.time()

            views = batch["views"].to(device, non_blocking=True)
            vmask = batch["vmask"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True)
                       for k, v in batch.items()
                       if k in ("county", "city", "prov", "coord")}
            targets["vmask"] = vmask
            update = (step + 1) % accum == 0 or step + 1 == n_batches
            group_size = min(accum, n_batches - (step // accum) * accum)
            sync = model.no_sync() if distributed and not update else contextlib.nullcontext()
            with sync:
                with autocast_ctx(device, amp_dtype):
                    out = model(views, vmask, return_view_logits=bool(cfg.get("loss", {}).get("w_view", 0)))
                    loss, _ = loss_fn(out, targets, collect_parts=False)
                    loss = loss / group_size
                scaler.scale(loss).backward()

            if update:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.get("clip", 5.0))
                # 跳过时不要推进学习率调度：fp16 初始 scale 偏大，头几步
                # 可能因 inf/nan 被 GradScaler 跳过，此时若仍 sched.step()
                # 会白白消耗调度进度，并触发 "step() before optimizer.step()"
                # 警告。scale 只在跳过时下降，据此判断。
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= before and sched.last_epoch < sched.total_steps:
                    sched.step()
                opt.zero_grad(set_to_none=True)

            # 训练集上的即时准确率：冒烟测试里"能否过拟合"靠它看，
            # 不必等验证集跑完
            with torch.no_grad():
                tgt = targets["county"]
                ok = tgt >= 0
                if ok.any():
                    pred = out["county"].argmax(dim=1)
                    hit_num += int((pred[ok] == tgt[ok]).sum())
                    hit_den += int(ok.sum())

            running += float(loss.detach()) * group_size
            seen += 1
            compute_sec = time.time() - t_work
            t_compute += compute_sec
            recent.append((data_sec, compute_sec))

            # 按时间而非步数打点：步速随硬件变化，定步数要么刷屏要么没动静
            now = time.time()
            if now - last_log >= cfg.get("log_seconds", 15) or step == n_batches - 1:
                print(f"  [rank{rank} {now - _T0:7.1f}s] e{epoch:02d} {step+1}/{n_batches} "
                      f"步 {min(sched.last_epoch, total_steps)}/{total_steps} "
                      f"损失 {running/seen:.4f} 训练top1 "
                      f"{hit_num/max(1,hit_den):.3f} "
                      f"lr {_fmt_lr(opt)} "
                      f"recent{len(recent)}={sum(d+c for d,c in recent)/len(recent):.2f}s/batch "
                      f"data={sum(d for d,c in recent)/len(recent):.2f}s "
                      f"work={sum(c for d,c in recent)/len(recent):.2f}s "
                      f"epoch_avg={(now-t0)/max(1,seen):.2f}s/batch", flush=True)
                last_log = now

            # 到点就存盘——Colab 随时会断，不能等到 epoch 结束
            if _IS_MAIN and update and now - last_ckpt > ckpt_minutes * 60:
                checkpoint_start = time.time()
                save(ckpt_path, raw_model, opt, sched, scaler, epoch, best,
                     epoch_complete=False, next_batch=step + 1,
                     run_meta=run_meta, loader_contract=checkpoint_contract)
                print(f"  [{now - _T0:7.1f}s] checkpoint 已存：epoch {epoch} "
                      f"step {step} ({time.time()-checkpoint_start:.1f}s)", flush=True)
                last_ckpt = time.time()

        if distributed:
            totals = torch.tensor([running, seen, hit_num, hit_den], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(totals)
            total_running, total_seen, total_hit, total_den = totals.tolist()
        else:
            total_running, total_seen, total_hit, total_den = running, seen, hit_num, hit_den
        train_loss = total_running / max(1, total_seen)
        epoch_sec = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss,
               "train_acc": total_hit / max(1, total_den),
               "minutes": round(epoch_sec / 60, 1),
               "world_size": world, "effective_batch_size": batch_size * world * accum,
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else 0.0,
               "data_seconds": round(t_data, 1),
               "compute_seconds": round(t_compute, 1),
               "samples_per_sec": round(seen * batch_size * world / max(epoch_sec, 1e-6), 1)}
        left = (epochs - epoch - 1) * epoch_sec
        print(f"  e{epoch:02d} 损失 {train_loss:.4f} 训练top1 "
              f"{rec['train_acc']:.3f}  {epoch_sec:.0f}s "
              f"({t_data:.0f}s 等数据 / {t_compute:.0f}s 计算)  "
              f"{rec['samples_per_sec']:.0f} 样本/s  剩余约 {left/60:.0f} 分钟")

        if distributed:
            torch.distributed.barrier()
        if do_eval:
            for name, loader in val_loaders.items():
                if name.startswith("test_county") and epoch != epochs - 1:
                    continue
                # DDP collectives belong to training; evaluation uses independent shards.
                m = evaluate(raw_model, loader, device, amp_dtype, len(adcodes), cent)
                rec[name] = m
                stage(f"{name}: top1={m.get('top1', 0):.3f} top5={m.get('top5', 0):.3f}")
            selection = cfg.get("selection_split", "val_same")
            metric = cfg.get("selection_metric", "top5")
            score = rec.get(selection, {}).get(metric)
            if score is not None and score > best:
                best = score
                save(args.out / "best.pt", raw_model, opt, sched, scaler, epoch,
                     best, model_only=True, run_meta=run_meta)
                stage(f"New best {selection}/{metric}={best:.4f}")
        if _IS_MAIN:
            log_fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            log_fh.flush()
            save(ckpt_path, raw_model, opt, sched, scaler, epoch, best,
                 run_meta=run_meta, loader_contract=checkpoint_contract)
            last_ckpt = time.time()
        if distributed:
            torch.distributed.barrier()
        resume_batch = 0

    if log_fh:
        log_fh.close()
    if distributed:
        torch.distributed.destroy_process_group()
    stage(f"完成。最佳选择指标 {best:.4f}  产物 {args.out}")


def save(path, model, opt, sched, scaler, epoch, best, model_only=False,
         epoch_complete=True, next_batch=0, run_meta=None, loader_contract=None):
    """存 checkpoint。

    model_only=True 只存权重（约 112 MB），用于 best.pt——它是要长期保留、
    可能被导出或上传的产物，不需要优化器状态。last.pt 则必须完整
    （约 340 MB），否则无法续跑。
    """
    if not _IS_MAIN:
        return
    sd = model.state_dict()
    # 防御：万一传进来的是 DDP 包装过的模型，去掉 module. 前缀，
    # 否则推理端 load_state_dict 会因键名不匹配而失败
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    payload = {"model": sd, "epoch": epoch, "best": best,
               "epoch_complete": epoch_complete, "next_batch": next_batch,
               "run_meta": run_meta, "loader_contract": loader_contract}
    if not model_only:
        payload.update(opt=opt.state_dict(), sched=sched.state_dict(),
                       scaler=scaler.state_dict())
    partial = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, partial)
    os.replace(partial, path)


if __name__ == "__main__":
    main()

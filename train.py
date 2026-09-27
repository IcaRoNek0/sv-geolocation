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
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import PanoramaDataset, make_loader
from data.labels import (build_classes, centroids, city_of_adcode,
                         province_of_adcode, soft_targets)
from data.prepare import ViewConfig
from data.shards import ShardIndex, load_samples, load_split
from models.env_model import build_model
from models.losses import MultiTaskLoss
from utils.geo_utils import CountyPoints, denormalize_coords, normalize_coords
from utils.metrics import summarize


_T0 = time.time()


def stage(msg):
    """打印启动阶段，带累计耗时。

    必须 flush：Colab 的 stdout 走管道默认块缓冲，不刷的话会静默十几秒。
    """
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


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
        scores = out["county"].float().cpu().numpy()
        all_scores.append(scores)
        all_targets.append(batch["county"].numpy())
        # 必须 stack 成 (N, 2)，不能留着元组列表去 concatenate：
        # np.concatenate 会把每个 (lon, lat) 元组当一维序列拼起来，
        # 得到长度 2N 的平铺数组，后面的布尔索引会静默取错行。
        lon, lat = denormalize_coords(batch["coord"][:, 0].numpy(),
                                      batch["coord"][:, 1].numpy())
        all_true_coord.append(np.stack([lon, lat], axis=1))
    model.train()

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)
    valid = targets >= 0
    scores, targets = scores[valid], targets[valid]
    if len(targets) == 0:
        return {}

    m = summarize(scores, targets, n_classes,
                  class_centroids=cent if len(cent) == n_classes else None)

    # 距离误差只在两条线索融合选点后才有意义，这里先用 top-1 县的质心近似
    pred_loc = np.stack([
        cent[scores[i].argmax()] if len(cent) > scores[i].argmax() else (np.nan, np.nan)
        for i in range(len(scores))
    ])
    true_loc = np.concatenate(all_true_coord, axis=0)[valid]
    ok = np.isfinite(pred_loc).all(axis=1)
    if ok.any():
        from utils.metrics import distance_summary
        m.update({f"top1_{k}": v for k, v in
                  distance_summary(pred_loc[ok], true_loc[ok]).items()})
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

    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_amp_dtype(device)
    print(f"设备 {device}  精度 {amp_dtype}")
    if device.type == "cuda":
        print(f"GPU {torch.cuda.get_device_name(0)}  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        if amp_dtype is torch.float16:
            print("提示：当前是 Turing 及更早架构，只能 fp16；若换到 L4/A100 会自动切 bf16")

    args.out.mkdir(parents=True, exist_ok=True)
    stage(f"设备 {device}  精度 {amp_dtype}")
    shard_paths = sorted(args.data.glob("*.tar"))
    if not shard_paths:
        raise SystemExit(f"{args.data} 下没有 .tar 分片——第 ⑥ 格拷盘跑了吗？")
    samples = load_samples(args.data / "samples.jsonl")
    stage(f"元数据 {len(samples):,} 条，分片 {len(shard_paths)} 个")
    split, split_payload = load_split(args.data / "split.json")
    cp = CountyPoints(args.points)
    stage(f"划分与点位表就绪（点位表 {len(cp)} 县）")

    adcodes, class_index = build_classes(list(samples.values()), cp)
    cent = centroids(adcodes, cp)
    stage(f"类别空间 {len(adcodes)} 县，质心已算")
    soft = soft_targets(cent, half_km=cfg.get("soft_half_km", 50.0))
    stage(f"地理软标签矩阵 {soft.shape[0]}×{soft.shape[1]} 已算")
    (county_index, coord_index, city_index, prov_index,
     city_list, prov_list) = build_indices(samples, class_index)
    print(f"类别 {len(adcodes)} 县 / {len(city_list)} 市 / {len(prov_list)} 省")

    # 类别表随 checkpoint 一起存：推理端靠它把 logit 下标映回县码。
    # 不这样做的话，推理时重建类别表一旦与训练时不一致，预测会被静默地
    # 解释成别的县——而且完全看不出来。
    (args.out / "classes.json").write_text(json.dumps({
        "adcodes": adcodes,
        "cities": city_list,
        "provinces": prov_list,
        "backbone": cfg.get("backbone", "convnext_tiny"),
        "views": cfg.get("views", {}),
        "soft_half_km": cfg.get("soft_half_km", 50.0),
        "seed": args.seed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def dataset(split_name, augment):
        ds = PanoramaDataset(
            shard_paths, split, samples, county_index, coord_index,
            city_index, prov_index, split=split_name, view_cfg=vcfg,
            augment=augment, seed=args.seed)
        # 索引只在父进程建一次（扫 tar 头要几秒）。fd 跨 fork 共享是安全的，
        # 因为 read() 用 os.pread——它在偏移量处读、不移动文件位置。
        if "idx" not in idx_cache:
            idx_cache["idx"] = ShardIndex(
                shard_paths,
                progress=lambda i, n, name: stage(f"扫描分片 {i+1}/{n} {name}"))
        ds._idx = idx_cache["idx"]
        if args.overfit:
            ds.keys = stratified_subset(ds.keys, samples, args.overfit, args.seed)
        return ds

    # 各类样本量分布：长尾有多长直接决定县级能做到什么程度
    from collections import Counter
    cnt = Counter(samples[k]["adcode"] for k, v in split.items() if v == "train")
    dist = Counter(cnt.values())
    print(f"训练集类别分布：{len(cnt)} 个县有样本；"
          f"≥100 条的 {sum(1 for v in cnt.values() if v >= 100)} 个，"
          f"<10 条的 {sum(1 for v in cnt.values() if v < 10)} 个")

    train_ds = dataset("train", augment=True)
    val_loaders = {}
    for name in ("val_same", "val_national", "test_county"):
        try:
            val_loaders[name] = make_loader(dataset(name, augment=False),
                                            cfg.get("eval_batch", 16), False,
                                            num_workers=cfg.get("workers", 2))
        except ValueError:
            print(f"划分 {name} 为空，跳过")
    batch_size = cfg.get("batch_size", 8)
    import os as _os
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    n_cpu = _os.cpu_count() or 1
    workers = cfg.get("workers", 2)
    stage(f"CPU {n_cpu} 核，DataLoader {workers} 个 worker"
          + ("（worker 数超过核数会互相抢 CPU）" if workers > n_cpu else ""))
    train_loader = make_loader(train_ds, batch_size, True,
                               num_workers=cfg.get("workers", 2), seed=args.seed)
    stage(f"就绪：训练 {len(train_ds):,} 条，验证 "
          f"{ {k: len(v.dataset) for k, v in val_loaders.items()} }")
    print(f"训练样本 {len(train_ds):,}  验证 {[f'{k}:{len(v.dataset)}' for k, v in val_loaders.items()]}")

    # ── 模型 ────────────────────────────────────────────────────────
    stage("构建模型…")
    model = build_model(len(adcodes), len(city_list), len(prov_list), cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    stage(f"模型就绪：{n_par:.1f}M 参数，其中可训练 {n_tr:.1f}M")
    loss_fn = MultiTaskLoss(soft, **cfg.get("loss", {})).to(device)
    # 主干用低学习率微调而非冻结（冻结时 ImageNet 特征不足以分辨县级）。
    # 收全部参数：冻结参数 grad 为 None 会被自动跳过，但漏收的话之后解冻
    # 的参数就永远进不了优化器。
    backbone_scale = cfg.get("backbone_lr_scale", 0.1)
    backbone_ids = {id(q) for q in model.backbone.parameters()}
    groups = [
        {"params": [q for q in model.parameters() if id(q) in backbone_ids],
         "lr": cfg.get("lr", 3e-4) * backbone_scale},
        {"params": [q for q in model.parameters() if id(q) not in backbone_ids],
         "lr": cfg.get("lr", 3e-4)},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.get("lr", 3e-4),
                            weight_decay=cfg.get("weight_decay", 0.05))
    epochs = cfg.get("epochs", 40)
    if args.overfit:
        # 原有轮数只有 160 次优化步，测的是"步数不够"而非"管线通不通"。
        # 预热 10% 在 3200 步下等于前 40 轮全在热身，一并压缩。
        epochs = max(epochs, cfg.get("overfit_epochs", 400))
        cfg["accum"] = 1
        cfg["pct_start"] = 0.02
    steps_per_epoch = max(1, len(train_loader) // cfg.get("accum", 1))
    total_steps = epochs * steps_per_epoch
    pct_start = cfg.get("pct_start", 0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=total_steps, pct_start=pct_start)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    start_epoch, best = 0, -1.0
    ckpt_path = args.out / "last.pt"
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state.get("best", -1.0)
        apply_freeze_policy(model, start_epoch, cfg)
        print(f"从 epoch {start_epoch} 续跑（当前最好 {best:.4f}，"
              f"主干{'可训练' if not model.freeze_backbone else '冻结'}）")

    # ── 训练 ────────────────────────────────────────────────────────
    log_fh = (args.out / "log.jsonl").open("a", encoding="utf-8")
    ckpt_minutes = cfg.get("ckpt_minutes", 30)
    last_ckpt = time.time()
    accum = cfg.get("accum", 1)
    eval_every = (args.eval_every if args.eval_every is not None
                  else cfg.get("eval_every", 1))
    if eval_every > 1:
        print(f"每 {eval_every} 轮验证一次")

    stage(f"开始训练：{epochs} 轮 × {steps_per_epoch} 步 = {total_steps} 次优化步")
    for epoch in range(start_epoch, epochs):
        model.train()
        was_frozen = model.freeze_backbone
        apply_freeze_policy(model, epoch, cfg)
        if was_frozen and not model.freeze_backbone:
            n_tr = sum(p.numel() for p in model.parameters()
                       if p.requires_grad) / 1e6
            print(f"  [{time.time() - _T0:7.1f}s] epoch {epoch}：解冻主干，"
                  f"可训练参数 {n_tr:.1f}M", flush=True)

        do_eval = (eval_every <= 1 or epoch % eval_every == 0
                   or epoch == epochs - 1)
        t0, running, seen = time.time(), 0.0, 0
        t_data = t_compute = 0.0
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
            t_data += time.time() - t_batch
            t_work = time.time()

            views = batch["views"].to(device, non_blocking=True)
            vmask = batch["vmask"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True)
                       for k, v in batch.items()
                       if k in ("county", "city", "prov", "coord")}
            with autocast_ctx(device, amp_dtype):
                out = model(views, vmask)
                loss, _ = loss_fn(out, targets)
                loss = loss / accum
            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.get("clip", 5.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()

            # 训练集上的即时准确率：冒烟测试里"能否过拟合"靠它看，
            # 不必等验证集跑完
            with torch.no_grad():
                tgt = targets["county"]
                ok = tgt >= 0
                if ok.any():
                    pred = out["county"].argmax(dim=1)
                    hit_num += int((pred[ok] == tgt[ok]).sum())
                    hit_den += int(ok.sum())

            running += float(loss.detach()) * accum
            seen += 1
            t_compute += time.time() - t_work

            # 按时间而非步数打点：步速随硬件变化，定步数要么刷屏要么没动静
            now = time.time()
            if now - last_log >= cfg.get("log_seconds", 15) or step == n_batches - 1:
                print(f"  [{now - _T0:7.1f}s] e{epoch:02d} {step+1}/{n_batches} "
                      f"步 {min(sched.last_epoch, total_steps)}/{total_steps} "
                      f"损失 {running/seen:.4f} 训练top1 "
                      f"{hit_num/max(1,hit_den):.3f} "
                      f"lr {opt.param_groups[0]['lr']:.1e} "
                      f"{(now-t0)/max(1,seen):.2f}s/batch", flush=True)
                last_log = now

            # 到点就存盘——Colab 随时会断，不能等到 epoch 结束
            if now - last_ckpt > ckpt_minutes * 60:
                save(ckpt_path, model, opt, sched, scaler, epoch, best)
                print(f"  [{now - _T0:7.1f}s] checkpoint 已存：epoch {epoch} "
                      f"step {step}", flush=True)
                last_ckpt = now

        train_loss = running / max(1, seen)
        epoch_sec = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss,
               "train_acc": hit_num / max(1, hit_den),
               "minutes": round(epoch_sec / 60, 1),
               "data_seconds": round(t_data, 1),
               "compute_seconds": round(t_compute, 1),
               "samples_per_sec": round(seen * batch_size / max(epoch_sec, 1e-6), 1)}
        left = (epochs - epoch - 1) * epoch_sec
        print(f"  e{epoch:02d} 损失 {train_loss:.4f} 训练top1 "
              f"{rec['train_acc']:.3f}  {epoch_sec:.0f}s "
              f"({t_data:.0f}s 等数据 / {t_compute:.0f}s 计算)  "
              f"{rec['samples_per_sec']:.0f} 样本/s  剩余约 {left/60:.0f} 分钟")

        if not do_eval:
            log_fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            log_fh.flush()
            continue

        for name, loader in val_loaders.items():
            m = evaluate(model, loader, device, amp_dtype, len(adcodes), cent)
            rec[name] = m
            if name == "val_same":
                print(f"  e{epoch} 同县留出 top1 {m.get('top1', 0):.3f} "
                      f"top5 {m.get('top5', 0):.3f} "
                      f"宏平均 {m.get('macro_recall', 0):.3f} "
                      f"多数类基线 {m.get('majority_baseline', 0):.3f} "
                      f"中位误差 {m.get('top1_median_km', float('nan')):.1f} km")
                print(f"        候选邻近率(top5 落在真值 150km 内) "
                      f"{m.get('top5_nearby_150km', float('nan')):.3f}")
                score = m.get("top5", 0.0)
                if score > best:
                    best = score
                    save(args.out / "best.pt", model, opt, sched, scaler, epoch, best)
                    print(f"        新最好，已存 best.pt")

        log_fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        log_fh.flush()
        save(ckpt_path, model, opt, sched, scaler, epoch, best)
        last_ckpt = time.time()

    log_fh.close()
    print(f"完成。最好 top5 {best:.4f}  产物 {args.out}")


def save(path, model, opt, sched, scaler, epoch, best):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best": best,
    }, path)


if __name__ == "__main__":
    main()

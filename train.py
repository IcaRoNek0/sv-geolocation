#!/usr/bin/env python
"""训练主程序。

设计约束来自 Colab（见 PLAN.md §2.2）：

- T4 是 Turing，**不支持 bf16**，只能 fp16 + GradScaler；sm≥8 才用 bf16。
  这里自动判断，不写死。
- 会话 4–6 小时就会被回收，所以每 --ckpt-minutes 存一次 checkpoint，
  --resume 从断点续跑。**分片顺序用固定 seed**，否则续跑后数据顺序漂移，
  等于偷换了训练集。
- /content 会随会话清空，所以 checkpoint 与日志必须写到挂载的 Drive 路径。

用法：
    python train.py --config configs/default.yaml --data data/shards \\
                    --out /content/drive/MyDrive/sv/runs/base
    python train.py --config configs/default.yaml --out ... --resume
    python train.py --config configs/default.yaml --data data/shards \\
                    --overfit 64          # M0 冒烟：应当迅速过拟合到接近 100%
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import PanoramaDataset, make_loader
from data.labels import (build_classes, centroids, city_of_adcode,
                         hierarchical_labels, province_of_adcode, soft_targets)
from data.prepare import ViewConfig
from data.shards import ShardIndex, load_samples, load_split
from models.env_model import build_model
from models.losses import MultiTaskLoss
from utils.geo_utils import CountyPoints, denormalize_coords, normalize_coords
from utils.metrics import summarize


def pick_amp_dtype(device):
    """T4（sm_75）不支持 bf16，只能 fp16；sm≥8 用 bf16 更稳。"""
    if device.type != "cuda":
        return None
    cc = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if cc[0] >= 8 else torch.float16


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
    all_scores, all_targets, all_coord, all_true_coord = [], [], [], []
    losses = []
    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        vmask = batch["vmask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model(views, vmask)
        if loss_fn is not None:
            b = {k: v.to(device) for k, v in batch.items()
                 if k in ("county", "city", "prov", "coord")}
            _, parts = loss_fn(out, b)
            losses.append(parts)
        scores = out["county"].float().cpu().numpy()
        coord = out["coord"].float().cpu().numpy()
        all_scores.append(scores)
        all_targets.append(batch["county"].numpy())
        all_coord.append(denormalize_coords(coord[:, 0], coord[:, 1]))
        all_true_coord.append(denormalize_coords(batch["coord"][:, 0].numpy(),
                                                 batch["coord"][:, 1].numpy()))
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
    true_loc = np.concatenate(all_true_coord)[valid]
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
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

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
    shard_paths = sorted(args.data.glob("*.tar"))
    samples = load_samples(args.data / "samples.jsonl")
    split, split_payload = load_split(args.data / "split.json")
    cp = CountyPoints(args.points)

    adcodes, class_index = build_classes(list(samples.values()), cp)
    cent = centroids(adcodes, cp)
    soft = soft_targets(cent, half_km=cfg.get("soft_half_km", 50.0))
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
    idx_cache = {}          # 各划分共用一份索引，避免重复扫 tar

    def dataset(split_name, augment):
        ds = PanoramaDataset(
            shard_paths, split, samples, county_index, coord_index,
            city_index, prov_index, split=split_name, view_cfg=vcfg,
            augment=augment, seed=args.seed)
        ds._idx = idx_cache.setdefault("idx", ShardIndex(shard_paths))
        if args.overfit:
            ds.keys = ds.keys[: args.overfit]
        return ds

    train_ds = dataset("train", augment=True)
    val_loaders = {}
    for name in ("val_same", "val_national", "test_county"):
        try:
            val_loaders[name] = make_loader(dataset(name, augment=False),
                                            cfg.get("eval_batch", 16), False,
                                            num_workers=cfg.get("workers", 2))
        except ValueError:
            print(f"划分 {name} 为空，跳过")
    train_loader = make_loader(train_ds, cfg.get("batch_size", 8), True,
                               num_workers=cfg.get("workers", 2), seed=args.seed)
    print(f"训练样本 {len(train_ds):,}  验证 {[f'{k}:{len(v.dataset)}' for k, v in val_loaders.items()]}")

    # ── 模型 ────────────────────────────────────────────────────────
    model = build_model(len(adcodes), len(city_list), len(prov_list), cfg).to(device)
    loss_fn = MultiTaskLoss(soft, **cfg.get("loss", {})).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 0.05))
    epochs = cfg.get("epochs", 40)
    steps_per_epoch = max(1, len(train_loader) // cfg.get("accum", 1))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.get("lr", 3e-4), total_steps=epochs * steps_per_epoch,
        pct_start=0.1)
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
        print(f"从 epoch {start_epoch} 续跑（当前最好 {best:.4f}）")

    # ── 训练 ────────────────────────────────────────────────────────
    log_fh = (args.out / "log.jsonl").open("a", encoding="utf-8")
    ckpt_minutes = cfg.get("ckpt_minutes", 30)
    last_ckpt = time.time()
    accum = cfg.get("accum", 1)

    for epoch in range(start_epoch, epochs):
        model.train()
        if (cfg.get("unfreeze_after", 0) and epoch == cfg["unfreeze_after"]
                and model.freeze_backbone):
            model.set_backbone_trainable(True)
            print(f"epoch {epoch}：解冻主干")

        t0, running, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            views = batch["views"].to(device, non_blocking=True)
            vmask = batch["vmask"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True)
                       for k, v in batch.items()
                       if k in ("county", "city", "prov", "coord")}
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
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

            running += float(loss) * accum
            seen += 1
            if step % cfg.get("log_every", 20) == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"  e{epoch} {step}/{len(train_loader)} "
                      f"loss {running/max(1,seen):.4f} lr {lr:.2e}")

            # 到点就存盘——Colab 随时会断，不能等到 epoch 结束
            if time.time() - last_ckpt > ckpt_minutes * 60:
                save(ckpt_path, model, opt, sched, scaler, epoch, best)
                print(f"  [checkpoint] epoch {epoch} step {step}")
                last_ckpt = time.time()

        train_loss = running / max(1, seen)
        rec = {"epoch": epoch, "train_loss": train_loss,
               "minutes": round((time.time() - t0) / 60, 1)}

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

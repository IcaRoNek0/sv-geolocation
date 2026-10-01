#!/usr/bin/env python
"""准确率 vs 与最近训练样本的距离——区分"学会了县"与"记住了街"。

    python tools/analyze_recall.py --run round2 --n 120

对每条样本，先在**同县**内找最近的训练样本（按 split.json 的 train 划分），
再按该距离分桶统计 top1/top5。若准确率随距离迅速塌掉，说明模型靠的是具体
街景的复现而不是可迁移的地理线索。

同时把 round2/eval150 的野外点位也算一遍同一指标，放在同一条曲线上比较。
"""
import argparse
import json
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR / "tools"))

import numpy as np                                              # noqa: E402

import data.shards as sh                                        # noqa: E402
from data.prepare import ViewConfig, make_sample                # noqa: E402
from inference_onnx import quiet_stderr                         # noqa: E402
from models.fusion import fuse, predict_location, top_counties  # noqa: E402
from utils.divisions import describe, load_names                # noqa: E402
from eval_common import read_positions, truth_coordinates
from utils.geo_utils import CountyPoints, haversine             # noqa: E402

BUCKETS = [(0, 0.1), (0.1, 1), (1, 5), (5, 20), (20, 100), (100, 1e9)]


def nearest_train_km(coord, trains):
    """(lng,lat) 到同县训练样本的最近距离；该县没有训练样本则 None。"""
    pts = trains.get(coord[2])
    if not pts:
        return None
    lng, lat = coord[0], coord[1]
    return min(haversine(lng, lat, p[0], p[1]) for p in pts) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=AI_DIR / "data" / "shards")
    ap.add_argument("--work", type=Path, default=AI_DIR / "round2" / "eval150")
    ap.add_argument("--points", type=Path,
                    default=AI_DIR / "data" / "pool" / "county_points.npz")
    ap.add_argument("--n", type=int, default=120, help="每个划分抽多少条")
    ap.add_argument("--eval-mode", choices=("single", "panorama"), default="panorama")
    ap.add_argument("--positions", type=Path, help="原始位置 JSON，旧结果必须提供")
    ap.add_argument("--output", type=Path, help="默认写入 work/recall_corrected.jsonl")
    args = ap.parse_args()
    positions = read_positions(args.positions) if args.positions else None

    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    adcodes = meta["adcodes"]
    v = meta.get("views", {})
    size, fov, n_max = v.get("size", 224), v.get("fov", 90.0), v.get("n_max", 4)
    cfg = ViewConfig(**v)

    with quiet_stderr():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.run / "model.onnx"),
                                    providers=["CPUExecutionProvider"])
    cp = CountyPoints(args.points)

    samples = {}
    for line in (args.data / "samples.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        samples[r["panoid"]] = r
    split = json.loads((args.data / "split.json").read_text(
        encoding="utf-8"))["assignments"]

    trains = defaultdict(list)
    for k, s in split.items():
        if s == "train" and k in samples:
            r = samples[k]
            trains[r["adcode"]].append((r["lng"], r["lat"]))

    idx = sh.open_index(args.data)

    def predict(img, seed):
        # 种子必须逐样本不同：make_sample 随机决定视图数（15% 只给单视图）和
        # 起始朝向，若所有样本共用一次抽样，就等于拿同一套视图评整个集合。
        views, vmask = make_sample(img, np.random.default_rng(seed), cfg,
                                   augment_on=False, eval_mode=args.eval_mode)
        x = views[None].transpose(0, 1, 4, 2, 3)
        out = sess.run(["county"], {"views": x, "vmask": vmask[None]})[0][0]
        probs = fuse(out, adcodes=adcodes, from_logits=True)
        top = [a for a, _ in top_counties(probs, k=5)]
        lon, lat, _ = predict_location(probs, cp, top_k=20)
        return top, max(probs.values()), (lon, lat)

    rows = []
    excluded = defaultdict(int)
    rng = random.Random(20260929)
    for sp in ("val_same", "val_national"):
        keys = [k for k, s in split.items()
                if s == sp and k in samples and samples[k]["adcode"] in set(adcodes)]
        keys = rng.sample(keys, min(args.n, len(keys)))
        print(f"{sp}: {len(keys)} 条", flush=True)
        for i, k in enumerate(keys):
            r = samples[k]
            d = nearest_train_km((r["lng"], r["lat"], r["adcode"]), trains)
            if d is None:
                excluded[sp] += 1
                continue
            img = sh.decode_jpeg(idx.read(k), min_width=4 * size)
            top, prob, (lon, lat) = predict(img, zlib.crc32(k.encode()))
            rows.append({
                "panoid": k, "src": sp, "truth": r["adcode"], "d_train": d,
                "hit1": r["adcode"] == top[0], "hit5": r["adcode"] in top,
                "prob": prob,
                "km": haversine(lon, lat, r["lng"], r["lat"]) / 1000,
            })
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(keys)}", flush=True)

    # 野外 150 条：距离用的是各自真值点到训练样本
    wf = args.work / "results.jsonl"
    if wf.exists():
        for line in wf.open(encoding="utf-8"):
            r = json.loads(line)
            if not r["truth"]:
                continue
            lon, lat = truth_coordinates([r], positions)[0]
            d = nearest_train_km((lon, lat, r["truth"]), trains)
            if d is None:
                excluded["野外150"] += 1
                continue
            rows.append({"panoid": r["panoid"], "src": "野外150",
                         "truth": r["truth"], "d_train": d, "hit1": r["hit1"],
                         "hit5": r["hit5"], "prob": r["top"][0]["prob"],
                         "km": r["km"]})

    output = args.output or args.work / "recall_corrected.jsonl"
    output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")

    names = load_names()
    print("\n" + "=" * 72)
    print("准确率 vs 与最近训练样本的距离（同县内）")
    print("没有同县训练点、未计入距离统计的样本:", dict(excluded))
    print(f"{'距离桶(km)':>16} {'n':>5} {'top1':>7} {'top5':>7} "
          f"{'中位选点误差':>10} {'top1概率':>8}  分源")
    for lo, hi in BUCKETS:
        g = [r for r in rows if lo <= r["d_train"] < hi]
        if not g:
            continue
        lab = f"[{lo},{hi if hi < 1e8 else '∞'})"
        srcs = defaultdict(int)
        for r in g:
            srcs[r["src"]] += 1
        print(f"{lab:>16} {len(g):5d} {sum(r['hit1'] for r in g)/len(g)*100:6.1f}% "
              f"{sum(r['hit5'] for r in g)/len(g)*100:6.1f}% "
              f"{sorted(r['km'] for r in g)[len(g)//2]:9.0f}km "
              f"{np.mean([r['prob'] for r in g]):8.3f}  "
              + " ".join(f"{k}:{v}" for k, v in sorted(srcs.items())))

    print("\n按来源汇总")
    for src in ("val_same", "val_national", "野外150"):
        g = [r for r in rows if r["src"] == src]
        if not g:
            continue
        print(f"  {src:12s} n={len(g):4d}  top1 {sum(r['hit1'] for r in g)/len(g)*100:5.1f}%"
              f"  top5 {sum(r['hit5'] for r in g)/len(g)*100:5.1f}%"
              f"  中位距离 {sorted(r['d_train'] for r in g)[len(g)//2]:7.0f}km"
              f"  中位误差 {sorted(r['km'] for r in g)[len(g)//2]:7.0f}km")
    print(f"\n明细 {output}")


if __name__ == "__main__":
    main()

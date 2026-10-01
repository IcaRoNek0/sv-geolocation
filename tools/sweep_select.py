#!/usr/bin/env python
"""选点超参扫描：温度 × top_k。

    python tools/sweep_select.py --run round2 --work round2/eval150

Candidate-oracle distances are diagnostics, not attainable prediction accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR / "tools"))

import numpy as np                                              # noqa: E402
from PIL import Image                                           # noqa: E402

from inference_onnx import quiet_stderr, to_views               # noqa: E402
from models.fusion import fuse, predict_location, top_counties  # noqa: E402
from utils.divisions import CountyLocator, load_names           # noqa: E402
from eval_common import cache_identity, read_positions, truth_coordinates
from utils.geo_utils import CountyPoints, haversine             # noqa: E402

TEMPS = (0.2, 0.35, 0.5, 0.7, 1.0, 1.5)
KS = (1, 3, 5, 10, 20, 50)


def collect(args, cache):
    rows = [json.loads(l) for l in
            (args.work / "results.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["truth"]]
    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    v = meta.get("views", {})
    size, fov, n_max = v.get("size", 224), v.get("fov", 90.0), v.get("n_max", 4)

    identity = cache_identity(args.run, rows, args.work)
    if cache.exists():
        with np.load(cache) as z:
            if "identity" in z and json.loads(str(z["identity"])) == identity:
                return rows, z["logits"], z["truth"], meta["adcodes"]
        print("Cache provenance missing or changed; recomputing logits.", flush=True)

    with quiet_stderr():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.run / "model.onnx"),
                                    providers=["CPUExecutionProvider"])
    logits, truth = [], []
    for i, r in enumerate(rows):
        img = np.asarray(Image.open(
            args.work / "images" / f"{r['panoid']}.jpg").convert("RGB"))
        views, vmask, _ = to_views(img, True, 8, fov, size, n_max)
        x = views[None].transpose(0, 1, 4, 2, 3)
        out = sess.run(["county"], {"views": x, "vmask": vmask[None]})[0][0]
        logits.append(out.astype(np.float32))
        truth.append(r["truth"])
        if (i + 1) % 30 == 0:
            print(f"  推理 {i+1}/{len(rows)}", flush=True)
    lg = np.stack(logits)
    np.savez_compressed(cache, logits=lg, truth=np.array(truth),
                        identity=np.array(json.dumps(identity)))
    print(f"已缓存 logits {lg.shape} → {cache}")
    return rows, lg, np.array(truth), meta["adcodes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--points", type=Path,
                    default=AI_DIR / "data" / "pool" / "county_points.npz")
    ap.add_argument("--positions", type=Path, help="原始位置 JSON，旧结果必须提供")
    ap.add_argument("--cache", type=Path, help="默认新建 logits_v2.npz，保留旧缓存")
    args = ap.parse_args()

    rows, logits, truth, adcodes = collect(args, args.cache or args.work / "logits_v2.npz")
    cp = CountyPoints(args.points)
    positions = read_positions(args.positions) if args.positions else None
    coord = truth_coordinates(rows, positions)
    n = len(rows)

    # 分类器自身的候选质量：top1 / top5 / top20 里最近候选县的质心
    cen = {a: cp.centroid(a) for a in adcodes if a in cp}
    print("\n分类器候选质量（候选县质心到真值，km）")
    for k in (1, 5, 20, 100):
        ds = []
        for i in range(n):
            order = np.argsort(-logits[i])[:k]
            d = [haversine(cen[adcodes[j]][0], cen[adcodes[j]][1],
                           coord[i][0], coord[i][1]) / 1000
                 for j in order if adcodes[j] in cen]
            ds.append(min(d) if d else 1e9)
        ds.sort()
        print(f"  top{k:<4d} 中位 {ds[n//2]:7.0f}   p25 {ds[n//4]:7.0f}"
              f"   p75 {ds[3*n//4]:7.0f}")

    print(f"\n选点扫描（{n} 条）  中位/均值误差 km，50km 内比例")
    hdr = "  温度 " + "".join(f"{'k='+str(k):>22}" for k in KS)
    print(hdr)
    best = None
    for t in TEMPS:
        cells = []
        for k in KS:
            km = []
            for i in range(n):
                probs = fuse(logits[i].astype(np.float64), adcodes=adcodes,
                             temperature=t, from_logits=True)
                lon, lat, _ = predict_location(probs, cp, top_k=k)
                km.append(haversine(lon, lat, coord[i][0], coord[i][1]) / 1000)
            km = np.array(km)
            med = float(np.median(km))
            cells.append(f"{med:6.0f}/{km.mean():5.0f} "
                         f"{(km <= 50).mean()*100:4.0f}%")
            if best is None or med < best[0]:
                best = (med, t, k, float(km.mean()), float((km <= 50).mean()))
        print(f"  {t:4.2f} " + "".join(f"{c:>22}" for c in cells))
    print(f"\n本集合探索性最优（需独立验证）：温度 {best[1]}  top_k {best[2]}  "
          f"中位 {best[0]:.0f}km  均值 {best[3]:.0f}km  50km 内 {best[4]*100:.0f}%")

    # 参考上界：直接取 top1 县的点位质心
    km = []
    for i in range(n):
        a = adcodes[int(np.argmax(logits[i]))]
        lo, la = cen[a]
        km.append(haversine(lo, la, coord[i][0], coord[i][1]) / 1000)
    km = np.array(km)
    print(f"参考：top1 县质心        中位 {np.median(km):.0f}km  "
          f"均值 {km.mean():.0f}km  50km 内 {(km<=50).mean()*100:.0f}%")


if __name__ == "__main__":
    main()

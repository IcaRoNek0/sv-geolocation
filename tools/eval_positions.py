#!/usr/bin/env python
"""批量评估：一批 (panoID, 坐标) → 抓图 → ONNX 推理 → 命中率与误差统计。

    python tools/eval_positions.py --run round2 \
        --json "round2/China（150 个位置）.json" --work round2/eval150 --jobs 6

真值取 json 坐标的点面判定（已验证与主库 GCJ02 逐条相等），在线 sdata 的
X/Y 作为独立复核；两者不一致会单独列出。图与真值缓存在 --work 下，重跑只
做推理。

抓图统一 1024 宽：decode_jpeg(min_width=896) 会把 2048 的图 draft 回 1024，
而 1024 的原样保留，两者进模型的效果相同。
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR / "tools"))

import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402

from inference_onnx import quiet_stderr, to_views              # noqa: E402
from models.fusion import fuse, predict_location, top_counties  # noqa: E402
from utils.divisions import (CountyLocator, describe, load_names,  # noqa: E402
                             province_code)
from utils.geo_utils import CountyPoints, haversine            # noqa: E402

FETCH_PX = 1024
NEAR_KM = 50.0            # 选点落在此半径内算"抓到了区域"


def truth_at(locator, lng, lat):
    """点面判定；边界线上 contains 为假，退化成极小缓冲再试一次。"""
    code = locator.at(lng, lat)
    if code:
        return code
    from shapely.geometry import Point
    pt = Point(lng, lat).buffer(1e-5)
    for c, g in locator.entries:
        if g.intersects(pt):
            return c
    return None


def fetch_all(todos, img_dir, truth_path):
    """并发抓图，边抓边写 truth.jsonl（中断也不丢已完成的部分）。"""
    import fetch_one

    sys.path.insert(0, str(AI_DIR.parent))
    from sv_coordinates import convert

    lock = __import__("threading").Lock()
    fh = truth_path.open("a", encoding="utf-8")

    def one(pos):
        pid = pos["panoId"]
        t0 = time.time()
        try:
            img, obj = fetch_one.fetch_pano(pid, FETCH_PX)
            img.save(img_dir / f"{pid}.jpg", "JPEG", quality=92)
            lng, lat = convert(float(obj["X"]) / 100.0, float(obj["Y"]) / 100.0,
                               "bd09mc", "gcj02")
            rec = {
                "panoid": pid, "ok": True,
                "online_lng": lng, "online_lat": lat,
                "online_adcode": truth_at(locator_global, lng, lat),
                "width": img.width, "height": img.height,
                "date": obj.get("Date"), "type": obj.get("Type"),
                "obsolete": obj.get("Obsolete"),
                "seconds": round(time.time() - t0, 1),
            }
        except BaseException as e:
            rec = {"panoid": pid, "ok": False,
                   "error": f"{type(e).__name__}: {e}",
                   "seconds": round(time.time() - t0, 1)}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{len(done)+1}/{len(todos)}] {pid} "
                  f"{'ok' if rec['ok'] else rec['error'][:60]}  {rec['seconds']}s",
                  flush=True)
            done.append(rec)
        return rec

    done = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(one, todos))
    fh.close()
    return done


def load_truth(truth_path, positions):
    recs = {}
    if truth_path.exists():
        for line in truth_path.open(encoding="utf-8"):
            r = json.loads(line)
            recs[r["panoid"]] = r
    out = []
    for pos in positions:
        r = recs.get(pos["panoId"], {"panoid": pos["panoId"], "ok": False,
                                     "error": "未抓取"})
        r["json_lng"], r["json_lat"] = pos["lng"], pos["lat"]
        r["heading"] = pos.get("heading")
        r["truth_adcode"] = truth_at(locator_global, pos["lng"], pos["lat"])
        r["truth_lng"], r["truth_lat"] = pos["lng"], pos["lat"]
        out.append(r)
    return out


def evaluate(recs, args):
    with quiet_stderr():
        import onnxruntime as ort

    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    vcfg = meta.get("views", {})
    size, fov, n_max = vcfg.get("size", 224), vcfg.get("fov", 90.0), vcfg.get("n_max", 4)
    adcodes = meta["adcodes"]
    cls = set(adcodes)

    onnx_path = args.model or (args.run / "model.onnx")
    with quiet_stderr():
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
    print(f"\n模型 {onnx_path.name} {onnx_path.stat().st_size/1e6:.0f} MB  "
          f"{len(adcodes)} 类  视图 {size}px fov{fov:.0f} n<={n_max}")

    cp = CountyPoints(args.points)
    rows = []
    for i, r in enumerate(recs):
        pid = r["panoid"]
        path = args.work / "images" / f"{pid}.jpg"
        if not r.get("ok") or not path.exists():
            continue
        try:
            img = np.asarray(Image.open(path).convert("RGB"))
        except Exception as e:
            r["error"] = f"读图失败 {e}"
            continue

        is_pano = True
        views, vmask, n = to_views(img, is_pano, args.views, fov, size, n_max)
        x = views[None].transpose(0, 1, 4, 2, 3)
        out = sess.run(["county"], {"views": x, "vmask": vmask[None]})[0][0]
        probs = fuse(out, adcodes=adcodes, from_logits=True)
        top = top_counties(probs, k=args.topk)
        lon, lat, used = predict_location(probs, cp, top_k=args.select_top_k)

        truth = r["truth_adcode"]
        preds = [a for a, _ in top]
        km = haversine(lon, lat, r["truth_lng"], r["truth_lat"]) / 1000
        landed = truth_at(locator_global, lon, lat)
        row = {
            **{k: r.get(k) for k in ("panoid", "date", "type", "obsolete",
                                     "online_adcode", "width")},
            "truth": truth, "truth_in_class": truth in cls,
            "truth_lng": r["truth_lng"], "truth_lat": r["truth_lat"],
            "truth_crs": "gcj02",
            "top": [{"adcode": a, "prob": float(p)} for a, p in top],
            "lon": float(lon), "lat": float(lat), "km": float(km), "landed": landed,
            "hit1": bool(truth == preds[0]),
            "hit3": bool(truth in preds[:3]),
            "hit5": bool(truth in preds[:5]),
            "landed_hit": bool(truth is not None and landed == truth),
            "prov_top1_hit": bool(truth and province_code(preds[0]) == province_code(truth)),
            "prov_hit": bool(truth is not None and any(
                province_code(a) == province_code(truth) for a in preds)),
            "near50": bool(km <= NEAR_KM),
        }
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  推理 {i+1}/{len(recs)}", flush=True)
    return rows, adcodes


def summarize(rows, names):
    n = len(rows)
    ev = [r for r in rows if r["truth"]]
    m = len(ev)
    print("\n" + "=" * 66)
    print(f"可评估 {m} / 抓到 {n}")
    if not m:
        return
    acc = lambda k: sum(r[k] for r in ev) / m                       # noqa: E731
    from collections import Counter
    maj = Counter(r["truth"] for r in ev).most_common(1)[0]
    print(f"\n县级 top1 {acc('hit1')*100:5.1f}%   top3 {acc('hit3')*100:5.1f}%"
          f"   top5 {acc('hit5')*100:5.1f}%")
    print(f"多数类基线   {maj[1]/m*100:5.1f}%   （{maj[0]} {describe(maj[0], names)}）")
    print(f"省级 top5 命中 {acc('prov_hit')*100:5.1f}%")
    print(f"选点落在真值县 {acc('landed_hit')*100:5.1f}%   "
          f"{NEAR_KM:.0f}km 内 {acc('near50')*100:5.1f}%")

    km = sorted(r["km"] for r in ev)
    med = km[len(km) // 2]
    print(f"\n选点误差  中位 {med:6.1f} km   均值 {sum(km)/len(km):7.1f} km")
    print("  分位  " + "  ".join(f"p{q}={km[min(len(km)-1, int(len(km)*q/100))]:.0f}"
                                for q in (10, 25, 50, 75, 90)))
    for lo, hi in ((0, 10), (10, 50), (50, 200), (200, 1000), (1000, 1e9)):
        c = sum(1 for r in ev if lo <= r["km"] < hi)
        print(f"  <{hi:>6.0f}km 区间 [{lo:>4.0f},{hi:>6.0f})  {c:3d}  {c/m*100:5.1f}%"
              + ("  " + "#" * int(c / m * 50)))

    print(f"\n真值县在类内  {sum(r['truth_in_class'] for r in ev)}/{m}"
          f"   （不在类内的答对不可能）")
    inb = [r for r in ev if r["truth_in_class"]]
    if inb:
        b = len(inb)
        print(f"  限类内  top1 {sum(r['hit1'] for r in inb)/b*100:5.1f}%"
              f"  top5 {sum(r['hit5'] for r in inb)/b*100:5.1f}%"
              f"  中位误差 {sorted(r['km'] for r in inb)[b//2]:.0f} km")

    print("\n── 按拍摄年份 ──")
    yrs = sorted({(r["date"] or "?")[:4] for r in ev})
    for y in yrs:
        g = [r for r in ev if (r["date"] or "?")[:4] == y]
        if len(g) < 3:
            continue
        print(f"  {y}  n={len(g):3d}  top1 {sum(r['hit1'] for r in g)/len(g)*100:5.1f}%"
              f"  top5 {sum(r['hit5'] for r in g)/len(g)*100:5.1f}%"
              f"  中位 {sorted(r['km'] for r in g)[len(g)//2]:6.0f} km")

    print("\n── 按省（≥4 条）──")
    provs = Counter(province_code(r["truth"]) for r in ev)
    for p, c in provs.most_common():
        if c < 4:
            continue
        g = [r for r in ev if province_code(r["truth"]) == p]
        nm = names.get(p, p)
        print(f"  {nm:<8} n={c:3d}  top1 {sum(r['hit1'] for r in g)/len(g)*100:5.1f}%"
              f"  top5 {sum(r['hit5'] for r in g)/len(g)*100:5.1f}%"
              f"  中位 {sorted(r['km'] for r in g)[len(g)//2]:6.0f} km"
              f"  省命中 {sum(r['prov_hit'] for r in g)/len(g)*100:5.1f}%")

    print("\n── 失败明细（top5 未命中）──")
    for r in sorted((r for r in ev if not r["hit5"]), key=lambda r: -r["km"])[:25]:
        pred = r["top"][0]
        print(f"  {r['panoid']}  {r['km']:7.1f} km  "
              f"真值 {r['truth']} {describe(r['truth'], names)}")
        print(f"      预测 {pred['adcode']} {pred['prob']*100:4.1f}% "
              f"{describe(pred['adcode'], names)}"
              f"   省{'对' if r['prov_hit'] else '错'}")


locator_global = None

def main():
    global args, locator_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--points", type=Path,
                    default=AI_DIR / "data" / "pool" / "county_points.npz")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--select-top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    (args.work / "images").mkdir(parents=True, exist_ok=True)
    positions = json.loads(args.json.read_text(encoding="utf-8"))["customCoordinates"]
    if args.limit:
        positions = positions[:args.limit]
    print(f"位置 {len(positions)} 个   缓存 {args.work}")

    names = load_names()
    locator_global = CountyLocator()

    known = set()
    tp = args.work / "truth.jsonl"
    if tp.exists():
        known = {json.loads(l)["panoid"] for l in tp.open(encoding="utf-8")}
    successful = {}
    if tp.exists():
        successful = {r["panoid"]: r for r in map(json.loads, tp.open(encoding="utf-8"))}
    todos = [p for p in positions if not successful.get(p["panoId"], {}).get("ok")
             or not (args.work / "images" / (p["panoId"] + ".jpg")).exists()]
    print(f"已缓存 {len(known & {p['panoId'] for p in positions})}   待抓 {len(todos)}")
    if todos:
        fetch_all(todos, args.work / "images", tp)

    recs = load_truth(tp, positions)
    rows, adcodes = evaluate(recs, args)
    (args.work / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")

    bad = [r for r in recs if r.get("ok") and r.get("online_adcode")
           and r["online_adcode"] != r["truth_adcode"]]
    if bad:
        print(f"\n在线复核与 json 坐标不一致 {len(bad)} 条：")
        for r in bad[:10]:
            print(f"  {r['panoid']}  json={r['truth_adcode']}  在线={r['online_adcode']}")
    diff = [r for r in recs if r.get("ok") and r.get("online_lng")
            and haversine(r["online_lng"], r["online_lat"],
                          r["json_lng"], r["json_lat"]) > 50]
    print(f"在线坐标与 json 坐标相差 >50m：{len(diff)} 条")
    fail = [r for r in recs if not r.get("ok")]
    if fail:
        print(f"抓取失败 {len(fail)} 条：" +
              "".join(f"\n  {r['panoid']} {r.get('error','')[:70]}" for r in fail))

    summarize(rows, names)
    print(f"\n结果 {args.work/'results.jsonl'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""把全景退化成"截图"的几种典型形态，看输出往哪儿跑。

    python tools/probe_degrade.py --run round2 --work round2/eval150

动机：单张正视角截图的实测效果远差于全景，且用户报告"很多错误地选了四川"，
而全景评估里四川只占 1%。这里对同一批真值点做退化，分别统计县级命中率和
各省在 top-20 里的占比，找出偏移发生在哪一步。

四川在训练时按 z=3（2048）抓、其余按 z=2（1024），decode_jpeg 后虽同为
1024×512，但一条走 DCT 半尺寸 draft、一条是原生解码，纹理不同——截图恰好
也是重编码/缩放的产物，所以重点看四川占比是否随之抬升。
"""
import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR / "tools"))

import numpy as np                                              # noqa: E402
from PIL import Image                                           # noqa: E402

from inference_onnx import extract_views, quiet_stderr, to_views  # noqa: E402
from models.fusion import fuse, top_counties                    # noqa: E402
from utils.views import extract_perspective
from utils.divisions import load_names                          # noqa: E402


def variants(img, size):
    """全景 → 若干退化版本。每个返回 (标签, 图, 是否当全景处理)。"""
    h, w = img.shape[:2]
    out = [("原图 2:1 全景", img, True)]
    for angle in (0, 90, 180, 270):
        shot = extract_perspective(img, heading=angle, fov_y=90, width=size, height=size)
        out.append((f"单视图90度 朝向{angle}", shot, False))

    # Render a camera view before encoding; equirectangular crops have wrong geometry.
    crop = extract_perspective(img, heading=0, fov_y=60,
                               width=round(size * 16 / 9), height=size)
    out.append(("正视角 16:9 截图", crop, False))

    # 同上去重编码（截图必然经过一次 JPEG）
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, "JPEG", quality=75)
    out.append(("正视角 + JPEG q75", np.asarray(Image.open(buf).convert("RGB")), False))

    # 正方形单视图（老版交互程序的做法）
    buf = io.BytesIO()
    sq = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
    sq.save(buf, "JPEG", quality=75)
    out.append((f"正视角 缩到 {size}²", np.asarray(Image.open(buf).convert("RGB")), False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    adcodes = meta["adcodes"]
    v = meta.get("views", {})
    size, fov, n_max = v.get("size", 224), v.get("fov", 90.0), v.get("n_max", 4)
    names = load_names()

    with quiet_stderr():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.run / "model.onnx"),
                                    providers=["CPUExecutionProvider"])

    rows = [json.loads(l) for l in
            (args.work / "results.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["truth"]]
    if args.limit:
        rows = rows[:args.limit]

    stats = {}
    for i, r in enumerate(rows):
        p = args.work / "images" / f"{r['panoid']}.jpg"
        img = np.asarray(Image.open(p).convert("RGB"))
        for tag, im, is_pano in variants(img, size):
            vv, vm, n = to_views(im, is_pano, 8, fov, size, n_max)
            out = sess.run(["county"], {"views": vv[None].transpose(0, 1, 4, 2, 3),
                                        "vmask": vm[None]})[0][0]
            probs = fuse(out, adcodes=adcodes, from_logits=True)
            top20 = [a for a, _ in top_counties(probs, k=20)]
            s = stats.setdefault(tag, {"n": 0, "hit1": 0, "hit5": 0,
                                       "prov20": Counter(), "sc20": 0})
            s["n"] += 1
            s["hit1"] += r["truth"] == top20[0]
            s["hit5"] += r["truth"] in top20[:5]
            s["prov20"].update(a[:2] for a in top20)
            s["sc20"] += sum(1 for a in top20 if a[:2] == "51")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    print(f"\n{'形态':<22}{'n':>4}{'top1':>7}{'top5':>7}"
          f"{'四川占top20':>12}   top20 里最多的省")
    for tag, s in stats.items():
        n = s["n"]
        top3 = " ".join(f"{names.get(p+'0000', p)}{c}" for p, c in s["prov20"].most_common(3))
        print(f"{tag:<22}{n:>4}{s['hit1']/n*100:6.1f}%{s['hit5']/n*100:6.1f}%"
              f"{s['sc20']/(n*20)*100:11.1f}%   {top3}")


if __name__ == "__main__":
    main()

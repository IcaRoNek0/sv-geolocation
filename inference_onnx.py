#!/usr/bin/env python
"""本地 CPU 推理：用 onnxruntime 跑导出的 ONNX 模型，不需要 torch。

给训练在云端、本地没有可用 torch 的场景用（实测 Termux/ARM 上 apt 的
python3-torch 是 SIGSEGV，而 onnxruntime 正常）。

    python inference_onnx.py --run <含 model.onnx 与 classes.json 的目录> \
        --image shot.jpg [--topk 5]

与 inference.py 的分工：那边用 torch，这边用 onnxruntime。两者都只依赖
numpy + PIL 做前处理（视图切分），差别仅在推理后端。

视图切分与坐标选点直接复用仓库里的 numpy 实现，因此**归一化必须由 ONNX
图内部完成**——导出时模型的 forward 原样保留，本地不做任何额外预处理，
两边就不会出现静默的不一致。
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.fusion import fuse, predict_location, top_counties
from utils.geo_utils import CountyPoints, haversine
from utils.views import extract_views, surround_headings

PANORAMA_ASPECT = 1.6      # 与 inference.py 保持一致；宽高比 ≥ 此值视为全景


def load_image(path):
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    return img, (w / h) >= PANORAMA_ASPECT


def to_views(img, is_pano, n_views, fov, size, n_max):
    """图 → (views, vmask, 实际视图数)。

    与 inference.py 里的同名函数逐行一致。没有抽成共用模块，是因为
    inference.py 顶层就 import torch，而本脚本要能在没有 torch 的机器上
    独立运行——抽出去会把 torch 依赖带进来。改动其中一处时另一处要同步。
    """
    if is_pano:
        n = max(1, min(n_views, n_max))
        used = extract_views(img, surround_headings(n), fov_y=fov, size=size)
    else:
        used = np.asarray(
            Image.fromarray(img).resize((size, size), Image.BILINEAR))[None]
        n = 1
    views = np.zeros((n_max, size, size, 3), dtype=np.uint8)
    views[:n] = used
    vmask = np.zeros(n_max, dtype=bool)
    vmask[:n] = True
    return views, vmask, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None,
                    help="默认用 <run>/model.onnx")
    ap.add_argument("--points", type=Path, default=Path("data/pool/county_points.npz"))
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--select-top-k", type=int, default=20)
    ap.add_argument("--truth", default=None, metavar="ADCODE,LON,LAT",
                    help="给了就算误差，例如 511424,103.51,30.02")
    args = ap.parse_args()

    import onnxruntime as ort

    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    adcodes = meta["adcodes"]
    vcfg = meta.get("views", {})
    size, fov = vcfg.get("size", 224), vcfg.get("fov", 90.0)
    n_max = vcfg.get("n_max", 4)

    onnx_path = args.model or (args.run / "model.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"模型 {onnx_path.name}  {onnx_path.stat().st_size/1e6:.0f} MB")

    img, is_pano = load_image(args.image)
    views, vmask, n = to_views(img, is_pano, args.views, fov, size, n_max)
    print(f"输入 {args.image.name}  {img.shape[1]}x{img.shape[0]}  "
          f"{'全景' if is_pano else '截图'} → {n} 个视图")

    # 形状必须是 (B, V, 3, H, W)，与导出时一致
    x = views[None].transpose(0, 1, 4, 2, 3)
    out = sess.run(["county"], {"views": x, "vmask": vmask[None]})[0][0]

    probs = fuse(out, adcodes=adcodes)
    cp = CountyPoints(args.points)
    lon, lat, used = predict_location(probs, cp, top_k=args.select_top_k)

    print(f"\n县级候选（前 {args.topk}）：")
    for a, p in top_counties(probs, k=args.topk):
        print(f"  {a}  概率 {p*100:5.1f}%")
    print(f"\n选点   经度 {lon:.5f}  纬度 {lat:.5f}")

    # 传了真值就能算误差：--truth <adcode>,<lng>,<lat>
    if args.truth:
        adc, tlon, tlat = args.truth.split(",")
        d = haversine(lon, lat, float(tlon), float(tlat)) / 1000
        hit = "命中" if adc in [a for a, _ in top_counties(probs, k=args.topk)] else "未命中"
        print(f"真值   {adc}  {tlon},{tlat}")
        print(f"误差   {d:.1f} km    top{args.topk} {hit}")


if __name__ == "__main__":
    main()

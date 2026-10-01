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
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.inference_views import load_image, to_views
from models.fusion import fuse, predict_location, top_counties
from utils.divisions import CountyLocator, describe, load_names
from utils.geo_utils import CountyPoints, haversine
from utils.views import extract_views, surround_headings

@contextlib.contextmanager
def quiet_stderr():
    """临时把文件描述符 2 指向 /dev/null。

    onnxruntime 建 session 时会重复注册 onnx schema，往 stderr 打几百条
    "Schema error: ... already registered"（实测 629 条 / 129 KB）。结果完全
    正确，但会淹没输出，而且极易被当成真错误——用户就把它写进过 err.txt。

    这些消息出自 C++ 层，Python 的 contextlib.redirect_stderr 拦不住，
    只能换文件描述符。仅包住建 session 那一步，之后的报错照常可见。
    """
    try:
        saved = os.dup(2)
    except OSError:            # fd 2 不可用（比如已关闭）
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--mode", choices=("screenshot", "panorama", "auto"),
                    default="screenshot", help="输入形式；默认单张截图")
    ap.add_argument("--model", type=Path, default=None,
                    help="默认用 <run>/model.onnx")
    ap.add_argument("--points", type=Path, default=Path("data/pool/county_points.npz"))
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--select-top-k", type=int, default=20)
    ap.add_argument("--truth", default=None, metavar="ADCODE,LON,LAT",
                    help="给了就算误差，例如 511424,103.51,30.02")
    ap.add_argument("--json", type=Path, help="Save full county probabilities for optional fusion")
    args = ap.parse_args()

    import onnxruntime as ort

    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    if meta.get("task", "environment") != "environment":
        raise ValueError("Use inference_vehicle.py for vehicle models")
    adcodes = meta["adcodes"]
    vcfg = meta.get("views", {})
    size, fov = vcfg.get("size", 224), vcfg.get("fov", 90.0)
    n_max = vcfg.get("n_max", 4)

    onnx_path = args.model or (args.run / "model.onnx")
    with quiet_stderr():
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
    print(f"模型 {onnx_path.name}  {onnx_path.stat().st_size/1e6:.0f} MB")

    img, is_pano = load_image(args.image, args.mode)
    views, vmask, n = to_views(img, is_pano, args.views, fov, size, n_max)
    print(f"输入 {args.image.name}  {img.shape[1]}x{img.shape[0]}  "
          f"{'全景' if is_pano else '截图'} → {n} 个视图")

    # 形状必须是 (B, V, 3, H, W)，与导出时一致
    x = views[None].transpose(0, 1, 4, 2, 3)
    out = sess.run(["county"], {"views": x, "vmask": vmask[None]})[0][0]

    probs = fuse(out, adcodes=adcodes, from_logits=True)
    cp = CountyPoints(args.points)
    lon, lat, used = predict_location(probs, cp, top_k=args.select_top_k)

    # adcode 没有可读性，配上行政区划名。两张表都读（见 utils/divisions.py）
    from utils.divisions import optional_geography
    names, locator = optional_geography()

    top = top_counties(probs, k=args.topk)
    truth_code = args.truth.split(",")[0] if args.truth else None
    print(f"\n县级候选（前 {args.topk}）：")
    for a, p in top:
        mark = "  <- 真值" if a == truth_code else ""
        print(f"  {a}  {p*100:5.1f}%  {describe(a, names)}{mark}")

    landed = locator.at(lon, lat) if locator is not None else None
    print(f"\n选点   经度 {lon:.5f}  纬度 {lat:.5f}")
    print(f"       落在 {describe(landed, names) if landed else '未提供边界或点位在县界之外'}")

    if args.json:
        args.json.write_text(json.dumps({"county_probabilities": probs,
            "mode": "panorama" if is_pano else "screenshot",
            "county_topk": [{"adcode": a, "prob": p} for a, p in top]},
            ensure_ascii=False), encoding="utf-8")

    # 传了真值就能算误差：--truth <adcode>,<lng>,<lat>
    if args.truth:
        adc, tlon, tlat = args.truth.split(",")
        d = haversine(lon, lat, float(tlon), float(tlat)) / 1000
        hit = "命中" if adc in [a for a, _ in top] else "未命中"
        print(f"真值   {adc}  {describe(adc, names)}  {tlon},{tlat}")
        print(f"误差   {d:.1f} km    top{args.topk} {hit}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""单图推理：街景图 → 县级概率分布 + 期望得分最高的坐标。

输入可以是两种形态，自动判别：

    全景      宽高比约 2:1，按多个朝向切视图（默认 8 个）
    截图      其它情况，缩放到模型输入尺寸当作单视图

两种都走同一条路径，因为模型训练时视图数就是可变的（1–8），单视图是被
训练过的能力而非推理时才遇到的情形。

用法：
    python inference.py --run runs/base --image shot.jpg
    python inference.py --run runs/base --image pano.jpg --views 8 --topk 5
    python inference.py --run runs/base --image shot.jpg --json out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.prepare import ViewConfig
from models.env_model import EnvModel
from models.fusion import fuse, predict_location, top_counties
from utils.geo_utils import CountyPoints
from utils.views import extract_views, surround_headings

PANORAMA_ASPECT = 1.6      # 宽高比超过此值视为全景（2:1 是标准，留些余量）


def load_run(run_dir):
    meta = json.loads((run_dir / "classes.json").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{run_dir} 下没有 checkpoint")
    return meta, ckpt_path


def load_image(path, size):
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    is_pano = (w / h) >= PANORAMA_ASPECT
    return img, is_pano


def to_views(img, is_pano, n_views, fov, size):
    """图 → (views, vmask)。全景切多视图，截图缩放为单视图。"""
    if is_pano:
        n = max(1, min(n_views, 8))
        used = extract_views(img, surround_headings(n), fov_y=fov, size=size)
    else:
        used = np.asarray(
            Image.fromarray(img).resize((size, size), Image.BILINEAR))[None]
        n = 1
    views = np.zeros((8, size, size, 3), dtype=np.uint8)
    views[:n] = used
    vmask = np.zeros(8, dtype=bool)
    vmask[:n] = True
    return views, vmask, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="含 classes.json 与 checkpoint 的目录")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--points", type=Path, default=Path("data/pool/county_points.npz"))
    ap.add_argument("--views", type=int, default=8, help="全景切几个视图")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--select-top-k", type=int, default=20,
                    help="选点时参与加权中位数的县数")
    ap.add_argument("--json", type=Path, help="把结果写到此文件")
    args = ap.parse_args()

    meta, ckpt_path = load_run(args.run)
    adcodes = meta["adcodes"]
    vcfg = ViewConfig(**meta.get("views", {}))
    size, fov = vcfg.size, vcfg.fov

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnvModel(len(adcodes), len(meta["cities"]), len(meta["provinces"]),
                     backbone=meta["backbone"], pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    img, is_pano = load_image(args.image, size)
    views, vmask, n_used = to_views(img, is_pano, args.views, fov, size)
    print(f"输入 {args.image.name}  {img.shape[1]}×{img.shape[0]}  "
          f"{'全景' if is_pano else '截图'} → {n_used} 个视图")

    with torch.no_grad(), torch.autocast(device_type=device.type,
                                         enabled=device.type == "cuda"):
        out = model(
            torch.from_numpy(views).permute(0, 3, 1, 2)[None].to(device),
            torch.from_numpy(vmask)[None].to(device),
        )
    logits = out["county"][0].float().cpu().numpy()

    # 单线索（本期无文字线索）；融合框架在 M2 接入
    probs = fuse(logits, adcodes=adcodes, temperature=1.0)

    cp = CountyPoints(args.points)
    lon, lat, used = predict_location(probs, cp, top_k=args.select_top_k)

    top = top_counties(probs, k=args.topk)
    print(f"\n县级候选（前 {args.topk}）：")
    for a, p in top:
        n_pts = len(cp.points(a)) if a in cp else 0
        print(f"  {a}  概率 {p*100:5.1f}%   点位数 {n_pts}")

    print(f"\n选点（{len(used)} 个候选县的真实点位加权中位数）：")
    print(f"  经度 {lon:.5f}   纬度 {lat:.5f}")

    # 注意力分布能看出模型更信哪个朝向的视图
    attn = out["attn"][0].float().cpu().numpy()[:n_used]
    if n_used > 1:
        print(f"\n视图注意力：{np.round(attn, 3).tolist()}")

    if args.json:
        args.json.write_text(json.dumps({
            "image": str(args.image),
            "mode": "panorama" if is_pano else "screenshot",
            "views_used": n_used,
            "county_topk": [{"adcode": a, "prob": p} for a, p in top],
            "location": {"lng": lon, "lat": lat},
            "candidates_used": used,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()

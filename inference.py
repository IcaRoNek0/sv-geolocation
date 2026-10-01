#!/usr/bin/env python
"""单图推理：街景图 → 县级概率分布 + 县名 + 坐标。

默认按单张截图处理，全景输入使用 --mode panorama。
两种情况走同一条路径，因为模型训练时视图数就是可变的。

    python inference.py --run <checkpoint 目录> --image shot.jpg [--topk 5] [--json out.json]
"""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.prepare import ViewConfig
from models.env_model import EnvModel
from data.inference_views import load_image, to_views
from models.fusion import fuse, predict_location, top_counties
from utils.divisions import CountyLocator, describe, load_names
from utils.geo_utils import CountyPoints
from utils.views import extract_views, surround_headings




def pick_amp_dtype(device):
    """Match training precision: Turing uses fp16, newer GPUs use bf16."""
    if device.type != "cuda":
        return None
    cc = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if cc[0] >= 8 else torch.float16


def load_run(run_dir):
    meta = json.loads((run_dir / "classes.json").read_text(encoding="utf-8"))
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{run_dir} 下没有 checkpoint")
    return meta, ckpt_path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="含 classes.json 与 checkpoint 的目录")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--mode", choices=("screenshot", "panorama", "auto"),
                    default="screenshot", help="输入形式；默认单张截图")
    ap.add_argument("--points", type=Path, default=Path("data/pool/county_points.npz"))
    ap.add_argument("--views", type=int, default=8, help="全景切几个视图")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--select-top-k", type=int, default=20,
                    help="选点时参与加权中位数的县数")
    ap.add_argument("--json", type=Path, help="把结果写到此文件")
    args = ap.parse_args()

    meta, ckpt_path = load_run(args.run)
    if meta.get("task", "environment") != "environment":
        raise ValueError("Use inference_vehicle.py for vehicle models")
    adcodes = meta["adcodes"]
    vcfg = ViewConfig(**meta.get("views", {}))
    size, fov = vcfg.size, vcfg.fov
    n_max = vcfg.n_max

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnvModel(len(adcodes), len(meta["cities"]), len(meta["provinces"]),
                     backbone=meta["backbone"], pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if state.get("run_meta") and state["run_meta"] != meta:
        raise ValueError("Checkpoint metadata mismatch")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    img, is_pano = load_image(args.image, args.mode)
    views, vmask, n_used = to_views(img, is_pano, args.views, fov, size, n_max)
    print(f"输入 {args.image.name}  {img.shape[1]}×{img.shape[0]}  "
          f"{'全景' if is_pano else '截图'} → {n_used} 个视图")

    amp_dtype = pick_amp_dtype(device)
    amp_ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype is not None else contextlib.nullcontext())
    with torch.no_grad(), amp_ctx:
        out = model(
            torch.from_numpy(views).permute(0, 3, 1, 2)[None].to(device),
            torch.from_numpy(vmask)[None].to(device),
        )
    logits = out["county"][0].float().cpu().numpy()

    # 单线索（文字线索 M2 接入）
    probs = fuse(logits, adcodes=adcodes, temperature=1.0, from_logits=True)

    cp = CountyPoints(args.points)
    lon, lat, used = predict_location(probs, cp, top_k=args.select_top_k)

    # adcode 没有可读性，配上行政区划名与边界判定。主库坐标与 geojson 同为
    # GCJ02，两边直接用；换 CRS 会整体错位到隔壁县。
    from utils.divisions import optional_geography
    names, locator = optional_geography()

    top = top_counties(probs, k=args.topk)
    print(f"\n县级候选（前 {args.topk}）：")
    for a, p in top:
        n_pts = len(cp.points(a)) if a in cp else 0
        print(f"  {a}  概率 {p*100:5.1f}%  {describe(a, names)}  点位数 {n_pts}")

    print(f"\n选点（{len(used)} 个候选县的真实点位加权中位数）：")
    print(f"  经度 {lon:.5f}   纬度 {lat:.5f}")
    landed = locator.at(lon, lat) if locator is not None else None
    print(f"  落在 {describe(landed, names) if landed else '未提供边界或点位在县界之外'}")

    attn = out["attn"][0].float().cpu().numpy()[:n_used]
    if n_used > 1:
        print(f"\n视图注意力：{np.round(attn, 3).tolist()}")

    if args.json:
        args.json.write_text(json.dumps({
            "image": str(args.image),
            "county_probabilities": probs,
            "mode": "panorama" if is_pano else "screenshot",
            "views_used": n_used,
            "county_topk": [{"adcode": a, "prob": p, "name": describe(a, names)}
                            for a, p in top],
            "location": {"lng": lon, "lat": lat, "adcode": landed,
                         "name": describe(landed, names) if landed else None},
            "candidates_used": used,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()

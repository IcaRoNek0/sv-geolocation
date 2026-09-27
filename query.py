#!/usr/bin/env python
"""交互式测试：输入 panoID → 取图 → 推理 → 与真值对比，q 退出。

    /usr/bin/python3.14 query.py --run round1 --model tools/model.onnx
    /usr/bin/python3.14 query.py --run round1            # 模型在 <run>/model.onnx

取图优先用本地分片（快，且能直接给出真值与所属划分）；分片里没有的 panoID
就去百度抓，真值由 sdata 返回的 BD09MC 坐标转 GCJ02 后做点面判定得到。
两种情况都展示县名——adcode 没有可读性。
"""
import argparse
import io
import json
import os
import sys
import contextlib
from pathlib import Path

import numpy as np
from PIL import Image

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR / "tools"))

from data.shards import open_index                      # noqa: E402
from inference_onnx import quiet_stderr, to_views       # noqa: E402
from models.fusion import fuse, predict_location, top_counties   # noqa: E402
from utils.divisions import (CountyLocator, describe, load_names)  # noqa: E402
from utils.geo_utils import CountyPoints, haversine     # noqa: E402


class Tester:
    def __init__(self, run, model_path, data_dir, points_path, topk, select_k):
        import onnxruntime as ort

        self.run = Path(run)
        self.meta = json.loads((self.run / "classes.json").read_text(encoding="utf-8"))
        vcfg = self.meta.get("views", {})
        self.size = vcfg.get("size", 224)
        self.fov = vcfg.get("fov", 90.0)
        self.n_max = vcfg.get("n_max", 4)
        self.adcodes = self.meta["adcodes"]
        self.topk, self.select_k = topk, select_k

        onnx_path = Path(model_path) if model_path else self.run / "model.onnx"
        with quiet_stderr():
            self.sess = ort.InferenceSession(str(onnx_path),
                                             providers=["CPUExecutionProvider"])

        self.cp = CountyPoints(points_path)
        self.names = load_names()
        print(f"边界表 {len(self.names)} 个区划名，加载点位表与边界中…", flush=True)
        self.locator = CountyLocator()
        print(f"就绪：{onnx_path.name}  {len(self.adcodes)} 类  "
              f"边界 {len(self.locator)} 个县\n", flush=True)

        # 本地分片与采样池（用于快速取图和真值）
        self.data_dir = Path(data_dir)
        self._idx = None
        self.samples = {}
        p = self.data_dir / "samples.jsonl"
        if p.exists():
            self.samples = {json.loads(l)["panoid"]: json.loads(l)
                            for l in p.open(encoding="utf-8")}
        sp = self.data_dir / "split.json"
        self.split = json.loads(sp.read_text(encoding="utf-8"))["assignments"] \
            if sp.exists() else {}

    def index(self):
        if self._idx is None:
            self._idx = open_index(self.data_dir)
        return self._idx

    # ── 取图 ────────────────────────────────────────────────────────
    def get_image(self, pid):
        """返回 (图像, 真值 adcode, 真值经纬度, 来源说明)。本地没有就去百度抓。"""
        m = self.samples.get(pid)
        if m:
            img = np.asarray(Image.open(io.BytesIO(self.index().read(pid))).convert("RGB"))
            which = self.split.get(pid, "?")
            return img, m["adcode"], (m.get("lng"), m.get("lat")), f"本地分片 · 划分={which}"

        import fetch_one
        sys.path.insert(0, str(AI_DIR.parent))       # sv_coordinates 在工作区根
        from sv_coordinates import convert
        # 取 2048 宽，与训练时的 z=3 一致。不能拿 self.size 去推——那是
        # 模型输入边长(224)，与源图分辨率无关；按它算会抓到 1024×512，
        # 等于换了个输入分布。
        pil, obj = fetch_one.fetch_pano(pid, 2048)
        img = np.asarray(pil.convert("RGB"))
        # sdata 给的是 BD09MC，先换成 GCJ02 再做点面判定
        lng, lat = convert(float(obj["X"]) / 100.0, float(obj["Y"]) / 100.0,
                           "bd09mc", "gcj02")
        return img, self.locator.at(lng, lat), (lng, lat), \
            f"百度现抓 · 拍摄 {obj.get('Date')}"

    # ── 推理并展示 ──────────────────────────────────────────────────
    def probe(self, pid):
        try:
            img, truth_code, truth_ll, src = self.get_image(pid)
        except SystemExit as e:
            print(f"  取图失败：{e}\n")
            return
        except Exception as e:
            print(f"  取图失败：{type(e).__name__}: {e}\n")
            return

        is_pano = img.shape[1] / img.shape[0] >= 1.6
        views, vmask, n = to_views(img, is_pano, 8, self.fov, self.size, self.n_max)
        x = views[None].transpose(0, 1, 4, 2, 3)
        out = self.sess.run(["county"], {"views": x, "vmask": vmask[None]})[0][0]
        probs = fuse(out, adcodes=self.adcodes)
        top = top_counties(probs, k=self.topk)
        lon, lat, used = predict_location(probs, self.cp, top_k=self.select_k)

        print(f"{pid}")
        print(f"  {src}   图 {img.shape[1]}x{img.shape[0]} "
              f"{'全景' if is_pano else '截图'} → {n} 视图")

        print(f"  候选（前 {self.topk}）：")
        for code, p in top:
            mark = "  ← 真值" if code == truth_code else ""
            print(f"    {code}  {p*100:5.2f}%  {describe(code, self.names)}{mark}")

        landed = self.locator.at(lon, lat)
        print(f"  选点 {lon:.5f}, {lat:.5f}  →  "
              f"{describe(landed, self.names) if landed else '边界外'}")

        if truth_code:
            d = haversine(lon, lat, truth_ll[0], truth_ll[1]) / 1000 \
                if truth_ll and None not in truth_ll else None
            hit5 = truth_code in [c for c, _ in top]
            tname = describe(truth_code, self.names)
            line = f"  真值 {truth_code}  {tname}"
            if d is not None:
                line += f"    误差 {d:.1f} km"
            line += f"    top{self.topk} {'命中' if hit5 else '未命中'}"
            print(line)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=AI_DIR / "data" / "shards")
    ap.add_argument("--points", type=Path,
                    default=AI_DIR / "data" / "pool" / "county_points.npz")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--select-top-k", type=int, default=20)
    ap.add_argument("panoids", nargs="*",
                    help="给了就直接跑这些，不给则进入交互输入")
    args = ap.parse_args()

    t = Tester(args.run, args.model, args.data, args.points, args.topk,
               args.select_top_k)

    if args.panoids:
        for pid in args.panoids:
            t.probe(pid.strip())
        return

    print("输入 panoID 回车查询，q 退出。")
    while True:
        try:
            line = input("panoID> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break
        t.probe(line)
    print("已退出。")


if __name__ == "__main__":
    main()

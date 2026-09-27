#!/usr/bin/env python
"""按 panoID 抓单张全景，并算出它的真实归属县。用于拿测试图。

    python tools/fetch_one.py 09019800121812261324400955A 09020600011607211758027359D \
        --out round1/test

抓下来的图不在训练集里（这些 ID 通常来自数据采集时未选中的点），正好用来
测泛化。归属县由 sdata 返回的 BD09MC 坐标转 GCJ02 后做点面判定得到——
不依赖主库，也就不需要再扫那三千万行。

层级取 ImgLayer 里最接近目标尺寸的一级；缺该级时回退，与 fetch_pano.py
的策略一致（ImgLayer 列出的层级不保证 pdata 真的提供）。
"""
import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SDATA = "https://mapsv0.bdimg.com/sv?qt=sdata&sid="
PDATA = "https://mapsv0.bdimg.com/?qt=pdata&sid={sid}&pos={r}_{c}&z={z}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.baidu.com/",
}
GRID = {1: (2, 1), 2: (4, 2), 3: (8, 4), 4: (16, 8)}   # ImgLevel -> 分块数


def get(url, binary=False):
    req = urllib.request.Request(url, headers=HEADERS)
    body = urllib.request.urlopen(req, timeout=30).read()
    return body if binary else json.loads(body)


def pick_level(layers, want_px=2048):
    """选边长最接近 want_px 的一级。"""
    best, best_d = None, None
    for l in layers or []:
        bx = int(l["BlockX"])
        px = bx * 512
        d = abs(px - want_px)
        if best_d is None or d < best_d:
            best, best_d = l, d
    return best


def fetch_pano(pid, want_px=2048):
    data = get(SDATA + pid)
    content = data.get("content")
    if not isinstance(content, list) or not content:
        raise SystemExit(f"{pid}: sdata 无内容（已失效？）")
    obj = next((o for o in content if o.get("ID") == pid), content[0])
    layer = pick_level(obj.get("ImgLayer"), want_px)
    if layer is None:
        raise SystemExit(f"{pid}: 没有 ImgLayer")
    level = int(layer["ImgLevel"])
    z, (bx, by) = level + 1, GRID.get(level, (int(layer["BlockX"]), int(layer["BlockY"])))

    tiles = {}
    for r in range(by):
        for c in range(bx):
            for attempt in range(3):
                try:
                    b = get(PDATA.format(sid=pid, r=r, c=c, z=z), binary=True)
                    if len(b) > 500:
                        tiles[(r, c)] = b
                        break
                except Exception:
                    time.sleep(0.5 * (attempt + 1))
    if len(tiles) != bx * by:
        raise SystemExit(f"{pid}: 瓦片 {len(tiles)}/{bx * by}")

    from PIL import Image
    first = Image.open(io.BytesIO(next(iter(tiles.values()))))
    tw, th = first.size
    canvas = Image.new("RGB", (bx * tw, by * th))
    for (r, c), blob in tiles.items():
        canvas.paste(Image.open(io.BytesIO(blob)).convert("RGB"), (c * tw, r * th))
    return canvas, obj


def adcode_of(x_bd09mc, y_bd09mc, index):
    """BD09MC → GCJ02 → 点面判定 → 县级 adcode。"""
    from shapely.geometry import Point, shape
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from sv_coordinates import convert
    lng, lat = convert(x_bd09mc / 100.0, y_bd09mc / 100.0, "bd09mc", "gcj02")
    pt = Point(lng, lat)
    for code, geom in index:
        if geom.contains(pt):
            return code, lng, lat
    return None, lng, lat


def load_boundaries():
    """(adcode, 几何) 列表，只为几个点做判定，不必建索引。"""
    from shapely.geometry import shape
    root = Path(__file__).resolve().parent.parent.parent / "geojson" / "counties"
    out = []
    for f in sorted(root.glob("*_full.json")):
        if f.name == "100000_full.json":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for feat in d.get("features", []):
            props = feat.get("properties", {})
            if props.get("level") != "district":
                continue
            try:
                out.append((str(props.get("adcode")), shape(feat["geometry"])))
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panoids", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bnd = load_boundaries()
    print(f"边界 {len(bnd)} 个县级多边形\n")

    for pid in args.panoids:
        img, obj = fetch_pano(pid, args.size)
        path = args.out / f"{pid}.jpg"
        img.save(path, "JPEG", quality=88, optimize=True)
        code, lng, lat = adcode_of(float(obj["X"]), float(obj["Y"]), bnd)
        print(f"{pid}")
        print(f"  图 {img.width}x{img.height}  {path.stat().st_size/1024:.0f} KB  →  {path}")
        print(f"  拍摄 {obj.get('Date')}  类型 {obj.get('Type')}  Obsolete {obj.get('Obsolete')}")
        print(f"  真值 adcode {code}  ({lng:.5f}, {lat:.5f})")
        print(f"  提示  python inference_onnx.py --run <目录> --image {path.name}"
              + (f" --truth {code},{lng:.6f},{lat:.6f}" if code else ""))
        print()


if __name__ == "__main__":
    main()

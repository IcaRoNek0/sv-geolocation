#!/usr/bin/env python
"""把抓好的图打成 WebDataset 分片，并导出全量元数据表。

分片里放的是**等距柱状原图**，不是切好的视图：一张全景在训练时可以按任意
朝向切出任意数量的视图，增强自由度最大；而 8 张 224² 视图的像素总量是全景的
两倍多，存视图既更大又更死。视图在训练时现切（成本是几毫秒的双线性采样）。

分片用顺序读的大 tar，而不是散图：Colab 挂载的 Drive 是 FUSE，读海量小文件
会灾难性变慢。

产物：
    data/shards/sv-XXXX.tar      每片约 500 MB，内含 {panoid}.jpg + {panoid}.json
    data/shards/samples.jsonl    全量元数据，统计与划分只读它，不碰图像
    data/shards/manifest.json    分片清单与实际体积

用法：
    python pack_shards.py
    python pack_shards.py --shard-mb 300
"""
import argparse
import json
import sqlite3
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AI_ROOT  # noqa: E402

STATE_DB = AI_ROOT / "data" / "fetch_state.sqlite3"
IMAGE_DIR = AI_ROOT / "data" / "images"
SHARD_DIR = AI_ROOT / "data" / "shards"

META_KEYS = ("adcode", "lng", "lat", "part", "vehicle", "date", "city",
             "obsolete", "width", "height")


def load_done():
    conn = sqlite3.connect(STATE_DB, timeout=60.0)
    rows = conn.execute(
        "SELECT panoid, adcode, part, bytes, width, height, date, obsolete "
        "FROM panos WHERE state='done' ORDER BY panoid"
    ).fetchall()
    conn.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-mb", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="只打包前 N 条，用于验证")
    ap.add_argument("--out", type=Path, default=SHARD_DIR)
    args = ap.parse_args()

    rows = load_done()
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("状态库里没有已完成的条目")
        return

    # 采样池提供 vehicle/date/city —— 状态库里没存这些
    pool = {}
    with (AI_ROOT / "data" / "pool" / "sample_pool.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            pool[r["panoid"]] = r

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.shard_mb * 1024 * 1024

    t0 = time.time()
    shard_idx = 0
    written = 0
    missing = []
    manifest = []
    cur = None
    cur_size = 0
    cur_count = 0
    samples_fh = (args.out / "samples.jsonl").open("w", encoding="utf-8")

    def close_shard():
        nonlocal cur, cur_size, cur_count, shard_idx
        if cur is None:
            return
        cur.close()
        manifest.append({
            "file": f"sv-{shard_idx:04d}.tar",
            "samples": cur_count,
            "bytes": cur_size,
        })
        shard_idx += 1
        cur, cur_size, cur_count = None, 0, 0

    try:
        for panoid, adcode, part, nbytes, width, height, date, obsolete in rows:
            img = IMAGE_DIR / part / adcode / f"{panoid}.jpg"
            if not img.exists():
                missing.append(panoid)
                continue

            if cur is None:
                path = args.out / f"sv-{shard_idx:04d}.tar"
                cur = tarfile.open(path, "w", format=tarfile.GNU_FORMAT)

            cur.add(str(img), arcname=f"{panoid}.jpg")
            meta = {k: v for k, v in zip(
                ("adcode", "part", "width", "height", "date", "obsolete"),
                (adcode, part, width, height, date, obsolete))}
            src = pool.get(panoid, {})
            meta.update({k: src.get(k) for k in ("lng", "lat", "vehicle", "city")})
            blob = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(f"{panoid}.json")
            info.size = len(blob)
            info.mtime = int(time.time())
            import io
            cur.addfile(info, io.BytesIO(blob))

            samples_fh.write(json.dumps({"panoid": panoid, **meta},
                                        ensure_ascii=False) + "\n")
            size = img.stat().st_size + len(blob)
            cur_size += size
            cur_count += 1
            written += 1

            if cur_size >= target:
                close_shard()
    finally:
        close_shard()
        samples_fh.close()

    total = sum(m["bytes"] for m in manifest)
    (args.out / "manifest.json").write_text(json.dumps({
        "shards": manifest,
        "samples": written,
        "bytes": total,
        "shard_mb": args.shard_mb,
        "missing": missing[:50],
        "missing_count": len(missing),
        "elapsed_seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"打包 {written:,} 条 → {len(manifest)} 个分片，{total/1e9:.2f} GB")
    if missing:
        print(f"缺图 {len(missing)} 条（前几个：{missing[:3]}）")
    print(f"耗时 {time.time()-t0:.1f}s  产物 {args.out}")


if __name__ == "__main__":
    main()

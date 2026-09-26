#!/usr/bin/env python
"""抓取街景图：批量 sdata 定层级 → pdata 拼等距柱状全景 → 存 JPEG。

两个阶段，均可断点续传（状态存 sqlite）：

    阶段 meta    每 100 个 ID 一次 sdata，读 ImgLayer 决定层级与分块数
    阶段 images  按 pos={行}_{列} 取瓦片，拼接，存 JPEG q85 + optimize

**不能硬编码分块数**：不同年份的街景层级数不同（2014 年 4 级，后续可能不同）。
先取元数据再决定抓法，顺便把 Date/Obsolete 等字段留下来。

层级映射（pdata z = ImgLayer.ImgLevel + 1，已实测）：

    四川   目标 2048×1024 → ImgLevel 2 → z=3 → 4×2 块
    全国   目标 1024×512  → ImgLevel 1 → z=2 → 2×1 块

层级缺失时回退到可用的最高层级，实际尺寸记进状态库。

用法：
    python fetch_pano.py meta   --limit 300
    python fetch_pano.py images --limit 300 --concurrency 16
    python fetch_pano.py images --part sichuan
    python fetch_pano.py report
"""
import argparse
import asyncio
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import aiohttp
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AI_ROOT, OUT_DIR  # noqa: E402

SDATA_URL = "https://mapsv0.bdimg.com/sv"
PDATA_URL = "https://mapsv0.bdimg.com/"
STATE_DB = AI_ROOT / "data" / "fetch_state.sqlite3"
IMAGE_DIR = AI_ROOT / "data" / "images"

# 每个归属部分的目标层级（pdata z）与实际像素
TIER = {"sichuan": 3, "national": 2}
TILE_PX = 512
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.baidu.com/",
}
SDATA_BATCH = 100        # 项目约定；400 表示批次过大，见 bisect 逻辑
JPEG_QUALITY = 85

SCHEMA = """
CREATE TABLE IF NOT EXISTS panos(
    panoid   TEXT PRIMARY KEY,
    adcode   TEXT,
    part     TEXT,
    state    TEXT,          -- pending_meta | meta_ok | meta_fail |
                            -- pending_img | done | fail
    error    TEXT,
    tile_z   INTEGER,
    block_x  INTEGER,
    block_y  INTEGER,
    bytes    INTEGER,
    width    INTEGER,
    height   INTEGER,
    date     TEXT,
    obsolete INTEGER,
    updated  REAL
);
CREATE INDEX IF NOT EXISTS panos_state ON panos(state);
"""


def open_state():
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB, timeout=60.0)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def load_pool(conn, limit=0, part=None):
    """把采样池灌进状态库，已存在的跳过。"""
    rows = []
    with (OUT_DIR / "sample_pool.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if part and r["part"] != part:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    now = time.time()
    conn.executemany(
        "INSERT OR IGNORE INTO panos(panoid, adcode, part, state, updated) "
        "VALUES(?,?,?,'pending_meta',?)",
        [(r["panoid"], r["adcode"], r["part"], now) for r in rows],
    )
    conn.commit()
    return len(rows)


# ── 阶段一：元数据 ──────────────────────────────────────────────────────

def pick_layer(layers, want_level):
    """在 ImgLayer 中选目标层级；缺失则回退到可用的最高层级。"""
    by_level = {int(l["ImgLevel"]): l for l in layers if "ImgLevel" in l}
    if not by_level:
        return None, False
    if want_level in by_level:
        return by_level[want_level], True
    top = max(by_level)
    return by_level[top], False


async def sdata_batch(session, ids, depth=0):
    """取一批 sdata；批次过大时二分重试（项目既有纪律）。"""
    url = f"{SDATA_URL}?qt=sdata&sid=" + ";".join(ids)
    try:
        async with session.get(url, headers=HEADERS, timeout=30) as r:
            body = await r.read()
        data = json.loads(body)
    except Exception as e:
        if depth < 4 and len(ids) > 1:
            mid = len(ids) // 2
            a = await sdata_batch(session, ids[:mid], depth + 1)
            b = await sdata_batch(session, ids[mid:], depth + 1)
            return {**a, **b}
        return {i: (None, f"http:{e}") for i in ids}

    err = (data.get("result") or {}).get("error")
    content = data.get("content")
    # 批次过大只表现为 error=400 或空 content——不能拿 len(content) < len(ids)
    # 当判据：sdata 对已失效的 ID 本就不返回，缺项是正常现象而非批次出错。
    broken = err not in (0, None) or not isinstance(content, list) \
        or (not content and len(ids) > 1)
    if broken:
        if depth < 4 and len(ids) > 1:
            mid = len(ids) // 2
            a = await sdata_batch(session, ids[:mid], depth + 1)
            b = await sdata_batch(session, ids[mid:], depth + 1)
            return {**a, **b}
        if not isinstance(content, list) or not content:
            return {i: (None, f"error={err}") for i in ids}

    out = {}
    seen = set()
    for obj in content if isinstance(content, list) else []:
        pid = obj.get("ID")
        if pid in ids:
            out[pid] = (obj, None)
            seen.add(pid)
    for i in ids:
        out.setdefault(i, (None, "absent"))
    return out


async def phase_meta(args):
    conn = open_state()
    n = load_pool(conn, args.limit, args.part)
    pending = [r[0] for r in conn.execute(
        "SELECT panoid FROM panos WHERE state='pending_meta'"
        + (" AND part=?" if args.part else ""),
        (args.part,) if args.part else (),
    )]
    print(f"池中 {n} 条（本次范围），待取元数据 {len(pending)} 条")
    if not pending:
        conn.close()
        return

    sem = asyncio.Semaphore(args.concurrency)
    done = fail = 0

    async def work(session, batch, pbar):
        nonlocal done, fail
        async with sem:
            res = await sdata_batch(session, batch)
        now = time.time()
        rows = []
        for pid, (obj, err) in res.items():
            if obj is None:
                rows.append((err or "no-data", pid))
                fail += 1
                continue
            layers = obj.get("ImgLayer") or []
            want = TIER.get(obj_part.get(pid, "national")) - 1
            layer, exact = pick_layer(layers, want)
            if layer is None:
                rows.append(("no-imglayer", pid))
                fail += 1
                continue
            rows.append((
                int(layer["ImgLevel"]) + 1, int(layer["BlockX"]), int(layer["BlockY"]),
                obj.get("Date"), int(obj.get("Obsolete") or 0), pid,
            ))
            done += 1
        conn.executemany(
            "UPDATE panos SET state='meta_ok', tile_z=?, block_x=?, block_y=?, "
            "date=?, obsolete=?, updated=? WHERE panoid=?",
            [(*r[:5], now, r[5]) for r in rows if len(r) == 6],
        )
        conn.executemany(
            "UPDATE panos SET state='meta_fail', error=?, updated=? WHERE panoid=?",
            [(r[0], now, r[1]) for r in rows if len(r) == 2],
        )
        conn.commit()
        pbar.update(len(batch))

    # part 查表，供 work 内使用
    obj_part = dict(conn.execute("SELECT panoid, part FROM panos"))

    batches = [pending[i:i + SDATA_BATCH] for i in range(0, len(pending), SDATA_BATCH)]
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(pending), unit=" 条", desc="元数据") as pbar:
            await asyncio.gather(*(work(session, b, pbar) for b in batches))

    ok = conn.execute("SELECT COUNT(*) FROM panos WHERE state='meta_ok'").fetchone()[0]
    # 层级分布
    dist = dict(conn.execute(
        "SELECT tile_z, COUNT(*) FROM panos WHERE state='meta_ok' GROUP BY tile_z"))
    print(f"元数据完成 {ok} 条，失败 {fail} 条；层级分布 {dist}")
    conn.close()


# ── 阶段二：图像 ────────────────────────────────────────────────────────

async def fetch_tile(session, sem, panoid, z, row, col, tries=3):
    url = f"{PDATA_URL}?qt=pdata&sid={panoid}&pos={row}_{col}&z={z}"
    async with sem:
        for attempt in range(tries):
            try:
                async with session.get(url, headers=HEADERS, timeout=30) as r:
                    body = await r.read()
                if len(body) > 500:
                    return body
            except Exception:
                pass
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


# 标准层级几何：z 层对应 2^(z-1) × 2^(z-2) 块 512² 瓦片。
# 实测 z=2→2×1、z=3→4×2、z=4→8×4、z=5→16×8。
def grid_for_z(z):
    return (2 ** (z - 1), 2 ** (z - 2))


# 各归属部分的**目标**像素尺寸。层级回退后要缩放到这个尺寸，
# 否则同一批数据里混着两种分辨率，训练侧得额外处理。
TARGET_SIZE = {"sichuan": (2048, 1024), "national": (1024, 512)}


async def fetch_grid(session, sem, panoid, z):
    """取该层级下的完整瓦片阵列。任何一块缺失即视为该层级不可用。

    不能只信 sdata 的 ImgLayer：实测有点位的元数据列出了 4 个层级，
    但 pdata 只提供 z=1（预览）与 z=4，z=2/z=3 全返回 404。
    """
    bx, by = grid_for_z(z)
    tiles = await asyncio.gather(*(
        fetch_tile(session, sem, panoid, z, r, c)
        for r in range(by) for c in range(bx)
    ))
    if any(t is None for t in tiles) or not tiles:
        return None, sum(1 for t in tiles if t is None), len(tiles)
    return tiles, 0, len(tiles)


def stitch(tiles, bx, by):
    first = Image.open(io.BytesIO(tiles[0]))
    tw, th = first.size
    canvas = Image.new("RGB", (bx * tw, by * th))
    for idx, body in enumerate(tiles):
        r, c = divmod(idx, bx)
        canvas.paste(Image.open(io.BytesIO(body)).convert("RGB"), (c * tw, r * th))
    return canvas


async def phase_images(args):
    conn = open_state()
    todo = list(conn.execute(
        "SELECT panoid, adcode, part, tile_z, block_x, block_y FROM panos "
        "WHERE state='meta_ok'" + (" AND part=?" if args.part else "") +
        " ORDER BY panoid LIMIT ?",
        ((args.part,) if args.part else ()) + (args.limit or -1,),
    ))
    print(f"待抓图 {len(todo)} 条")
    if not todo:
        conn.close()
        return

    sem = asyncio.Semaphore(args.concurrency)
    done = fail = 0
    total_bytes = 0

    async def work(session, rec, pbar):
        nonlocal done, fail, total_bytes
        panoid, adcode, part, z, bx, by = rec
        target = TARGET_SIZE.get(part)
        # 目标层级优先，失败则按尺寸接近程度回退。最多试 3 个层级——
        # 对真正失效的点位，穷举 5 个层级 × 最多 16 块瓦片纯属浪费。
        fallbacks = sorted((c for c in (2, 3, 4, 5) if c != z),
                           key=lambda c: abs(c - z))[:2]
        canvas = None
        tried = []
        for cz in [z] + fallbacks:
            tiles, missing, total = await fetch_grid(session, sem, panoid, cz)
            if tiles is None:
                tried.append(f"z{cz}:{missing}/{total}")
                continue
            try:
                cbx, cby = grid_for_z(cz)
                canvas = stitch(tiles, cbx, cby)
            except Exception as e:
                tried.append(f"z{cz}:stitch:{e}")
                continue
            if cz != z:
                # 回退层级必须缩放回目标尺寸，否则同一批数据里混着两种分辨率
                if target and (canvas.width, canvas.height) != target:
                    canvas = canvas.resize(target, Image.LANCZOS)
            break

        if canvas is None:
            conn.execute(
                "UPDATE panos SET state='fail', error=?, updated=? WHERE panoid=?",
                (";".join(tried), time.time(), panoid),
            )
            conn.commit()
            fail += 1
            pbar.update(1)
            pbar.set_postfix(ok=done, fail=fail, mb=f"{total_bytes/1e6:.0f}")
            return

        path = IMAGE_DIR / part / adcode / f"{panoid}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        size = path.stat().st_size
        conn.execute(
            "UPDATE panos SET state='done', bytes=?, width=?, height=?, updated=? "
            "WHERE panoid=?",
            (size, canvas.width, canvas.height, time.time(), panoid),
        )
        conn.commit()
        done += 1
        total_bytes += size
        pbar.update(1)
        pbar.set_postfix(ok=done, fail=fail, mb=f"{total_bytes/1e6:.0f}")

    # aiohttp 默认连接池上限是 100，不显式放开的话并发设再大也无效——
    # 而且不会报错，只是悄悄卡在 100。
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(todo), unit=" 条", desc="抓图") as pbar:
            await asyncio.gather(*(work(session, r, pbar) for r in todo))

    print(f"完成 {done}，失败 {fail}，共 {total_bytes/1e9:.2f} GB")
    conn.close()


def report():
    conn = open_state()
    print("\n状态分布：")
    for state, n in conn.execute(
            "SELECT state, COUNT(*) FROM panos GROUP BY state ORDER BY 2 DESC"):
        print(f"  {state:<14} {n:>7,}")
    print("\n按归属部分的实际体积：")
    for part, n, b, w, h in conn.execute(
            "SELECT part, COUNT(*), SUM(bytes), AVG(width), AVG(height) FROM panos "
            "WHERE state='done' GROUP BY part"):
        if n:
            print(f"  {part:<10} {n:>6,} 条  平均 {b/n/1024:>6.1f} KB  "
                  f"合计 {b/1e9:.2f} GB  尺寸 {int(w)}×{int(h)}")
    size_dist = dict(conn.execute(
        "SELECT tile_z, COUNT(*) FROM panos WHERE state='done' GROUP BY tile_z"))
    print(f"\n实际层级分布：{size_dist}")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["meta", "images", "report"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--part", choices=["sichuan", "national"])
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    if args.phase == "meta":
        asyncio.run(phase_meta(args))
    elif args.phase == "images":
        asyncio.run(phase_images(args))
    else:
        report()


if __name__ == "__main__":
    main()

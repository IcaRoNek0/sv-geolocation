#!/usr/bin/env python
"""单次全表扫描主库，产出采样池、覆盖直方图与每县真实点位表。

这是整个方案中**唯一**全表扫描 qsresult/all_streetviews.sqlite3 的地方。
三千万行的扫描代价很高，因此三个产物必须一次产出：

    sample_pool.jsonl   训练用采样池（四川按县配额，其余按省配额）
    histogram.json      各省/县的可用量与选中量，用于定配额与识别无覆盖县
    county_points.npz   每县真实点位表（网格抽稀），选点用，覆盖全部候选县

点位表覆盖全部县而非仅采样县——整县留出的测试县也必须有真实点位，
否则划分 B 无法评估选点质量。

采样用**蓄水池**而非"取前 N 个"：主库的物理顺序高度地理聚集
（前 20 万行只覆盖 6 个省），直接取前 N 会把采样锁死在某个爬取批次上。
蓄水池保证每组是均匀随机的结果，之后再做网格抽稀去冗余。

用法：
    python sample_pool.py --limit 200000     # 小样验证
    python sample_pool.py                    # 完整扫描（实测约 2 分钟）
"""
import argparse
import json
import random
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    NATIONAL_PER_PROVINCE,
    OUT_DIR,
    POINTS_GRID_M,
    POINTS_PER_COUNTY,
    SAMPLE_GRID_M,
    SICHUAN_PER_COUNTY,
    SICHUAN_PREFIX,
    GridThinner,
    open_main_db,
    parse_panoid,
    province_of,
)

QUERY = """
SELECT panoid, lng, lat, adcode
  FROM panos
 WHERE kind = 'normal'
   AND adcode IS NOT NULL
   AND adcode_status IN ('local', 'api')
   AND lng IS NOT NULL
   AND lat IS NOT NULL
"""

SEED = 20260926
RESERVOIR_FACTOR = 2      # 蓄水池容量 = 配额 × 2，留出抽稀损耗


def quota_for(adcode):
    """返回 (分组键, 配额, 归属部分)。四川按县配额，其余按省配额。"""
    if adcode.startswith(SICHUAN_PREFIX):
        return adcode, SICHUAN_PER_COUNTY, "sichuan"
    prov = province_of(adcode)
    return prov, NATIONAL_PER_PROVINCE, "national"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只扫描前 N 行，用于验证")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = random.Random(SEED)

    county_total = defaultdict(int)
    province_total = defaultdict(int)
    reservoirs = defaultdict(list)          # 分组 -> 蓄水池
    seen_per_group = defaultdict(int)       # 分组 -> 已见候选数
    point_thinners = {}                     # adcode -> GridThinner
    points = defaultdict(lambda: array("d"))  # adcode -> 交错 lng,lat

    scanned = 0
    conn = open_main_db()
    try:
        cur = conn.execute(QUERY)
        pbar = tqdm(total=args.limit or None, unit=" 行", desc="扫描主库",
                    mininterval=5.0)
        for panoid, lng, lat, adcode in cur:
            scanned += 1
            if args.limit and scanned > args.limit:
                break
            pbar.update(1)

            prov = province_of(adcode)
            county_total[adcode] += 1
            province_total[prov] += 1

            # ── 点位表：覆盖全部县，与采样配额无关 ──────────────────
            bucket = points[adcode]
            if len(bucket) < POINTS_PER_COUNTY * 2:
                thinner = point_thinners.get(adcode)
                if thinner is None:
                    thinner = point_thinners[adcode] = GridThinner(POINTS_GRID_M)
                if thinner.accept(lng, lat):
                    bucket.append(lng)
                    bucket.append(lat)

            # ── 采样池：每组一个蓄水池 ──────────────────────────────
            group, quota, part = quota_for(adcode)
            meta = parse_panoid(panoid)
            if meta is None:
                continue
            row = {
                "panoid": panoid,
                "adcode": adcode,
                "lng": round(lng, 6),
                "lat": round(lat, 6),
                "part": part,
                **meta,
            }
            n = seen_per_group[group] + 1
            seen_per_group[group] = n
            res = reservoirs[group]
            cap = quota * RESERVOIR_FACTOR
            if len(res) < cap:
                res.append(row)
            else:
                j = rng.randrange(n)
                if j < cap:
                    res[j] = row

        pbar.close()
    finally:
        conn.close()

    # ── 蓄水池 → 网格抽稀 → 最终采样池 ──────────────────────────────
    sample_rows = []
    county_selected = defaultdict(int)
    province_selected = defaultdict(int)
    for group, rows in reservoirs.items():
        _, quota, _ = quota_for(group)      # 分组键本身就是县或省 adcode
        rows.sort(key=lambda r: r["panoid"])      # 固定顺序，保证可复现
        thinner = GridThinner(SAMPLE_GRID_M)
        kept = 0
        for r in rows:
            if thinner.accept(r["lng"], r["lat"]):
                sample_rows.append(r)
                county_selected[r["adcode"]] += 1
                province_selected[province_of(r["adcode"])] += 1
                kept += 1
                if kept >= quota:
                    break
    sample_rows.sort(key=lambda r: r["panoid"])

    # ── 落盘 ────────────────────────────────────────────────────────
    with (args.out / "sample_pool.jsonl").open("w", encoding="utf-8") as fh:
        for row in sample_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    codes = sorted(c for c in points if len(points[c]) > 0)
    flat = (
        np.concatenate([np.frombuffer(points[c], dtype="<f8") for c in codes])
        if codes
        else np.zeros(0, dtype="<f8")
    )
    offsets = np.zeros(len(codes) + 1, dtype="<i8")
    for i, c in enumerate(codes):
        offsets[i + 1] = offsets[i] + len(points[c]) // 2
    np.savez_compressed(
        args.out / "county_points.npz",
        adcodes=np.array(codes),
        coords=flat.astype("<f8"),
        offsets=offsets,
    )

    sic_counties = [c for c in sorted(county_total) if c.startswith(SICHUAN_PREFIX)]
    hist = {
        "scanned_rows": scanned,
        "limit": args.limit or None,
        "seed": SEED,
        "elapsed_seconds": round(time.time() - t0, 1),
        "grid": {"sample_m": SAMPLE_GRID_M, "points_m": POINTS_GRID_M},
        "quota": {
            "sichuan_per_county": SICHUAN_PER_COUNTY,
            "national_per_province": NATIONAL_PER_PROVINCE,
        },
        "totals": {
            "sample_rows": len(sample_rows),
            "sichuan_rows": sum(1 for r in sample_rows if r["part"] == "sichuan"),
            "national_rows": sum(1 for r in sample_rows if r["part"] == "national"),
            "counties_with_points": len(codes),
            "points_total": int(offsets[-1]) if len(codes) else 0,
            "provinces_with_data": len(province_total),
            "sichuan_counties": len(sic_counties),
            "sichuan_counties_covered": sum(
                1 for c in sic_counties if county_total[c] > 0
            ),
            "sichuan_counties_under_quota": sum(
                1 for c in sic_counties if county_total[c] < SICHUAN_PER_COUNTY
            ),
            "national_groups_under_quota": sum(
                1
                for p in province_total
                if not p.startswith(SICHUAN_PREFIX)
                and province_total[p] < NATIONAL_PER_PROVINCE
            ),
        },
        "provinces": {
            p: {
                "available": province_total[p],
                "selected": province_selected.get(p, 0),
                "quota": (
                    SICHUAN_PER_COUNTY * len(sic_counties)
                    if p.startswith(SICHUAN_PREFIX)
                    else NATIONAL_PER_PROVINCE
                ),
            }
            for p in sorted(province_total)
        },
        "counties": {
            c: {"available": county_total[c], "selected": county_selected.get(c, 0)}
            for c in sorted(county_total)
        },
    }
    (args.out / "histogram.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    t = hist["totals"]
    print(f"\n扫描行数      {scanned:,}")
    print(f"采样池        {t['sample_rows']:,} 条"
          f"（四川 {t['sichuan_rows']:,} + 全国 {t['national_rows']:,}）")
    print(f"点位表        {t['counties_with_points']:,} 县 / {t['points_total']:,} 点")
    print(f"有数据的省    {t['provinces_with_data']}")
    print(f"耗时          {hist['elapsed_seconds']}s")
    print(f"\n四川          {t['sichuan_counties_covered']}/{t['sichuan_counties']} 个县有数据，"
          f"{t['sichuan_counties_under_quota']} 个县可用量不足配额")
    print(f"全国          配额不足的省 {t['national_groups_under_quota']}")
    print(f"产物          {args.out}")


if __name__ == "__main__":
    main()

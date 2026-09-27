#!/usr/bin/env python
"""生成训练/验证/测试划分。

划分单位是 (车辆, 日期) 分组而非单张图：同车同日相邻帧几乎重复，随机按图
划分会让验证集出现训练集见过的地点，指标虚高却不报错。整个分组要么全进
训练、要么全进评估。

    test_county  整县留出（测没见过的县，对外敢报的数字）
    val_same     同县留出，各留 ~15 条（测见过的地方）
    val_national 全国铺底按分组留出 ~10%，只用于省级指标
    train        其余

split.json 固定下来随分片上传，不在每次训练时重算，否则指标失去可比性。

    python make_splits.py [--holdout-counties N --per-county N]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AI_ROOT  # noqa: E402

SAMPLES = AI_ROOT / "data" / "shards" / "samples.jsonl"
OUT = AI_ROOT / "data" / "shards" / "split.json"
SEED = 20260926


def group_key(s):
    """划分单位：(车辆, 日期)。缺字段时退回样本自身，绝不与别的合并。"""
    v, d = s.get("vehicle"), s.get("date")
    if v and d:
        return f"{v}|{d}"
    return f"solo|{s['panoid']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-counties", type=int, default=10,
                    help="整县留出的四川县数量")
    ap.add_argument("--per-county", type=int, default=15,
                    help="同县留出时每县留出的样本数")
    ap.add_argument("--national-val-ratio", type=float, default=0.10)
    ap.add_argument("--min-county-samples", type=int, default=40,
                    help="只有样本量达到此数的县才有资格被整县留出")
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    samples = [json.loads(l) for l in args.samples.open(encoding="utf-8")]
    rng = random.Random(SEED)

    by_county = defaultdict(list)
    for s in samples:
        by_county[s["adcode"]].append(s)

    # ── 选整县留出的县 ──────────────────────────────────────────────
    sic = [a for a in by_county if a.startswith("51")]
    eligible = sorted(a for a in sic if len(by_county[a]) >= args.min_county_samples)
    if len(eligible) < args.holdout_counties:
        raise SystemExit(
            f"样本量 ≥{args.min_county_samples} 的四川县只有 {len(eligible)} 个，"
            f"不足 {args.holdout_counties} 个")
    # 按 adcode 排序后等距抽取，使留出县在地理上分散而非集中在同一片
    step = len(eligible) / args.holdout_counties
    holdout = [eligible[int(i * step)] for i in range(args.holdout_counties)]

    # ── 分组 ────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for s in samples:
        groups[group_key(s)].append(s)

    split = {}
    # 阶段 1：整县留出。只要分组里有样本落在留出县，整组一起进测试集
    for gid, members in groups.items():
        if any(m["adcode"] in holdout for m in members):
            for m in members:
                split[m["panoid"]] = "test_county"

    # 阶段 2：同县留出，按分组累积到每县 --per-county 条
    held = defaultdict(int)
    group_order = sorted(groups)
    rng.shuffle(group_order)
    for gid in group_order:
        members = groups[gid]
        if any(m["panoid"] in split for m in members):
            continue
        if all(m["part"] == "national" for m in members):
            continue
        # 只按四川的县计数
        counties = {m["adcode"] for m in members if m["adcode"].startswith("51")}
        if not counties:
            continue
        if all(held[c] >= args.per_county for c in counties):
            continue
        for m in members:
            split[m["panoid"]] = "val_same"
        for c in counties:
            held[c] += sum(1 for m in members if m["adcode"] == c)

    # 阶段 3：全国铺底按分组留出
    assigned = {group_key(m) for m in samples if m["panoid"] in split}
    nat_groups = [g for g in sorted(groups)
                  if g not in assigned
                  and all(m["part"] == "national" for m in groups[g])]
    rng.shuffle(nat_groups)
    budget = int(len(nat_groups) * args.national_val_ratio)
    for gid in nat_groups[:budget]:
        for m in groups[gid]:
            split[m["panoid"]] = "val_national"

    # 其余全部进训练集
    for s in samples:
        split.setdefault(s["panoid"], "train")

    # ── 自检：任何分组都不得跨越训练与评估 ──────────────────────────
    leaked = []
    for gid, members in groups.items():
        kinds = {split[m["panoid"]] for m in members}
        if "train" in kinds and len(kinds) > 1:
            leaked.append((gid, kinds))
    if leaked:
        raise SystemExit(
            f"{len(leaked)} 个分组跨越了训练与评估，划分有泄漏：{leaked[:3]}")

    counts = defaultdict(int)
    for v in split.values():
        counts[v] += 1
    test_counties = sorted({s["adcode"] for s in samples
                            if split[s["panoid"]] == "test_county"})

    payload = {
        "seed": SEED,
        "params": {
            "holdout_counties": args.holdout_counties,
            "per_county": args.per_county,
            "national_val_ratio": args.national_val_ratio,
            "min_county_samples": args.min_county_samples,
        },
        "grouping": "(vehicle, date)",
        "holdout_adcodes": holdout,
        "test_counties_present": test_counties,
        "counts": dict(counts),
        "assignments": split,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"样本 {len(samples):,} 条，分组 {len(groups):,} 个")
    for k in ("train", "val_same", "val_national", "test_county"):
        print(f"  {k:<14} {counts.get(k, 0):>7,}")
    print(f"\n整县留出的县（{len(holdout)} 个）：{', '.join(holdout)}")
    print(f"分组泄漏自检：通过（0 个分组跨越训练与评估）")
    print(f"产物 {args.out}")


if __name__ == "__main__":
    main()

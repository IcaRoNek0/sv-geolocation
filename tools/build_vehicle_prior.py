#!/usr/bin/env python
"""Read existing SVTR/SVTB trajectory exports; no database scans or requests."""
import argparse
from collections import defaultdict
from datetime import date, timedelta
import gzip
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import file_hash


def read_exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise ValueError("Truncated trajectory")
    return data


def varint(stream):
    result = 0
    for shift in range(0, 70, 7):
        b = read_exact(stream, 1)[0]
        result |= (b & 127) << shift
        if not b & 128:
            return result
    raise ValueError("Invalid varint")


def string(stream):
    return read_exact(stream, varint(stream)).decode("utf-8")


def single(stream):
    version = varint(stream)
    if version not in (2, 3, 4):
        raise ValueError(f"Unsupported trajectory version {version}")
    kinds = 4 if version == 2 else 5
    vehicle = string(stream).upper()
    string(stream); string(stream)
    varint(stream)
    for _ in range(kinds):
        varint(stream)
    years = defaultdict(set)
    for _ in range(varint(stream)):
        varint(stream); varint(stream); varint(stream); string(stream)
        for _ in range(kinds):
            varint(stream)
        for _ in range(varint(stream)):
            day, code = varint(stream), varint(stream)
            varint(stream)
            for _ in range(kinds):
                varint(stream)
            if code:
                year = str((date(1970, 1, 1) + timedelta(days=day)).year)
                years[year].add(f"{code - 1:06d}")
        if version >= 4:
            for _ in range(varint(stream)):
                varint(stream); string(stream)
    return vehicle, {y: sorted(c) for y, c in years.items()}


def read_coverage(path):
    coverage = {}
    def consume(stream):
        magic = read_exact(stream, 4)
        if magic == b"SVTB":
            for _ in range(varint(stream)):
                blob = read_exact(stream, varint(stream))
                consume(io.BytesIO(gzip.decompress(blob)))
        elif magic == b"SVTR":
            vehicle, years = single(stream)
            dest = coverage.setdefault(vehicle, {})
            for year, codes in years.items():
                dest[year] = sorted(set(dest.get(year, [])) | set(codes))
        else:
            raise ValueError(f"Unknown trajectory magic: {magic!r}")
    with gzip.open(path, "rb") as f:
        consume(f)
    return coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    coverage = read_coverage(args.trajectory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"version": 1, "source": str(args.trajectory),
        "source_sha256": file_hash(args.trajectory), "coverage": coverage},
        ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(coverage)} vehicle trajectories to {args.out}")


if __name__ == "__main__":
    main()

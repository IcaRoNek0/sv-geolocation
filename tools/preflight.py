#!/usr/bin/env python
"""Check mounted Kaggle dataset integrity before spending GPU time."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.shards import open_index, load_split, decode_jpeg


def check(data):
    rows = [json.loads(line) for line in (data / 'samples.jsonl').open(encoding='utf-8')]
    samples = {r['panoid']: r for r in rows}
    if len(samples) != len(rows):
        raise ValueError('Duplicate panoIDs in samples.jsonl')
    split, _ = load_split(data / 'split.json')
    if set(split) != set(samples):
        raise ValueError('Split/sample keys differ; mount the matching dataset version')
    groups = {}
    for r in rows:
        key = (str(r.get('vehicle', '')).upper(), r.get('date'))
        if all(key):
            previous = groups.setdefault(key, split[r['panoid']])
            if previous != split[r['panoid']]:
                raise ValueError(f'Vehicle/date group crosses splits: {key}')
    idx = open_index(data)
    try:
        keys = set(idx.keys)
        missing = set(samples) - keys
        if missing or len(keys) != len(idx.keys):
            raise ValueError(f'Missing images={len(missing)} or duplicate image keys')
        for k in sorted(samples)[::max(1, len(samples)//12)]:
            im = decode_jpeg(idx.read(k))
            if im.ndim != 3 or im.shape[1] != im.shape[0] * 2:
                raise ValueError(f'Not a 2:1 panorama: {k}, {im.shape}')
    finally:
        idx.close()
    counts = Counter(split.values())
    train_classes = Counter(samples[k]['adcode'] for k, sp in split.items() if sp == 'train')
    report = {'samples':len(rows), 'splits':dict(counts), 'train_counties':len(train_classes),
              'untrained_counties':sorted({r['adcode'] for r in rows} - train_classes.keys()),
              'train_counties_under_10':sum(n < 10 for n in train_classes.values())}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    args = ap.parse_args()
    check(args.data)


if __name__ == '__main__':
    main()

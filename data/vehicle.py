"""Vehicle labels are training supervision only; inference never reads panoIDs."""
from collections import Counter, defaultdict

import numpy as np

from data.prepare import augment, sample_rng
from utils.views import extract_views


def vehicle_id(sample):
    vehicle = str(sample.get("vehicle", "")).upper()
    # Match the trajectory exporter: personal-upload suffixes are not official cars.
    return "" if vehicle.rsplit("_", 1)[-1] in {"IN", "OI", "UZ"} else vehicle


def year_id(sample):
    date = str(sample.get("date", ""))
    return date[:4] if len(date) == 8 and date.isdigit() else "unknown"


def build_vehicle_indices(samples, split, min_samples=20, min_dates=2):
    counts, dates = Counter(), defaultdict(set)
    for key, s in samples.items():
        if split.get(key) == "train" and vehicle_id(s):
            counts[vehicle_id(s)] += 1
            dates[vehicle_id(s)].add(s.get("date"))
    vehicles = sorted(v for v, n in counts.items() if n >= min_samples and len(dates[v]) >= min_dates)
    if not vehicles:
        raise ValueError("No vehicle classes meet training sample/date thresholds")
    years = sorted({year_id(s) for k, s in samples.items()
                    if split.get(k) == "train" and year_id(s) != "unknown"})
    prefixes = sorted({v.split("_")[0] for v in vehicles})
    vi, yi, pi = ({v: i for i, v in enumerate(a)} for a in (vehicles, years, prefixes))
    county = {k: vi.get(vehicle_id(s), -1) for k, s in samples.items()}
    city = {k: yi.get(year_id(s), -1) for k, s in samples.items()}
    prov = {k: pi.get(vehicle_id(s).split("_")[0], -1) for k, s in samples.items()}
    coords = {k: (0.0, 0.0) for k in samples}
    return vehicles, years, prefixes, county, city, prov, coords


def car_view(pano, cfg, rng, training=False):
    # Positive pitch points down in this project's coordinate convention.
    heading = float(rng.uniform(0, 360)) if training else 0.0
    fov = float(rng.choice(np.linspace(*cfg.fov_range, 5))) if training and cfg.fov_range else cfg.fov
    view = extract_views(pano, [heading], pitch=90.0, fov_y=fov, size=cfg.size)
    if training:
        view = augment(view, rng, cfg)
    return mask_vehicle_crop(view[0])[None]


def mask_vehicle_crop(img):
    # Suppress corner scenery, so road/building texture is less useful as a shortcut.
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = ((x + 0.5 - w / 2) / (w / 2)) ** 2 + ((y + 0.5 - h / 2) / (h / 2)) ** 2 <= 1
    return np.where(mask[..., None], img, 0).astype(np.uint8)

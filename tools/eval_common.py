"""Explicit evaluation truth and cache provenance; never infer truth from predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np


def read_positions(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["customCoordinates"]
    result = {}
    for r in rows:
        key = r["panoId"]
        coord = (float(r["lng"]), float(r["lat"]))
        if not np.isfinite(coord).all() or not (-180 <= coord[0] <= 180 and -90 <= coord[1] <= 90):
            raise ValueError(f"Invalid truth coordinate for {key}")
        if key in result and result[key] != coord:
            raise ValueError(f"Conflicting coordinates for {key}")
        result[key] = coord
    return result


def truth_coordinates(rows, positions=None):
    result = []
    for r in rows:
        if positions is not None:
            coord = positions[r["panoid"]]
        elif "truth_lng" in r and "truth_lat" in r:
            coord = (r["truth_lng"], r["truth_lat"])
        else:
            raise ValueError("Missing explicit truth coordinates; pass --positions. lon/lat are predictions.")
        result.append(coord)
    return np.asarray(result, dtype=np.float64)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_identity(run, rows, work):
    return {
        "version": 2,
        "model_sha256": file_hash(run / "model.onnx"),
        "classes_sha256": file_hash(run / "classes.json"),
        "panoids": [r["panoid"] for r in rows],
        "images_sha256": [file_hash(work / "images" / (r["panoid"] + ".jpg")) for r in rows],
        "preprocess": "full-panorama-surround-v1",
    }

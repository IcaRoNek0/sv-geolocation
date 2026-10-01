"""Shared, torch-free input preparation for both inference backends."""
import numpy as np
from PIL import Image, ImageOps

from utils.views import extract_views, surround_headings


def load_image(path, mode="screenshot"):
    if mode not in ("screenshot", "panorama", "auto"):
        raise ValueError(f"Unknown image mode: {mode}")
    with Image.open(path) as im:
        img = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = img.shape[:2]
    # Shape is only a hint; 2:1 screenshots still require an explicit mode.
    is_pano = mode == "panorama" or (mode == "auto" and abs(w / h - 2) <= 0.02)
    return img, is_pano


def to_views(img, is_pano, n_views, fov, size, n_max):
    if not 1 <= n_max <= 8 or size < 1 or n_views < 1:
        raise ValueError("Invalid view count or size")
    if is_pano:
        n = min(n_views, n_max)
        used = extract_views(img, surround_headings(n), fov_y=fov, size=size)
    else:
        # Keep buildings and signs undistorted. Training uses the same crop.
        used = np.asarray(ImageOps.fit(Image.fromarray(img), (size, size),
                                       method=Image.Resampling.BILINEAR))[None]
        n = 1
    views = np.zeros((n_max, size, size, 3), dtype=np.uint8)
    views[:n] = used
    mask = np.arange(n_max) < n
    return views, mask, n

"""样本准备与增强（纯 numpy，无 torch），可脱离 torch 测试。

视图数可变（1–8）并以一定概率取 1：推理输入可能只是一张截图，所以
"单图可用"必须是被训练过的能力，而不是只在推理时才遇到的情形。
"""
import io
import zlib

import numpy as np
from PIL import Image, ImageOps

from utils.views import extract_views, surround_headings, extract_perspective


class ViewConfig:
    """视图与增强的超参。"""

    def __init__(self, n_max=8, fov=90.0, size=224,
                 single_view_prob=0.15, min_views=1,
                 brightness=0.3, contrast=0.3, saturation=0.3,
                 channel_gain=0.05,
                 crop_scale=(0.7, 1.0), blur_prob=0.15,
                 fov_range=None, pitch_range=None, screenshot_prob=0.0,
                 jpeg_prob=0.0, jpeg_quality=(65, 95), resize_prob=0.0):
        self.n_max = n_max
        self.fov = fov
        self.size = size
        self.single_view_prob = single_view_prob
        self.min_views = min_views
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.channel_gain = channel_gain
        self.crop_scale = crop_scale
        self.blur_prob = blur_prob
        self.fov_range = fov_range
        self.pitch_range = pitch_range
        self.screenshot_prob = screenshot_prob
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality = jpeg_quality
        self.resize_prob = resize_prob
        if not 1 <= min_views <= n_max <= 8:
            raise ValueError("Require 1 <= min_views <= n_max <= 8")
        if not 0 <= single_view_prob <= 1:
            raise ValueError("single_view_prob must be in [0, 1]")


def choose_view_count(rng, cfg):
    """随机取视图数。以 single_view_prob 的概率取单视图，其余在 2..n_max 间取。"""
    if cfg.n_max == cfg.min_views:
        return cfg.n_max
    if rng.random() < cfg.single_view_prob:
        return cfg.min_views
    return int(rng.integers(max(2, cfg.min_views), cfg.n_max + 1))


def _resample_uint8(arr, out_hw):
    im = Image.fromarray(arr)
    return np.asarray(im.resize((out_hw[1], out_hw[0]), Image.BILINEAR))


def augment(views, rng, cfg):
    """亮度/对比度/饱和度抖动 + 缩放裁切 + 随机模糊。"""
    if cfg.brightness:
        b = rng.uniform(-cfg.brightness, cfg.brightness) * 255.0
        views = views.astype(np.float32) + b

    if cfg.contrast:
        c = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        mean = views.mean(axis=(1, 2, 3), keepdims=True)
        views = (views - mean) * c + mean

    if cfg.saturation:
        s = 1.0 + rng.uniform(-cfg.saturation, cfg.saturation)
        gray = views.mean(axis=3, keepdims=True)
        views = gray + (views - gray) * s

    # 轻微的通道增益，模拟白平衡差异
    if cfg.channel_gain:
        g = cfg.channel_gain
        gain = rng.uniform(1.0 - g, 1.0 + g, size=3).astype(np.float32)
        views = views * gain[None, None, None, :]

    views = np.clip(views, 0, 255).astype(np.uint8)

    # 缩放裁切：切出原图的 crop_scale 比例再放回原尺寸
    if cfg.crop_scale:
        lo, hi = cfg.crop_scale
        f = rng.uniform(lo, hi)
        if f < 0.999:
            h, w = views.shape[1], views.shape[2]
            ch, cw = min(h, max(1, int(h * f))), min(w, max(1, int(w * f)))
            out = np.empty_like(views)
            for i in range(len(views)):
                y0 = int(rng.integers(0, h - ch + 1))
                x0 = int(rng.integers(0, w - cw + 1))
                out[i] = _resample_uint8(views[i, y0:y0 + ch, x0:x0 + cw], (h, w))
            views = out

    if cfg.blur_prob and rng.random() < cfg.blur_prob:
        rad = float(rng.uniform(0.4, 1.2))
        views = np.stack([_blur(v, rad) for v in views])
    if cfg.resize_prob and rng.random() < cfg.resize_prob:
        h, w = views.shape[1:3]
        factor = float(rng.uniform(0.55, 1.0))
        small = (max(1, int(h * factor)), max(1, int(w * factor)))
        views = np.stack([_resample_uint8(_resample_uint8(v, small), (h, w))
                          for v in views])
    if cfg.jpeg_prob and rng.random() < cfg.jpeg_prob:
        quality = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
        encoded = []
        for v in views:
            buf = io.BytesIO()
            Image.fromarray(v).save(buf, format="JPEG", quality=quality)
            with Image.open(io.BytesIO(buf.getvalue())) as im:
                encoded.append(np.asarray(im.convert("RGB")))
        views = np.stack(encoded)
    return views


def sample_rng(seed, key, epoch=0):
    """Sample/epoch seed independent of DataLoader worker assignment."""
    return np.random.default_rng([seed, zlib.crc32(key.encode()), epoch])


def _blur(arr, rad):
    """降采样再升采样，等效于一次廉价模糊。"""
    h, w = arr.shape[0], arr.shape[1]
    small = (max(4, int(h / (1.0 + rad))), max(4, int(w / (1.0 + rad))))
    return _resample_uint8(_resample_uint8(arr, small), (h, w))


def make_sample(pano, rng, cfg, augment_on=True, eval_mode="random"):
    """由全景生成一条样本，返回 (views, vmask)。

    views (n_max, size, size, 3) uint8，未用槽位为 0；vmask (n_max,) bool。
    固定形状便于组装 batch，掩码保证池化忽略空槽。
    """
    if not augment_on and eval_mode in ("single", "panorama"):
        n = 1 if eval_mode == "single" else cfg.n_max
    else:
        n = choose_view_count(rng, cfg)
    offset = float(rng.uniform(0, 360)) if augment_on or eval_mode != "panorama" else 0.0
    fov = float(rng.choice(np.linspace(*cfg.fov_range, 5))) if augment_on and cfg.fov_range else cfg.fov
    pitch = float(rng.choice(np.linspace(*cfg.pitch_range, 3))) if augment_on and cfg.pitch_range else 0.0
    if augment_on and n == 1 and rng.random() < cfg.screenshot_prob:
        aspect = float(rng.choice([1.0, 4 / 3, 16 / 9]))
        shot = extract_perspective(pano, offset, pitch, fov,
                                   width=round(cfg.size * aspect), height=cfg.size)
        used = np.asarray(ImageOps.fit(Image.fromarray(shot), (cfg.size, cfg.size),
                                       method=Image.Resampling.BILINEAR))[None]
    else:
        headings = surround_headings(cfg.n_max, offset=offset)[:n]
        used = extract_views(pano, headings, fov_y=fov, size=cfg.size, pitch=pitch)
    if augment_on:
        used = augment(used, rng, cfg)

    views = np.zeros((cfg.n_max, cfg.size, cfg.size, 3), dtype=np.uint8)
    views[:n] = used
    vmask = np.zeros(cfg.n_max, dtype=bool)
    vmask[:n] = True
    return views, vmask

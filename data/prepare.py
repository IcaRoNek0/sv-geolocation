"""样本准备与增强（纯 numpy，无 torch），可脱离 torch 测试。

视图数可变（1–8）并以一定概率取 1：推理输入可能只是一张截图，所以
"单图可用"必须是被训练过的能力，而不是只在推理时才遇到的情形。
"""
import numpy as np
from PIL import Image

from utils.views import extract_views, surround_headings


class ViewConfig:
    """视图与增强的超参。"""

    def __init__(self, n_max=8, fov=90.0, size=224,
                 single_view_prob=0.15, min_views=1,
                 brightness=0.3, contrast=0.3, saturation=0.3,
                 channel_gain=0.05,
                 crop_scale=(0.7, 1.0), blur_prob=0.15):
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


def choose_view_count(rng, cfg):
    """随机取视图数。以 single_view_prob 的概率取单视图，其余在 2..n_max 间取。"""
    if rng.random() < cfg.single_view_prob:
        return cfg.min_views
    return int(rng.integers(2, cfg.n_max + 1))


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
            ch, cw = max(8, int(h * f)), max(8, int(w * f))
            out = np.empty_like(views)
            for i in range(len(views)):
                y0 = int(rng.integers(0, h - ch + 1))
                x0 = int(rng.integers(0, w - cw + 1))
                out[i] = _resample_uint8(views[i, y0:y0 + ch, x0:x0 + cw], (h, w))
            views = out

    if cfg.blur_prob and rng.random() < cfg.blur_prob:
        rad = float(rng.uniform(0.4, 1.2))
        views = np.stack([_blur(v, rad) for v in views])
    return views


def _blur(arr, rad):
    """降采样再升采样，等效于一次廉价模糊。"""
    h, w = arr.shape[0], arr.shape[1]
    small = (max(4, int(h / (1.0 + rad))), max(4, int(w / (1.0 + rad))))
    return _resample_uint8(_resample_uint8(arr, small), (h, w))


def make_sample(pano, rng, cfg, augment_on=True):
    """由全景生成一条样本，返回 (views, vmask)。

    views (n_max, size, size, 3) uint8，未用槽位为 0；vmask (n_max,) bool。
    固定形状便于组装 batch，掩码保证池化忽略空槽。
    """
    n = choose_view_count(rng, cfg)
    # 随机起始朝向，避免模型依赖"哪个方向恰好是 0 度"这种与地点无关的巧合
    offset = float(rng.uniform(0, 360))
    headings = surround_headings(cfg.n_max, offset=offset)[:n]

    used = extract_views(pano, headings, fov_y=cfg.fov, size=cfg.size)
    if augment_on:
        used = augment(used, rng, cfg)

    views = np.zeros((cfg.n_max, cfg.size, cfg.size, 3), dtype=np.uint8)
    views[:n] = used
    vmask = np.zeros(cfg.n_max, dtype=bool)
    vmask[:n] = True
    return views, vmask

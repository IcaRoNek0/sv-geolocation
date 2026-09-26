"""等距柱状全景 → 透视视图。

训练输入不是整幅全景：等距柱状投影在两极严重畸变，直接缩放会毁掉建筑与植被的
形状线索（而形状正是判县的主要依据）。标准做法是切成立方体面或透视视图。

这里切 N 个水平环绕的透视图，**视图数可变（1–8）**：单张截图能推理，多视图更准。
采样网格按 (输出尺寸, fov, 朝向) 缓存——朝向固定时，同一组网格可复用于所有样本。
"""
import math
from functools import lru_cache

import numpy as np

# 默认参数：8 个视图水平环绕，每视图 90° 垂直视场，输出 224×224。
# 90° × 8 = 720° 有重叠，覆盖完整水平圈且相邻视图留有余量。
DEFAULT_VIEWS = 8
DEFAULT_FOV_Y = 90.0
DEFAULT_SIZE = 224


@lru_cache(maxsize=64)
def _sample_maps(out_w, out_h, fov_y_deg, heading_deg, pitch_deg, src_w, src_h):
    """算出该 (尺寸, fov, 朝向) 下每个输出像素对应的源图像素坐标。

    返回 (u, v)，形状 (out_h, out_w)，float32。缓存后可反复复用。

    相机坐标系：x 向右、y 向上、z 向前。先俯仰（绕 x）后偏航（绕 y）。
    球面映射到等距柱状：经度→u 环绕，纬度→v（天顶为 0）。
    """
    focal = (out_h / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)

    u = np.arange(out_w, dtype=np.float32) + 0.5 - out_w / 2.0
    v = np.arange(out_h, dtype=np.float32) + 0.5 - out_h / 2.0
    uu, vv = np.meshgrid(u, v)

    rx, ry, rz = uu, -vv, np.full_like(uu, focal)

    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    # 俯仰：绕 x 轴
    ry2 = ry * cp - rz * sp
    rz2 = ry * sp + rz * cp

    ch, sh = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    # 偏航：绕 y 轴
    rx3 = rx * ch + rz2 * sh
    rz3 = -rx * sh + rz2 * ch

    norm = np.sqrt(rx3 * rx3 + ry2 * ry2 + rz3 * rz3)
    rx3, ry2, rz3 = rx3 / norm, ry2 / norm, rz3 / norm

    lat = np.arcsin(np.clip(ry2, -1.0, 1.0))
    lon = np.arctan2(rx3, rz3)

    # 先对源宽取模把 u 收回 [0, src_w)，再加 1 像素的环绕填充偏移。
    # 顺序不能颠倒：跨接缝的视图（朝向约 180°）u 会超出源宽一整段，
    # 若直接交给裁剪，取到的是边缘像素而不是绕回另一侧的内容，
    # 结果是 1/8 的视图边缘静默损坏。
    u_src = np.mod((lon / (2 * math.pi) + 0.5) * src_w, src_w) + 1.0
    v_src = (0.5 - lat / math.pi) * src_h

    return (u_src.astype(np.float32), np.clip(v_src, 0.0, src_h - 1.0).astype(np.float32))


def _bilinear(src, u, v):
    """双线性采样，u 已含环绕填充偏移，v 已裁剪在界内。

    必须先转 float32 再相减：源图是 uint8，`b - a` 在右邻更暗时会发生
    无符号回绕（5-10 变成 251），得到完全错误的颜色。
    """
    h, w, _ = src.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    du = (u - u0)[..., None].astype(np.float32)
    dv = (v - v0)[..., None].astype(np.float32)
    u0c = np.clip(u0, 0, w - 2)
    v0c = np.clip(v0, 0, h - 2)
    a = src[v0c, u0c].astype(np.float32)
    b = src[v0c, u0c + 1].astype(np.float32)
    c = src[v0c + 1, u0c].astype(np.float32)
    d = src[v0c + 1, u0c + 1].astype(np.float32)
    top = a + (b - a) * du
    bot = c + (d - c) * du
    return np.clip(top + (bot - top) * dv, 0, 255).astype(np.uint8)


def _wrapped(pano):
    """水平首尾相接的 1 像素填充。"""
    return np.concatenate([pano[:, -1:], pano, pano[:, :1]], axis=1)


def extract_views(pano, headings, fov_y=DEFAULT_FOV_Y, size=DEFAULT_SIZE,
                  pitch=0.0, wrapped=None):
    """按给定朝向列表切出透视视图。

    pano     (H, W, 3) uint8 等距柱状图
    headings 朝向角度序列，单位度
    wrapped  已填充过的全景，批量调用时传入以避免重复填充
    返回     (len(headings), size, size, 3) uint8
    """
    src = _wrapped(pano) if wrapped is None else wrapped
    src_h, src_w, _ = src.shape
    out = np.empty((len(headings), size, size, 3), dtype=np.uint8)
    for i, h in enumerate(headings):
        u, v = _sample_maps(size, size, fov_y, float(h) % 360.0, pitch,
                            src_w - 2, src_h)
        out[i] = _bilinear(src, u, v)
    return out


def surround_headings(n=DEFAULT_VIEWS, offset=0.0):
    """水平环绕的 n 个朝向。"""
    return [offset + i * 360.0 / n for i in range(n)]


def random_headings(n, rng):
    """随机朝向的 n 个视图，用于增强。"""
    return sorted(rng.uniform(0, 360) for _ in range(n))

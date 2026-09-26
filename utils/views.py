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


@lru_cache(maxsize=8)
def _base_maps(out_w, out_h, fov_y_deg, src_w, src_h):
    """朝向 0、俯仰 0 时的采样网格。只依赖几何参数，与被切的全景无关。

    缓存命中率是关键：训练时每个样本用随机朝向，若把朝向也算进缓存键，
    每个样本的每个视图都会重算一遍 224² 的三角函数，实测能把冒烟测试从
    几分钟拖到几十分钟。朝向的影响可以精确地化为对 u 的平移（见下），
    所以基准网格只需按几何参数缓存，命中率接近 100%。
    """
    focal = (out_h / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)

    u = np.arange(out_w, dtype=np.float32) + 0.5 - out_w / 2.0
    v = np.arange(out_h, dtype=np.float32) + 0.5 - out_h / 2.0
    uu, vv = np.meshgrid(u, v)

    rx, ry, rz = uu, -vv, np.full_like(uu, focal)
    norm = np.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / norm, ry / norm, rz / norm

    lat = np.arcsin(np.clip(ry, -1.0, 1.0))
    lon = np.arctan2(rx, rz)

    # 先对源宽取模把 u 收回 [0, src_w)，再加 1 像素的环绕填充偏移。
    # 顺序不能颠倒：跨接缝的视图 u 会超出源宽一整段，若直接交给裁剪，
    # 取到的是边缘像素而不是绕回另一侧的内容，结果是静默损坏。
    u_src = np.mod((lon / (2 * math.pi) + 0.5) * src_w, src_w) + 1.0
    v_src = (0.5 - lat / math.pi) * src_h

    return (u_src.astype(np.float32),
            np.clip(v_src, 0.0, src_h - 1.0).astype(np.float32))


def _sample_maps(out_w, out_h, fov_y_deg, heading_deg, pitch_deg, src_w, src_h):
    """每个输出像素对应的源图像素坐标 (u, v)。

    相机坐标系：x 向右、y 向上、z 向前。
    """
    u0, v = _base_maps(out_w, out_h, fov_y_deg, src_w, src_h)

    if pitch_deg == 0.0:
        # 等距柱状下改变朝向只是把经度整体平移，纬度不变——所以 u 平移、
        # v 原样。这是**精确**的，不是近似：绕竖直轴旋转不改变任何射线的
        # 纬度。俯仰不为 0 时该性质不成立，走下面的一般路径。
        shift = (heading_deg % 360.0) / 360.0 * src_w
        u = np.mod(u0 - 1.0 + shift, src_w) + 1.0
        return u, v

    return _general_maps(out_w, out_h, fov_y_deg, heading_deg, pitch_deg,
                         src_w, src_h)


def _general_maps(out_w, out_h, fov_y_deg, heading_deg, pitch_deg, src_w, src_h):
    """含俯仰的一般情形。慢路径，训练用不到（俯仰恒为 0）。"""
    focal = (out_h / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    u = np.arange(out_w, dtype=np.float32) + 0.5 - out_w / 2.0
    v = np.arange(out_h, dtype=np.float32) + 0.5 - out_h / 2.0
    uu, vv = np.meshgrid(u, v)
    rx, ry, rz = uu, -vv, np.full_like(uu, focal)

    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    ry2 = ry * cp - rz * sp
    rz2 = ry * sp + rz * cp

    ch, sh = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    rx3 = rx * ch + rz2 * sh
    rz3 = -rx * sh + rz2 * ch

    norm = np.sqrt(rx3 * rx3 + ry2 * ry2 + rz3 * rz3)
    rx3, ry2, rz3 = rx3 / norm, ry2 / norm, rz3 / norm

    lat = np.arcsin(np.clip(ry2, -1.0, 1.0))
    lon = np.arctan2(rx3, rz3)
    u_src = np.mod((lon / (2 * math.pi) + 0.5) * src_w, src_w) + 1.0
    v_src = (0.5 - lat / math.pi) * src_h
    return (u_src.astype(np.float32),
            np.clip(v_src, 0.0, src_h - 1.0).astype(np.float32))


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


try:
    from scipy.ndimage import map_coordinates as _map_coordinates
    _HAS_SCIPY = True
except ImportError:                                   # pragma: no cover
    _HAS_SCIPY = False


def _resample(src, u, v, n, size):
    """在 (u, v) 处采样。u/v 已按 n 个视图纵向堆叠。

    优先走 scipy：它是编译好的 C 循环，不必像下面的 numpy 版本那样先
    分配四张 (n,size,size,3) 的中间数组再算。实测（手机 ARM）8 个 224²
    视图：scipy 合并调用 62 ms，numpy 合并 127 ms，逐视图 numpy 150 ms。
    带宽受限时这个差距在 Colab 上同样存在。
    """
    if _HAS_SCIPY:
        coords = np.stack([v.ravel(), u.ravel()])
        chans = [
            _map_coordinates(src[..., c].astype(np.float32), coords,
                             order=1, mode="nearest")
            for c in range(3)
        ]
        out = np.stack(chans, axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8).reshape(n, size, size, 3)
    return _bilinear(src, u, v).reshape(n, size, size, 3)


def extract_views(pano, headings, fov_y=DEFAULT_FOV_Y, size=DEFAULT_SIZE,
                  pitch=0.0, wrapped=None):
    """按给定朝向列表切出透视视图。

    pano     (H, W, 3) uint8 等距柱状图
    headings 朝向角度序列，单位度
    wrapped  已填充过的全景，批量调用时传入以避免重复填充
    返回     (len(headings), size, size, 3) uint8

    所有视图合并成一次采样调用——分别调用时每个视图都要重建坐标数组并
    各自走一遍采样循环，开销随视图数线性增长。
    """
    if not headings:
        return np.zeros((0, size, size, 3), dtype=np.uint8)
    src = _wrapped(pano) if wrapped is None else wrapped
    src_h, src_w, _ = src.shape

    us, vs = [], []
    for h in headings:
        u, v = _sample_maps(size, size, fov_y, float(h) % 360.0, pitch,
                            src_w - 2, src_h)
        us.append(u)
        vs.append(v)
    return _resample(src, np.concatenate(us, axis=0), np.concatenate(vs, axis=0),
                     len(headings), size)


def surround_headings(n=DEFAULT_VIEWS, offset=0.0):
    """水平环绕的 n 个朝向。"""
    return [offset + i * 360.0 / n for i in range(n)]


def random_headings(n, rng):
    """随机朝向的 n 个视图，用于增强。"""
    return sorted(rng.uniform(0, 360) for _ in range(n))

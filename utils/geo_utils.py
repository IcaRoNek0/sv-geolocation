"""地理计算：距离、真实点位表、加权几何中位数选点。

选点用各县的真实街景点位而非质心：质心可能落在水域或山区，中位数不保证落在点集中、道路上或最高概率县内。给定县级概率分布后即求加权几何中位数

    min_c Σ_i p_i · d(c, c_i)

用 Weiszfeld 迭代。主库 lng/lat 是 GCJ02；球面距离对 CRS 只是近似，但
同一 CRS 内部比较是一致的。
"""
import math
from pathlib import Path

import numpy as np

EARTH_R = 6371008.8          # 平均地球半径，米
WEISZFELD_EPS = 1e-7         # 相对收敛阈值
WEISZFELD_MAX_ITER = 500
COINCIDENT_M = 0.5           # 判定迭代点与数据点重合的距离阈值

# 坐标回归头的归一化范围（中国含港澳）。训练与推理必须共用同一个边界，
# 否则模型的输出会被两套尺度解释成不同的地点。
CHINA_BBOX = (73.0, 18.0, 135.0, 54.0)   # lon_min, lat_min, lon_max, lat_max


def normalize_coords(lon, lat):
    """经纬度 → [0,1] 的 (x, y)，供坐标回归头使用。"""
    lon0, lat0, lon1, lat1 = CHINA_BBOX
    x = (np.asarray(lon, dtype=np.float64) - lon0) / (lon1 - lon0)
    y = (np.asarray(lat, dtype=np.float64) - lat0) / (lat1 - lat0)
    return x, y


def denormalize_coords(x, y):
    """[0,1] 的 (x, y) → 经纬度。裁剪到合法范围，越界输出没有意义。"""
    lon0, lat0, lon1, lat1 = CHINA_BBOX
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    return lon0 + x * (lon1 - lon0), lat0 + y * (lat1 - lat0)


def haversine(lon1, lat1, lon2, lat2):
    """球面大圆距离，单位米。参数可为标量或数组。"""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def local_xy(lon, lat, lon0, lat0):
    """局部等距投影，供中位数迭代用。县域尺度下误差远小于定位精度。"""
    cos0 = math.cos(math.radians(lat0))
    x = (np.asarray(lon) - lon0) * math.radians(1.0) * EARTH_R * cos0
    y = (np.asarray(lat) - lat0) * math.radians(1.0) * EARTH_R
    return x, y


def xy_to_lonlat(x, y, lon0, lat0):
    cos0 = math.cos(math.radians(lat0))
    lon = lon0 + np.asarray(x) / (math.radians(1.0) * EARTH_R * cos0)
    lat = lat0 + np.asarray(y) / (math.radians(1.0) * EARTH_R)
    return lon, lat


def weighted_geometric_median(points, weights):
    """加权几何中位数：最小化 Σ w_i·‖c − p_i‖，points 为 (N,2) 的 (lon,lat)。

    在局部等距平面上迭代。退化情形是迭代点恰好落在数据点上（该处梯度
    不可导），需检验次梯度条件，不能直接当成最优。
    """
    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points 必须是 (N, 2) 的 (lon, lat)")
    if not np.isfinite(pts).all() or not np.isfinite(w).all():
        raise ValueError("Non-finite coordinates or weights")
    if len(pts) == 0:
        raise ValueError("点位为空")
    if len(pts) != len(w):
        raise ValueError("点位与权重长度不一致")

    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError("权重全为零")
    w = w / total

    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1])

    lon0, lat0 = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x, y = local_xy(pts[:, 0], pts[:, 1], lon0, lat0)

    # 用加权重心起步
    cx, cy = float((w * x).sum()), float((w * y).sum())

    for _ in range(WEISZFELD_MAX_ITER):
        d = np.hypot(x - cx, y - cy)
        hit = d < COINCIDENT_M
        nonhit = ~hit
        if not nonhit.any():
            break
        inv = w[nonhit] / d[nonhit]
        mass = inv.sum()
        if mass == 0:
            break
        nx = float((inv * x[nonhit]).sum() / mass)
        ny = float((inv * y[nonhit]).sum() / mass)
        if hit.any():
            # Coincidence alone does not imply optimality (modified Weiszfeld).
            residual = math.hypot(float((inv * (x[nonhit] - cx)).sum()),
                                  float((inv * (y[nonhit] - cy)).sum()))
            hit_mass = float(w[hit].sum())
            if residual <= hit_mass:
                break
            fraction = hit_mass / residual
            nx, ny = ((1 - fraction) * nx + fraction * cx,
                      (1 - fraction) * ny + fraction * cy)
        shift = math.hypot(nx - cx, ny - cy)
        cx, cy = nx, ny
        if shift <= WEISZFELD_EPS * max(1.0, math.hypot(cx, cy)):
            break

    lon, lat = xy_to_lonlat(cx, cy, lon0, lat0)
    return float(lon), float(lat)


def expected_distance(center, points, weights):
    """Σ w_i·d(center, p_i)，用于评估选点质量（米）。"""
    lon, lat = center
    return float(
        (weights * haversine(lon, lat, points[:, 0], points[:, 1])).sum()
    )


class CountyPoints:
    """每县真实点位表（由 collect/sample_pool.py 产出）。"""

    def __init__(self, npz_path):
        z = np.load(Path(npz_path))
        self.adcodes = [str(c) for c in z["adcodes"]]
        self._index = {a: i for i, a in enumerate(self.adcodes)}
        self._coords = z["coords"].reshape(-1, 2)
        self._offsets = z["offsets"]

    def __len__(self):
        return len(self.adcodes)

    def __contains__(self, adcode):
        return adcode in self._index

    def points(self, adcode):
        """该县的真实点位，(N, 2) 的 (lon, lat)。"""
        i = self._index.get(adcode)
        if i is None:
            raise KeyError(f"点位表中没有 {adcode}")
        a, b = int(self._offsets[i]), int(self._offsets[i + 1])
        return self._coords[a:b]

    def centroid(self, adcode):
        p = self.points(adcode)
        return float(p[:, 0].mean()), float(p[:, 1].mean())


def select_point(probs, county_points, top_k=20, spread="uniform"):
    """给定县级概率分布，选出期望距离最小的坐标。

    probs 为 {adcode: 概率} 或 (adcodes, probs)，不必归一。
    top_k 只考虑概率最高的 K 个县：尾部贡献可忽略，却会把参与迭代的点数
    放大一个量级。返回 (lon, lat, 用到的县列表)。
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    items = [(a, float(p)) for a, p in
             (probs.items() if isinstance(probs, dict) else zip(*probs))]
    items = [(a, p) for a, p in items if p > 0 and a in county_points]
    if not items:
        raise ValueError("没有可用的候选县（点位表里一个都没有）")
    items.sort(key=lambda kv: -kv[1])
    items = items[:top_k]

    pts, wts = [], []
    for adcode, p in items:
        q = county_points.points(adcode)
        if len(q) == 0:
            continue
        pts.append(q)
        wts.append(np.full(len(q), p / len(q)) if spread == "uniform" else
                   np.full(len(q), p))
    if not pts:
        raise ValueError("候选县在点位表中都没有点位")

    pts = np.concatenate(pts, axis=0)
    wts = np.concatenate(wts, axis=0)
    lon, lat = weighted_geometric_median(pts, wts)
    return lon, lat, [a for a, _ in items]

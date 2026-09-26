"""类别空间与地理软标签。

不把标签当成互斥的 one-hot：相邻县的地貌与建筑高度相似，one-hot 会把
"隔壁县"和"隔着半个中国的县"同等惩罚。改用地理软标签

    w_j ∝ exp(-d(质心_j, 质心_i) / τ)

让邻近的县分享一部分概率质量。在本方案"无文字线索 + 每县约 70 个点位"的
条件下，这是把"隔壁县"从完全错误变成部分正确的唯一低成本手段。

τ 的取法：τ = D_half / ln 2，即相距 D_half 的县拿到约一半权重。
默认 D_half = 50 公里（同省相邻县的距离量级）。
"""
from pathlib import Path

import numpy as np

from utils.geo_utils import CountyPoints, haversine

DEFAULT_HALF_KM = 50.0


def build_classes(samples, county_points, min_samples=1):
    """确定类别空间。

    含**全部**有点位的县，而不是只含有训练样本的县：整县留出的测试县必须
    留在类别空间里，否则"预测没见过的县"无从评估。训练样本为 0 的县其
    logit 天然接近零，不需要特殊处理。

    返回 (adcodes, index) —— adcodes 为排序后的县码列表。
    """
    train_counts = {}
    for s in samples:
        train_counts[s["adcode"]] = train_counts.get(s["adcode"], 0) + 1
    codes = sorted(a for a in county_points.adcodes
                   if train_counts.get(a, 0) >= min_samples)
    return codes, {a: i for i, a in enumerate(codes)}


def centroids(adcodes, county_points):
    """各县点位质心，(N, 2) 的 (lon, lat)。"""
    out = np.zeros((len(adcodes), 2), dtype=np.float64)
    for i, a in enumerate(adcodes):
        out[i] = county_points.centroid(a)
    return out


def soft_targets(cent, half_km=DEFAULT_HALF_KM, blob=None):
    """预先算好 (C, C) 的软标签矩阵：第 i 行是类别 i 的软目标分布。

    类别数约 2600，矩阵约 2600² × 4 字节 = 27 MB，一次算好随取随用，
    比每个 batch 现算省得多。
    """
    tau = (half_km * 1000.0) / np.log(2.0)
    d = haversine(cent[:, None, 0], cent[:, None, 1],
                  cent[None, :, 0], cent[None, :, 1])
    w = np.exp(-d / tau)
    # 自身权重为 1，天然是最大值；归一化成分布
    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32)


def hierarchical_labels(adcodes, city_of, province_of):
    """由县码推出地级市与省级标签，返回 (city_codes, city_idx, prov_codes, prov_idx)。"""
    cities = sorted({city_of[a] for a in adcodes})
    provs = sorted({province_of[a] for a in adcodes})
    cidx = {c: i for i, c in enumerate(cities)}
    pidx = {p: i for i, p in enumerate(provs)}
    city_arr = np.array([cidx[city_of[a]] for a in adcodes], dtype=np.int64)
    prov_arr = np.array([pidx[province_of[a]] for a in adcodes], dtype=np.int64)
    return cities, city_arr, provs, prov_arr


def city_of_adcode(adcode):
    """6 位县级 adcode -> 6 位地级市 adcode。

    市级码 = 前 4 位 + '00'。省直辖县级单位（如 419001 济源）没有地级市，
    这时地级标签取省码，让它们各自成组而不是与真正的市混在一起。
    """
    if not adcode or len(adcode) < 6:
        return adcode
    if adcode[2:4] == "90":          # 省直辖县级单位
        return adcode[:2] + "0000"
    return adcode[:4] + "00"


def province_of_adcode(adcode):
    return adcode[:2] + "0000" if adcode and len(adcode) >= 2 else adcode

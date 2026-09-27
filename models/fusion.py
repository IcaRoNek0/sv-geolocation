"""两条线索的概率融合与选点。

    log p ∝ w_env·log p_env + w_text·log p_text

在对数空间加权：两条线索可信度差异大时，直接加权概率会被其中一条的峰值
主导。本期只有环境线索，text_probs 传 None 即退化为单线索。
"""
import numpy as np

from utils.geo_utils import CountyPoints, select_point

LOG_FLOOR = 1e-12


def to_log_probs(probs, adcodes):
    """把 {县码: 概率} 转成与 adcodes 对齐的对数概率向量。"""
    v = np.full(len(adcodes), LOG_FLOOR, dtype=np.float64)
    index = {a: i for i, a in enumerate(adcodes)}
    for a, p in probs.items():
        i = index.get(a)
        if i is not None and p > 0:
            v[i] = max(float(p), LOG_FLOOR)
    return np.log(v)


def fuse(env_probs, text_probs=None, w_env=1.0, w_text=1.0, adcodes=None,
         temperature=1.0):
    """融合两条线索，返回归一化的 {县码: 概率}。

    两个输入可以是 {县码: 概率} 或与 adcodes 对齐的数组。
    """
    if adcodes is None:
        adcodes = sorted(env_probs if isinstance(env_probs, dict) else
                         range(len(env_probs)))

    def as_log(x):
        if x is None:
            return None
        if isinstance(x, dict):
            return to_log_probs(x, adcodes)
        arr = np.asarray(x, dtype=np.float64)
        if len(arr) != len(adcodes):
            raise ValueError(f"分布长度 {len(arr)} 与类别数 {len(adcodes)} 不符")
        return np.log(np.maximum(arr, LOG_FLOOR))

    logp = w_env * as_log(env_probs)
    if text_probs is not None:
        logp = logp + w_text * as_log(text_probs)

    logp = logp / max(temperature, LOG_FLOOR)
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return {a: float(v) for a, v in zip(adcodes, p)}


def predict_location(probs, points: CountyPoints, top_k=20, spread="uniform"):
    """概率分布 → 坐标。返回 (lon, lat, 用到的县)。"""
    return select_point(probs, points, top_k=top_k, spread=spread)


def top_counties(probs, k=5, adcodes=None):
    """取概率最高的 k 个县，返回 [(县码, 概率), ...]。"""
    items = probs.items() if isinstance(probs, dict) else zip(adcodes, probs)
    return sorted(items, key=lambda kv: -kv[1])[:k]

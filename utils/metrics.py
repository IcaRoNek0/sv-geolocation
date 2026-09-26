"""评估指标。

县级指标必须同时报**宏平均**与多数类基线：主省占三分之二，县级样本量在
各县间极不均衡（可用量 p25 = 86、p75 = 502），微平均会被样本多的县主导，
看起来比实际好。只报微平均等于自欺。

距离误差是整县留出划分下唯一有意义的指标——模型从没见过那个县，指望它
在 2600 个类里正好点中一个没学过的县是不现实的，但它应该落在附近。
"""
import numpy as np

from utils.geo_utils import haversine


def top_k_accuracy(scores, targets, k=1):
    """scores (N, C)，targets (N,) 类别下标。返回 top-k 命中率。"""
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return float((top == targets[:, None]).any(axis=1).mean())


def per_class_recall(scores, targets, n_classes):
    """每类召回率，(n_classes,) 的数组，无样本的类为 nan。"""
    pred = np.asarray(scores).argmax(axis=1)
    targets = np.asarray(targets)
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = targets == c
        if m.any():
            out[c] = float((pred[m] == c).mean())
    return out


def macro_recall(scores, targets, n_classes):
    """宏平均召回。样本量为 0 的类不计入——否则一个没见过的类会拉低平均。"""
    r = per_class_recall(scores, targets, n_classes)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


def majority_baseline(targets, n_classes):
    """多数类基线：永远预测训练集里样本最多的类。

    必须与模型指标并列展示，否则无法判断模型是真的在学地理，还是只是
    背下了先验。
    """
    targets = np.asarray(targets)
    counts = np.bincount(targets, minlength=n_classes)
    if counts.sum() == 0:
        return float("nan")
    return float(counts.max() / counts.sum())


def distance_errors(pred_lonlat, true_lonlat):
    """预测坐标与真实坐标的球面距离，单位公里，(N,) 数组。"""
    pred = np.asarray(pred_lonlat, dtype=np.float64)
    true = np.asarray(true_lonlat, dtype=np.float64)
    return haversine(pred[:, 0], pred[:, 1], true[:, 0], true[:, 1]) / 1000.0


def distance_summary(pred_lonlat, true_lonlat):
    d = distance_errors(pred_lonlat, true_lonlat)
    return {
        "median_km": float(np.median(d)),
        "mean_km": float(d.mean()),
        "p25_km": float(np.percentile(d, 25)),
        "p75_km": float(np.percentile(d, 75)),
        "within_25km": float((d <= 25).mean()),
        "within_100km": float((d <= 100).mean()),
    }


def geo_score(pred_lonlat, true_lonlat, decay_km=150.0, max_score=5000.0):
    """常见的猜点计分：5000·exp(-d/D)。用于对照游戏口径。"""
    d = distance_errors(pred_lonlat, true_lonlat)
    return float((max_score * np.exp(-d / decay_km)).mean())


def nearest_neighbour_consistency(scores, targets, class_centroids, k=5, radius_km=150.0):
    """top-k 候选里有多少比例落在真值附近——软标签是否真的在起作用的直接证据。

    这个指标比 top-1 更早给出信号：模型可能还点不中正确的县，但只要它的
    候选**连成一片且围绕真值**，就说明它在学地理而不是背先验。
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    cent = np.asarray(class_centroids)
    hits = np.zeros(len(targets), dtype=bool)
    for i in range(len(targets)):
        t = cent[targets[i]]
        c = cent[top[i]]
        d = haversine(t[0], t[1], c[:, 0], c[:, 1]) / 1000.0
        hits[i] = (d <= radius_km).any()
    return float(hits.mean())


def summarize(scores, targets, n_classes, class_centroids=None):
    """一次性给出所有关键指标。"""
    out = {
        "top1": top_k_accuracy(scores, targets, 1),
        "top5": top_k_accuracy(scores, targets, 5),
        "top10": top_k_accuracy(scores, targets, 10),
        "macro_recall": macro_recall(scores, targets, n_classes),
        "majority_baseline": majority_baseline(targets, n_classes),
        "n": int(len(targets)),
    }
    if class_centroids is not None:
        out["top5_nearby_150km"] = nearest_neighbour_consistency(
            scores, targets, class_centroids, k=5, radius_km=150.0)
    return out

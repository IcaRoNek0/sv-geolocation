"""类别空间、软标签与评估指标的测试。

指标这块最容易自欺：微平均会被样本多的县主导。这里专门构造"多数类看起来
很好、宏平均很差"的场景，确保两者确实被区分开。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.labels import city_of_adcode, province_of_adcode, soft_targets  # noqa: E402
from utils.geo_utils import haversine  # noqa: E402
from utils.metrics import (  # noqa: E402
    macro_recall,
    majority_baseline,
    nearest_neighbour_consistency,
    top_k_accuracy,
)


class TestAdcodeHierarchy(unittest.TestCase):
    def test_city_and_province(self):
        self.assertEqual(city_of_adcode("510104"), "510100")      # 成都青羊区
        self.assertEqual(province_of_adcode("510104"), "510000")
        self.assertEqual(city_of_adcode("513336"), "513300")      # 甘孜州乡城县

    def test_province_directly_administered(self):
        """省直辖县级单位（419001 济源）没有地级市，应归到省而不是硬凑一个市。"""
        self.assertEqual(city_of_adcode("419001"), "410000")

    def test_districtless_city_maps_to_itself(self):
        """东莞这类不设区的市，末级码就是市级码，应映射到自身。"""
        self.assertEqual(city_of_adcode("441900"), "441900")


class TestSoftTargets(unittest.TestCase):
    def setUp(self):
        # 三个县：A 与 B 相距约 30 公里，C 远在 1000 公里外
        self.cent = np.array([
            [104.00, 30.00],
            [104.30, 30.00],     # 约 29 km
            [120.00, 40.00],     # 约 1800 km
        ])

    def test_rows_are_distributions(self):
        w = soft_targets(self.cent, half_km=50.0)
        self.assertTrue(np.allclose(w.sum(axis=1), 1.0, atol=1e-5))

    def test_self_weight_is_largest(self):
        w = soft_targets(self.cent, half_km=50.0)
        self.assertTrue(np.all(np.argmax(w, axis=1) == np.arange(3)))

    def test_nearby_neighbour_gets_more_than_distant(self):
        """这正是软标签存在的理由：邻近县应当分到明显更多的权重。"""
        w = soft_targets(self.cent, half_km=50.0)
        self.assertGreater(w[0, 1], w[0, 2] * 10)

    def test_half_km_controls_sharing(self):
        """半衰距离越大，质量摊得越开：自身权重下降、邻居权重上升。"""
        tight = soft_targets(self.cent, half_km=20.0)    # 衰减快，少分享
        loose = soft_targets(self.cent, half_km=200.0)   # 衰减慢，多分享
        self.assertGreater(tight[0, 0], loose[0, 0])
        self.assertLess(tight[0, 1], loose[0, 1])

    def test_diagonal_dominates(self):
        w = soft_targets(self.cent, half_km=50.0)
        self.assertTrue(np.all(np.diag(w) > 0.5))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # 3 个类，其中类 1 有 90 个样本——多数类基线很高
        self.targets = np.array([0] * 5 + [1] * 90 + [2] * 5)
        n = len(self.targets)
        self.scores = np.zeros((n, 3), dtype=np.float32)
        self.scores[:, 1] = 1.0            # 永远预测类 1

    def test_majority_baseline_matches_by_construction(self):
        self.assertAlmostEqual(majority_baseline(self.targets, 3), 0.9, places=6)

    def test_always_predicting_majority_looks_good_micro_but_bad_macro(self):
        """本测试的意义：证明微平均会掩盖模型什么都没学到。"""
        self.assertAlmostEqual(top_k_accuracy(self.scores, self.targets, 1), 0.9,
                               places=6)
        self.assertLess(macro_recall(self.scores, self.targets, 3), 0.4)

    def test_top_k_handles_small_k_and_small_c(self):
        s = np.array([[0.1, 0.9], [0.8, 0.2]])
        t = np.array([1, 0])
        self.assertEqual(top_k_accuracy(s, t, 1), 1.0)
        self.assertEqual(top_k_accuracy(s, t, 5), 1.0)   # k 超出类别数应被夹住

    def test_nearby_consistency_detects_geographic_sense(self):
        """候选县落在真值附近时该指标应为 1，落在远处应为 0。"""
        cent = np.array([[104.0, 30.0], [104.1, 30.0], [120.0, 40.0]])
        # 真值是类 0，但模型预测类 1——地理上只差约 10 公里
        scores = np.zeros((1, 3), dtype=np.float32)
        scores[0, 1] = 1.0
        self.assertEqual(
            nearest_neighbour_consistency(scores, np.array([0]), cent,
                                          k=1, radius_km=150.0), 1.0)
        # 同样预测类 1，但真值是类 2——相差近 2000 公里
        self.assertEqual(
            nearest_neighbour_consistency(scores, np.array([2]), cent,
                                          k=1, radius_km=150.0), 0.0)

    def test_centroid_ordering_consistent_with_haversine(self):
        a = np.array([104.0, 30.0])
        b = np.array([104.3, 30.0])
        self.assertLess(haversine(*a, *b), haversine(*a, 120.0, 40.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

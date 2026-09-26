"""选点与地理计算的测试。

核心断言是：Weiszfeld 求出的点，其目标函数值必须不劣于**任何**网格候选点。
只检查"看起来在中间"是不够的——加权情形的正确答案常常偏离重心。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.geo_utils import (  # noqa: E402
    CountyPoints,
    expected_distance,
    haversine,
    weighted_geometric_median,
)

POINTS_NPZ = Path(__file__).resolve().parent.parent / "data" / "pool" / "county_points.npz"


class TestHaversine(unittest.TestCase):
    def test_known_distance(self):
        """北京到上海约 1067 公里。"""
        d = haversine(116.4074, 39.9042, 121.4737, 31.2304)
        self.assertAlmostEqual(d / 1000, 1067, delta=15)

    def test_zero_distance(self):
        self.assertAlmostEqual(haversine(104.0, 30.5, 104.0, 30.5), 0.0, places=6)


class TestWeightedMedian(unittest.TestCase):
    def test_symmetric_square_is_center(self):
        pts = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
        w = np.ones(4)
        lon, lat = weighted_geometric_median(pts, w)
        self.assertAlmostEqual(lon, 1.0, places=4)
        self.assertAlmostEqual(lat, 1.0, places=4)

    def test_two_points_is_midpoint(self):
        pts = np.array([[100.0, 30.0], [100.2, 30.0]])
        lon, lat = weighted_geometric_median(pts, np.ones(2))
        self.assertAlmostEqual(lon, 100.1, places=4)
        self.assertAlmostEqual(lat, 30.0, places=4)

    def test_single_point(self):
        pts = np.array([[104.5, 30.5]])
        self.assertEqual(weighted_geometric_median(pts, np.ones(1)), (104.5, 30.5))

    def test_dominant_weight_pulls_to_that_point(self):
        """权重压倒性集中时，最优解就是那个数据点本身（Weiszfeld 的退化情形）。"""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        w = np.array([1.0, 1.0, 1000.0])
        lon, lat = weighted_geometric_median(pts, w)
        self.assertAlmostEqual(lon, 10.0, places=2)

    def test_beats_grid_search(self):
        """随机点位下，解的目标值必须不劣于网格搜索的最优值。"""
        rng = np.random.default_rng(7)
        for _ in range(20):
            pts = rng.uniform(0, 1, size=(25, 2))
            w = rng.uniform(0.1, 1.0, size=25)
            lon, lat = weighted_geometric_median(pts, w)
            got = expected_distance((lon, lat), pts, w / w.sum())

            # 粗网格搜索作为对照。必须与 expected_distance 用同一度量（米），
            # 否则是在拿度数和米比较。
            g = np.linspace(pts.min() - 0.1, pts.max() + 0.1, 60)
            gx, gy = np.meshgrid(g, g)
            d = haversine(gx.ravel()[:, None], gy.ravel()[:, None],
                          pts[None, :, 0], pts[None, :, 1])
            best_grid = float((d * (w / w.sum())[None, :]).sum(1).min())

            self.assertLessEqual(
                got, best_grid + 1e-9,
                f"Weiszfeld 解劣于网格搜索：{got:.6f} > {best_grid:.6f}",
            )

    def test_empty_and_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            weighted_geometric_median(np.zeros((0, 2)), np.zeros(0))
        with self.assertRaises(ValueError):
            weighted_geometric_median(np.zeros((3, 2)), np.ones(2))
        with self.assertRaises(ValueError):
            weighted_geometric_median(np.zeros((3, 2)), np.zeros(3))
        with self.assertRaises(ValueError):
            weighted_geometric_median(np.zeros((3, 3)), np.ones(3))


@unittest.skipUnless(POINTS_NPZ.exists(), "点位表尚未生成")
class TestRealPointSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CountyPoints(POINTS_NPZ)

    def test_point_table_loads(self):
        self.assertGreater(len(self.cp), 2000)
        p = self.cp.points("510105")
        self.assertGreater(len(p), 0)
        # 成都青羊区应在四川盆地范围内
        self.assertTrue(100 < p[:, 0].min() and p[:, 0].max() < 110)
        self.assertTrue(25 < p[:, 1].min() and p[:, 1].max() < 35)

    def test_selection_lands_inside_the_county(self):
        """把全部概率给一个县，选出的点必须落在该县点位的包围盒内。"""
        from utils.geo_utils import select_point
        for adcode in ("510105", "510104", "510107"):
            pts = self.cp.points(adcode)
            lon, lat, used = select_point({adcode: 1.0}, self.cp)
            self.assertEqual(used, [adcode])
            pad = 0.01  # 约 1 公里余量
            self.assertGreaterEqual(lon, pts[:, 0].min() - pad)
            self.assertLessEqual(lon, pts[:, 0].max() + pad)
            self.assertGreaterEqual(lat, pts[:, 1].min() - pad)
            self.assertLessEqual(lat, pts[:, 1].max() + pad)

    def test_uniform_over_province_lands_in_province(self):
        """四川省内均匀分布时，选点必须仍在四川。"""
        from utils.geo_utils import select_point
        sic = [a for a in self.cp.adcodes if a.startswith("51")]
        probs = {a: 1.0 for a in sic}
        lon, lat, _ = select_point(probs, self.cp, top_k=40)
        self.assertTrue(97 < lon < 109, f"经度 {lon} 落在四川之外")
        self.assertTrue(26 < lat < 34, f"纬度 {lat} 落在四川之外")


if __name__ == "__main__":
    unittest.main(verbosity=2)

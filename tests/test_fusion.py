"""线索融合与选点接口的测试。

融合在对数概率空间做，这个选择有个容易被忽略的后果：某条线索给出严格为
零的概率时，对数会取到下限，**该县不会被彻底排除**。这是刻意的——软标签
和模型误差都可能让真实县拿到极低分，一票否决会让融合丧失纠错能力。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.fusion import fuse, predict_location, to_log_probs, top_counties  # noqa: E402
from utils.geo_utils import CountyPoints  # noqa: E402

POINTS_NPZ = Path(__file__).resolve().parent.parent / "data" / "pool" / "county_points.npz"

CODES = ["510105", "510104", "510107"]


class TestFuse(unittest.TestCase):
    def test_single_clue_is_identity(self):
        p = {"510105": 0.7, "510104": 0.2, "510107": 0.1}
        out = fuse(p, adcodes=CODES)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        self.assertAlmostEqual(out["510105"], 0.7, places=5)

    def test_agreement_sharpens(self):
        """两条线索一致时，峰值应当更尖锐。"""
        a = {"510105": 0.5, "510104": 0.3, "510107": 0.2}
        b = {"510105": 0.6, "510104": 0.25, "510107": 0.15}
        single = fuse(a, adcodes=CODES)
        both = fuse(a, b, adcodes=CODES)
        self.assertGreater(both["510105"], single["510105"])

    def test_disagreement_is_resolved_by_weight(self):
        a = {"510105": 0.9, "510104": 0.05, "510107": 0.05}
        b = {"510105": 0.05, "510104": 0.9, "510107": 0.05}
        env_wins = fuse(a, b, w_env=3.0, w_text=1.0, adcodes=CODES)
        text_wins = fuse(a, b, w_env=1.0, w_text=3.0, adcodes=CODES)
        self.assertEqual(max(env_wins, key=env_wins.get), "510105")
        self.assertEqual(max(text_wins, key=text_wins.get), "510104")

    def test_zero_probability_does_not_veto(self):
        """某条线索给 0 的县不该被彻底排除——这是对数空间融合的刻意后果。"""
        a = {"510105": 1.0, "510104": 0.0, "510107": 0.0}
        b = {"510105": 0.0, "510104": 1.0, "510107": 0.0}
        out = fuse(a, b, adcodes=CODES)
        self.assertGreater(out["510104"], 0.0)
        self.assertTrue(np.isfinite(list(out.values())).all())

    def test_output_is_distribution(self):
        a = {"510105": 0.5, "510104": 0.5}
        out = fuse(a, {"510105": 0.3, "510104": 0.7}, adcodes=CODES)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        self.assertTrue(all(v >= 0 for v in out.values()))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            fuse(np.array([0.5, 0.5]), adcodes=CODES)

    def test_top_counties_ordering(self):
        p = {"510105": 0.2, "510104": 0.5, "510107": 0.3}
        top = top_counties(p, k=2)
        self.assertEqual([a for a, _ in top], ["510104", "510107"])


@unittest.skipUnless(POINTS_NPZ.exists(), "点位表尚未生成")
class TestPredictLocation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = CountyPoints(POINTS_NPZ)

    def test_location_near_the_confident_county(self):
        lon, lat, used = predict_location({"510105": 1.0}, self.cp)
        pts = self.cp.points("510105")
        self.assertLess(float(np.hypot(lon - pts[:, 0].mean(),
                                       lat - pts[:, 1].mean())), 0.2)
        self.assertEqual(used, ["510105"])

    def test_tail_counties_are_dropped_by_top_k(self):
        """top_k 之外的县不参与选点——尾部对期望距离的贡献可忽略。"""
        probs = {a: 1.0 for a in self.cp.adcodes if a.startswith("51")}
        _, _, used = predict_location(probs, self.cp, top_k=5)
        self.assertLessEqual(len(used), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""样本准备与增强的测试（纯 numpy，可在本机运行）。

重点在掩码与可变视图数：池化时必须忽略空槽，否则补零的槽位会被当成
"一张全黑的视图"参与注意力，悄悄污染特征。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_ROOT))

from data.prepare import ViewConfig, augment, choose_view_count, make_sample  # noqa: E402

SAMPLE_IMG = next(
    (AI_ROOT / "data" / "images" / "sichuan").rglob("*.jpg"), None)


def fake_pano(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)


class TestViewCount(unittest.TestCase):
    def test_single_view_probability_is_honored(self):
        cfg = ViewConfig(single_view_prob=0.5)
        rng = np.random.default_rng(0)
        counts = [choose_view_count(rng, cfg) for _ in range(4000)]
        ones = sum(1 for c in counts if c == 1)
        self.assertAlmostEqual(ones / len(counts), 0.5, delta=0.05)

    def test_count_within_bounds(self):
        cfg = ViewConfig(n_max=8, single_view_prob=0.0)
        rng = np.random.default_rng(1)
        counts = [choose_view_count(rng, cfg) for _ in range(500)]
        self.assertTrue(all(2 <= c <= 8 for c in counts))

    def test_deterministic_given_seed(self):
        cfg = ViewConfig()
        a = [choose_view_count(np.random.default_rng(3), cfg) for _ in range(5)]
        b = [choose_view_count(np.random.default_rng(3), cfg) for _ in range(5)]
        self.assertEqual(a, b)


class TestMakeSample(unittest.TestCase):
    def setUp(self):
        self.pano = fake_pano()
        self.cfg = ViewConfig(size=32, n_max=8)

    def test_shapes_and_mask_agree(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            views, vmask = make_sample(self.pano, rng, self.cfg, augment_on=False)
            self.assertEqual(views.shape, (8, 32, 32, 3))
            self.assertEqual(vmask.shape, (8,))
            n = int(vmask.sum())
            self.assertGreaterEqual(n, 1)
            self.assertTrue(vmask[:n].all() and not vmask[n:].any(),
                            "掩码必须是前 n 个为真、其余为假（与写入口一致）")

    def test_unused_slots_are_zero(self):
        rng = np.random.default_rng(0)
        views, vmask = make_sample(self.pano, rng, self.cfg, augment_on=False)
        n = int(vmask.sum())
        if n < 8:
            self.assertEqual(int(views[n:].max()), 0,
                             "未使用的槽位必须全零，否则会污染池化")

    def test_views_differ_from_each_other(self):
        """同一全景的不同朝向必须切出不同的图，否则视图是白给的。"""
        cfg = ViewConfig(size=32, n_max=4, single_view_prob=0.0)
        rng = np.random.default_rng(2)
        pano = np.zeros((64, 256, 3), np.uint8)
        pano[:, :128] = 255                      # 一半黑一半白，朝向差异可测
        views, vmask = make_sample(pano, rng, cfg, augment_on=False)
        n = int(vmask.sum())
        self.assertGreater(n, 1)
        means = [float(views[i].mean()) for i in range(n)]
        self.assertGreater(max(means) - min(means), 10.0,
                           "不同朝向的视图亮度应当明显不同")

    def test_heading_offset_is_randomized(self):
        """起始朝向必须随机：固定朝向会让模型学到与地点无关的巧合。"""
        cfg = ViewConfig(size=16, n_max=4, single_view_prob=0.0)
        pano = np.zeros((32, 128, 3), np.uint8)
        pano[:, :64] = 255
        firsts = set()
        rng = np.random.default_rng(5)
        for _ in range(30):
            views, vmask = make_sample(pano, rng, cfg, augment_on=False)
            firsts.add(round(float(views[0].mean()), 1))
        self.assertGreater(len(firsts), 2, "首个视图的统计量几乎不变，朝向没随机")

    def test_works_with_real_panorama(self):
        if SAMPLE_IMG is None or not SAMPLE_IMG.exists():
            self.skipTest("尚无已抓取的四川样本")
        import numpy as np
        from PIL import Image
        pano = np.asarray(Image.open(SAMPLE_IMG).convert("RGB"))
        cfg = ViewConfig(size=64, n_max=8)
        views, vmask = make_sample(pano, np.random.default_rng(0), cfg)
        self.assertEqual(views.shape, (8, 64, 64, 3))
        self.assertTrue(vmask.any())
        self.assertGreater(int(views[0].std()), 5, "真实全景切出的视图不应是纯色")


class TestAugment(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.views = rng.integers(0, 256, size=(4, 24, 24, 3), dtype=np.uint8)
        self.cfg = ViewConfig(size=24, n_max=4)

    def test_shape_and_dtype_preserved(self):
        out = augment(self.views.copy(), np.random.default_rng(1), self.cfg)
        self.assertEqual(out.shape, self.views.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_values_stay_in_range(self):
        for seed in range(10):
            out = augment(self.views.copy(), np.random.default_rng(seed), self.cfg)
            self.assertGreaterEqual(int(out.min()), 0)
            self.assertLessEqual(int(out.max()), 255)

    def test_augmentation_actually_changes_the_image(self):
        out = augment(self.views.copy(), np.random.default_rng(1), self.cfg)
        self.assertGreater(float(np.abs(out.astype(int) - self.views.astype(int)).mean()),
                           0.5, "增强没有实际改变图像")

    def test_differs_across_calls(self):
        """每个 epoch 看到的图都该不同。"""
        a = augment(self.views.copy(), np.random.default_rng(1), self.cfg)
        b = augment(self.views.copy(), np.random.default_rng(2), self.cfg)
        self.assertFalse(np.array_equal(a, b))

    def test_no_augmentation_config_is_identity(self):
        cfg = ViewConfig(size=24, brightness=0, contrast=0, saturation=0,
                         channel_gain=0, crop_scale=None, blur_prob=0)
        out = augment(self.views.copy(), np.random.default_rng(1), cfg)
        self.assertTrue(np.array_equal(out, self.views))


if __name__ == "__main__":
    unittest.main(verbosity=2)

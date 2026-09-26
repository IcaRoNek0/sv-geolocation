"""视图切分的回归测试。

重点是对抗 uint8 下溢：`b - a` 在两个 uint8 相减且右邻更暗时会回绕。
用单调渐变测试恰好不会触发它（差值恒为正），所以这里必须用**随机图**
并对拍一个 float64 的参考实现。
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.views import _bilinear, _sample_maps, _wrapped, extract_views  # noqa: E402


def reference_bilinear(src, u, v):
    """逐像素的 float64 参考实现，不做任何批量优化。"""
    h, w, _ = src.shape
    out = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float64)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            uu, vv = float(u[i, j]), float(v[i, j])
            u0, v0 = int(math.floor(uu)), int(math.floor(vv))
            du, dv = uu - u0, vv - v0
            u0 = min(max(u0, 0), w - 2)
            v0 = min(max(v0, 0), h - 2)
            for ch in range(3):
                a = float(src[v0, u0, ch])
                b = float(src[v0, u0 + 1, ch])
                c = float(src[v0 + 1, u0, ch])
                d = float(src[v0 + 1, u0 + 1, ch])
                top = a + (b - a) * du
                bot = c + (d - c) * du
                out[i, j, ch] = top + (bot - top) * dv
    return out


class TestBilinear(unittest.TestCase):
    def test_random_image_matches_reference(self):
        """随机图：相邻像素有升有降，能触发任何方向的下溢。"""
        rng = np.random.default_rng(0)
        pano = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
        src = _wrapped(pano)
        u, v = _sample_maps(48, 48, 90.0, 0.0, 0.0, 128, 64)
        got = _bilinear(src, u, v).astype(np.float64)
        want = reference_bilinear(src, u, v)
        self.assertLess(
            np.abs(got - want).max(), 1.01,
            "双线性采样与 float64 参考实现不符——检查 uint8 下溢",
        )

    def test_checkerboard_does_not_blow_up(self):
        """棋盘图：强制相邻像素在两个方向上剧烈交替。"""
        pano = np.zeros((32, 64, 3), dtype=np.uint8)
        pano[::2, ::2] = 255
        pano[1::2, 1::2] = 255
        src = _wrapped(pano)
        u, v = _sample_maps(32, 32, 90.0, 0.0, 0.0, 64, 32)
        got = _bilinear(src, u, v)
        self.assertTrue(got.dtype == np.uint8)
        self.assertLessEqual(int(got.max()), 255)
        # 任何回绕都会产生与邻近像素无关的孤立极值
        self.assertLess(np.abs(np.diff(got.astype(np.int32), axis=1)).max(), 256)


class TestGeometry(unittest.TestCase):
    def test_center_of_view_matches_heading_longitude(self):
        """朝向 h、俯仰 0 时，视图中心应落在全景的经度 h 处。"""
        w, h = 512, 256
        u, v = _sample_maps(64, 64, 90.0, 90.0, 0.0, w, h)
        # 经度 90° → u = (90/360 + 0.5) * w = 0.75 * 512 = 384，加 1 像素填充偏移
        self.assertAlmostEqual(u[32, 32], 385.0, delta=1.5)
        self.assertAlmostEqual(v[32, 32], 128.0, delta=1.5)

    def test_field_of_view_span(self):
        """90° 视场在 512 宽的全景上应恰好覆盖 128 像素。"""
        u, _ = _sample_maps(64, 64, 90.0, 0.0, 0.0, 512, 256)
        self.assertAlmostEqual(u.max() - u.min(), 512 / 4, delta=2.0)

    def test_full_surround_covers_every_column(self):
        """8 个 90° 视图应覆盖全景的每一列，不留空洞。

        输出尺寸必须不低于源跨度，否则欠采样本来就会漏列——这里跨度是
        512/4 = 128 列，用 192 像素输出（约 1.5 倍过采样）。
        实际训练用 224 像素输出跨 512 列，是降采样，只会更密。
        """
        w, h = 512, 256
        seen = np.zeros(w, dtype=bool)
        for heading in range(0, 360, 45):
            u, _ = _sample_maps(192, 192, 90.0, float(heading), 0.0, w, h)
            cols = np.clip((u - 1.0).astype(np.int32), 0, w - 1).ravel()
            seen[cols] = True
        self.assertTrue(seen.all(), f"有 {int((~seen).sum())} 列未被任何视图覆盖")

    def test_seam_straddling_view_wraps(self):
        """跨接缝的视图必须绕回另一侧，不能把 u 卡在边界。

        朝向约 180° 时 u 会超出源宽一整段；修补前这里会静默取到边缘像素。
        """
        w, h = 512, 256
        u, _ = _sample_maps(64, 64, 90.0, 180.0, 0.0, w, h)
        self.assertLessEqual(u.max(), w + 1.5, "u 上界越过了环绕填充")
        # 该视图应同时覆盖源图右端与左端，而不是被裁剪成单侧
        cols = (u - 1.0).astype(np.int32)
        self.assertGreater(cols.max(), w * 0.95)
        self.assertLess(cols.min(), w * 0.05)

    def test_heading_shift_path_matches_general_path(self):
        """俯仰为 0 时，"u 整体平移"的快速路径必须与一般路径等价。

        这是性能优化的正确性前提，不是近似——绕竖直轴旋转不改变任何射线
        的纬度，所以 v 真的不变，u 真的只是平移。训练时俯仰恒为 0，
        走的就是这条路径。
        """
        from utils.views import _general_maps, _sample_maps
        w, h = 512, 256
        for heading in (0.0, 37.5, 90.0, 180.0, 270.0, 359.9):
            uf, vf = _sample_maps(64, 64, 90.0, heading, 0.0, w, h)
            ug, vg = _general_maps(64, 64, 90.0, heading, 0.0, w, h)
            np.testing.assert_allclose(uf, ug, atol=0.05,
                                       err_msg=f"朝向 {heading}：u 不等价")
            np.testing.assert_allclose(vf, vg, atol=0.01,
                                       err_msg=f"朝向 {heading}：v 不等价")

    def test_heading_shift_yields_identical_pixels(self):
        """两条路径切出的图像必须完全一致。"""
        from utils.views import _bilinear, _general_maps, _sample_maps, _wrapped
        rng = np.random.default_rng(3)
        pano = rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)
        src = _wrapped(pano)
        for heading in (0.0, 45.0, 180.0, 300.0):
            uf, vf = _sample_maps(48, 48, 90.0, heading, 0.0, 256, 128)
            ug, vg = _general_maps(48, 48, 90.0, heading, 0.0, 256, 128)
            a = _bilinear(src, uf, vf)
            b = _bilinear(src, ug, vg)
            diff = np.abs(a.astype(int) - b.astype(int)).max()
            self.assertLessEqual(diff, 1,
                                 f"朝向 {heading}：两条路径切出的像素差 {diff}")

    def test_constant_panorama_gives_constant_view(self):
        """常量图在任何朝向下都应输出常量——几何错误的通用探测器。"""
        pano = np.full((128, 256, 3), 77, dtype=np.uint8)
        for heading in (0.0, 37.0, 180.0, 359.0):
            views = extract_views(pano, [heading], size=32)
            self.assertTrue(
                (views == 77).all(),
                f"朝向 {heading} 的常量图输出不是常量",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

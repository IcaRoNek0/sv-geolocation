"""Regression tests for evaluation truth, screenshot geometry and vehicle priors."""
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.inference_views import load_image, to_views
from data.prepare import ViewConfig, choose_view_count, make_sample, sample_rng
from data.vehicle import build_vehicle_indices, car_view, mask_vehicle_crop
from models.vehicle_prior import coverage_prior, blend_vehicle_prior
from tools.eval_common import truth_coordinates
from tools.build_vehicle_prior import read_coverage
from utils.geo_utils import weighted_geometric_median
from utils.views import extract_perspective


class RegressionTests(unittest.TestCase):
    def test_prediction_is_never_truth(self):
        with self.assertRaises(ValueError):
            truth_coordinates([{"panoid": "a", "lon": 100, "lat": 30}])
        a = truth_coordinates([{"panoid": "a", "lon": 100, "lat": 30}], {"a": (110, 20)})
        np.testing.assert_equal(a, [[110, 20]])

    def test_screenshot_not_guessed_from_16_9(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            Image.new("RGB", (320, 180)).save(p)
            self.assertFalse(load_image(p)[1])
            self.assertFalse(load_image(p, "auto")[1])
            self.assertTrue(load_image(p, "panorama")[1])
            Image.new("RGB", (320, 160)).save(p)
            self.assertFalse(load_image(p)[1])
            self.assertTrue(load_image(p, "auto")[1])

    def test_screenshot_crop_keeps_square_object_square(self):
        a = np.zeros((100, 200, 3), np.uint8)
        a[30:70, 80:120] = 255
        v, mask, n = to_views(a, False, 4, 90, 100, 4)
        y, x = np.where(v[0, :, :, 0] == 255)
        self.assertEqual(x.max()-x.min(), y.max()-y.min())
        self.assertEqual(n, 1)
        self.assertEqual(mask.sum(), 1)

    def test_epoch_randomness_and_repeatability(self):
        a = sample_rng(42, "one", 0).random(10)
        b = sample_rng(42, "one", 1).random(10)
        np.testing.assert_equal(a, sample_rng(42, "one", 0).random(10))
        self.assertFalse(np.array_equal(a, b))

    def test_one_view_config(self):
        cfg = ViewConfig(n_max=1, size=16, single_view_prob=0)
        self.assertEqual(choose_view_count(np.random.default_rng(0), cfg), 1)

    def test_eval_modes_have_fixed_counts(self):
        pano = np.zeros((64, 128, 3), np.uint8)
        cfg = ViewConfig(n_max=4, size=16)
        for mode, count in [("single", 1), ("panorama", 4)]:
            _, mask = make_sample(pano, sample_rng(1, "p"), cfg, False, mode)
            self.assertEqual(mask.sum(), count)

    def test_cached_pitch_maps_match_general_projection(self):
        from utils.views import _sample_maps, _general_maps
        for pitch in (-12, 12, 90):
            for heading in (0, 37, 123, 270):
                u, v = _sample_maps(32, 32, 70, heading, pitch, 1024, 512)
                expected_u, expected_v = _general_maps(32, 32, 70, heading, pitch, 1024, 512)
                np.testing.assert_allclose(v, expected_v, atol=1e-3)
                circular = (u - expected_u + 512) % 1024 - 512
                np.testing.assert_allclose(circular, 0, atol=1e-3)

    def test_perspective_nadir_looks_down(self):
        pano = np.zeros((180, 360, 3), np.uint8)
        pano[120:] = 255
        down = extract_perspective(pano, pitch=90, width=32, height=32, fov_y=60)
        horizon = extract_perspective(pano, pitch=0, width=32, height=32, fov_y=60)
        self.assertGreater(down.mean(), 250)
        self.assertLess(horizon.mean(), 5)

    def test_car_masks_corners_and_single_view(self):
        cfg = ViewConfig(n_max=1, size=32)
        view = car_view(np.full((64, 128, 3), 200, np.uint8), cfg, sample_rng(0, "p"))
        self.assertEqual(view.shape, (1, 32, 32, 3))
        self.assertEqual(view[0, 0, 0].max(), 0)
        self.assertEqual(view[0, 16, 16].min(), 200)

    def test_vehicle_classes_use_train_only(self):
        samples = {
            'a': {'vehicle': 'abc', 'date': '20200101'},
            'b': {'vehicle': 'ABC', 'date': '20200102'},
            'c': {'vehicle': 'unseen', 'date': '20210101'},
        }
        result = build_vehicle_indices(samples, {'a': 'train', 'b': 'train', 'c': 'val_same'}, 2, 2)
        self.assertEqual(result[0], ['ABC'])
        self.assertEqual(result[3]['c'], -1)
        self.assertEqual(result[4]['c'], -1)

    def test_vehicle_prior_missing_year_uses_union(self):
        prior, mass = coverage_prior({'CAR': 1}, {'2025': 1}, {'CAR': {'2020': ['1','2']}}, ['1','2','3'])
        self.assertEqual(prior, {'1': .5, '2': .5, '3': 0})
        self.assertEqual(mass, 1)

    def test_vehicle_fusion_fallback_and_bound(self):
        env = {'1': .8, '2': .2}
        prior = {'1': 0, '2': 1}
        self.assertEqual(blend_vehicle_prior(env, prior, .1, 1), (env, False))
        self.assertEqual(blend_vehicle_prior(env, prior, .9, 0), (env, False))
        fused, applied = blend_vehicle_prior(env, prior, .9, 1)
        self.assertTrue(applied)
        self.assertGreaterEqual(fused['1'], .75 * env['1'])
        self.assertAlmostEqual(sum(fused.values()), 1)

    def test_coincident_mean_is_not_necessarily_median(self):
        # Weighted mean lands exactly at the middle point; the median is the right point.
        lon, lat = weighted_geometric_median([[-2,0],[0,0],[1,0]], [1,.1,2])
        self.assertAlmostEqual(lon, 1, places=4)

    def test_trajectory_parser_matches_existing_json(self):
        base = Path(__file__).resolve().parents[2] / 'svc_trajectory' / 'exports'
        if not (base / '090_0001_7F.json').exists():
            self.skipTest('External trajectory fixture unavailable')
        data = json.loads((base / '090_0001_7F.json').read_text())
        expected = {}
        for seg in data['segments']:
            for r in seg['trajectory']:
                if r['adcode']:
                    expected.setdefault(r['date'][:4], set()).add(r['adcode'])
        actual = read_coverage(base / '090_0001_7F.svtraj')['090_0001_7F']
        self.assertEqual(actual, {k:sorted(v) for k,v in expected.items()})

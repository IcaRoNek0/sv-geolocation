"""Run these on Kaggle (PyTorch required); no downloads or timm weights needed."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from models.env_model import EnvModel
    from models.losses import MultiTaskLoss
    from data.dataset import PanoramaDataset, make_loader
    from data.prepare import ViewConfig

    class TinyBackbone(nn.Module):
        num_features = 8
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
        def forward(self, x):
            return self.conv(x).mean((2, 3))

    def model():
        fake = types.SimpleNamespace(create_model=lambda *a, **kw: TinyBackbone())
        with patch.dict(sys.modules, {'timm': fake}):
            return EnvModel(3, 2, 2, pretrained=False, drop=0)


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable; run on Kaggle')
class TorchTrainingTests(unittest.TestCase):
    def test_old_state_dict_still_loads(self):
        a, b = model(), model()
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))

    def test_padding_cannot_change_prediction(self):
        m = model().eval()
        x = torch.randint(0, 255, (2, 4, 3, 16, 16), dtype=torch.uint8)
        mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
        with torch.no_grad():
            a = m(x, mask)['county']
            x[~mask] = 255 - x[~mask]
            b = m(x, mask)['county']
        torch.testing.assert_close(a, b)

    def test_training_packed_views_match_dense_forward(self):
        m = model().train()
        x = torch.randint(0, 255, (2, 4, 3, 16, 16), dtype=torch.uint8)
        mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
        sizes = []
        hook = m.backbone.register_forward_pre_hook(lambda module, inputs: sizes.append(inputs[0].shape[0]))
        dense = m(x, mask, return_view_logits=True)
        m.pack_views = True
        packed = m(x, mask, return_view_logits=True)
        torch.testing.assert_close(dense['county'], packed['county'])
        torch.testing.assert_close(dense['view_county'][mask], packed['view_county'][mask])
        torch.testing.assert_close(m(x, mask)['county'], packed['county'])
        self.assertEqual(sizes, [8, 3, 3])
        hook.remove()
        self.assertEqual(set(m.state_dict()), set(model().state_dict()))
        loss = packed['county'].sum() + packed['view_county'][mask].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(m.backbone.conv.weight.grad).all())
        self.assertGreater(m.backbone.conv.weight.grad.abs().sum(), 0)

    def test_mixed_loss_supervises_real_views_and_handles_ignored_labels(self):
        m = model().train()
        x = torch.randint(0, 255, (2, 4, 3, 16, 16), dtype=torch.uint8)
        mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
        out = m(x, mask, return_view_logits=True)
        batch = {'county':torch.tensor([0, 1]), 'city':torch.tensor([-1, -1]),
                 'prov':torch.tensor([-1, -1]), 'coord':torch.zeros(2,2), 'vmask':mask}
        loss_fn = MultiTaskLoss(np.eye(3), geo_mix=.15, w_view=.25)
        loss, parts = loss_fn(out, batch)
        fast_loss, fast_parts = loss_fn(out, batch, collect_parts=False)
        torch.testing.assert_close(loss, fast_loss)
        self.assertTrue(parts)
        self.assertEqual(fast_parts, {})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(m.head_county.weight.grad.abs().sum(), 0)

    def test_persistent_workers_observe_new_epoch(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'images').mkdir()
            pixels = np.random.default_rng(1).integers(0,256,(64,128,3),dtype=np.uint8)
            Image.fromarray(pixels).save(root/'images'/'a.jpg')
            ds = PanoramaDataset(root, {'a':'train'}, {'a':{}}, {'a':0}, {'a':(0,0)},
                                 {'a':0}, {'a':0}, view_cfg=ViewConfig(size=16), seed=1)
            loader = make_loader(ds, 1, True, num_workers=1)
            a = next(iter(loader))['views']
            ds.set_epoch(1)
            b = next(iter(loader))['views']
            ds.set_epoch(0)
            c = next(iter(loader))['views']
            self.assertFalse(torch.equal(a, b))
            self.assertTrue(torch.equal(a, c))
            del loader

    def test_incomplete_accumulation_group_gets_update(self):
        # Regression for the actual training update/divisor expressions.
        import ast
        tree = ast.parse((Path(__file__).resolve().parents[1]/'train.py').read_text())
        expressions = {node.targets[0].id: node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in ('update','group_size')}
        env = {'n_batches':5,'accum':2}
        updates, divisors = [], []
        for step in range(5):
            env['step'] = step
            for key, value in expressions.items():
                result = eval(compile(ast.Expression(value), '<training>', 'eval'), {}, env)
                (updates if key=='update' else divisors).append(result)
        self.assertEqual(updates, [False, True, False, True, True])
        self.assertEqual(divisors, [2,2,2,2,1])


if __name__ == '__main__':
    unittest.main()

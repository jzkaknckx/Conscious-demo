import unittest
import numpy as np
import torch
from nns.cnns.features import (MultiScaleFeatureBank, curvature_from_prepooled,
    aspect_from_prepooled_J, resize_axial_orientation)
from nns.memorygraphs import graph_memorypool_onceoptimizer as g


class AdvancedFeatureStabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_and_constant_bank_have_no_structure(self):
        bank = MultiScaleFeatureBank([1, 2, 4])
        for scalar in (0., .5):
            values = bank(torch.full((1, 3, 24, 24), scalar), return_confidence=True)
            for value in values:
                self.assertTrue(torch.isfinite(value).all())
                self.assertLessEqual(float(value.abs().max()), 1e-5)

    def test_rank_one_tensor_is_finite_and_local_nan_does_not_spread(self):
        gen = torch.Generator().manual_seed(29)
        dx, dy = torch.randn(2, 1, 20, 20, generator=gen) * 10, torch.randn(2, 1, 20, 20, generator=gen) * 10
        xx, yy, xy = dx.square(), dy.square(), dx * dy
        xx[0, 0, 2, 2] = float('nan')
        yy[0, 0, 4, 4] = float('inf')
        aspect = aspect_from_prepooled_J(xx, yy, xy)
        self.assertTrue(torch.isfinite(aspect).all())
        self.assertEqual(float(aspect[0, 0, 2, 2]), 0.)
        self.assertEqual(float(aspect[0, 0, 4, 4]), 0.)
        self.assertGreater(int((aspect != 0).sum()), 700)
        self.assertLessEqual(float(aspect.abs().max()), 1.)
        curvature = curvature_from_prepooled(xx, yy, xy)
        self.assertTrue(torch.isfinite(curvature).all())

    def test_axial_wrap_and_undefined_interpolation(self):
        angles = torch.tensor([[[[.99, -.99]]]])
        result, confidence = resize_axial_orientation(angles, torch.ones_like(angles), (1, 3))
        self.assertGreater(float(result[0, 0, 0, 1].abs()), .98)
        self.assertGreater(float(confidence.min()), .99)
        result, confidence = resize_axial_orientation(angles, torch.zeros_like(angles), (1, 3))
        self.assertEqual(float(result.abs().max()), 0.)
        self.assertEqual(float(confidence.max()), 0.)

    def test_confidence_contract_gate_metric_and_tangent(self):
        cfg = g.configure_orientation_contract(g.MemoryConfig(), 2)
        cfg.device = torch.device('cpu'); cfg.H = cfg.W = 12
        cfg.once_bind_edge_attributes = False
        angles = torch.zeros(1, 2, 12, 12)
        named = {'orientation': angles, 'orientation_confidence': torch.zeros_like(angles)}
        inputs = g.prepare_feature_subspaces(named, cfg)
        self.assertFalse(g.SimilarityEngine.compute_gate_map(inputs[4], cfg, 4).any())
        named['orientation_confidence'].fill_(1.)
        inputs = g.prepare_feature_subspaces(named, cfg)
        self.assertTrue(g.SimilarityEngine.compute_gate_map(inputs[4], cfg, 4).all())
        gi, gii = g.GmemoryI(), g.GmemoryII(cfg)
        report = g.OnceGraphBuilder(cfg).build_once(inputs, gi, gii)
        self.assertTrue(gi.nodes)
        self.assertTrue(np.allclose(report.supports[4].tangent[5, 5], [0., 1.]))
        for node in gi.nodes.values():
            self.assertTrue(torch.equal(node.mask[:2], torch.ones_like(node.mask[:2])))
            self.assertTrue(torch.equal(node.mask[2:], torch.zeros_like(node.mask[2:])))
        node = next(iter(gi.nodes.values()))
        altered = inputs[4].clone(); altered[:, 2:] = .2
        a = g.SimilarityEngine.calculate_similarity(inputs[4], node.prototype.view(1, -1, 1, 1), cfg, 4, mask_W=node.mask)
        b = g.SimilarityEngine.calculate_similarity(altered, node.prototype.view(1, -1, 1, 1), cfg, 4, mask_W=node.mask)
        torch.testing.assert_close(a, b)
        coord = g.MultilevelCoordinator(cfg)
        state = coord.state_dict()
        state['feature_contract'][4].pop('encoding_version')
        with self.assertRaises(ValueError):
            coord.load_state_dict(state)
        with self.assertRaises(ValueError):
            g.prepare_feature_subspaces({4: angles}, cfg)

    def test_confident_bank_accepts_small_images(self):
        values = MultiScaleFeatureBank([1, 2, 4])(torch.zeros(1, 3, 1, 2), return_confidence=True)
        self.assertTrue(all(tuple(x.shape) == (1, 3, 1, 2) for x in values))
        self.assertTrue(all(torch.isfinite(x).all() for x in values))


if __name__ == '__main__':
    unittest.main()

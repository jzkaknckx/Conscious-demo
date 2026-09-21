import unittest
import numpy as np
import torch
from nns.memorygraphs import graph_memorypool_onceoptimizer as g


class EdgeAttributeBindingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = g.MemoryConfig(); self.cfg.device = torch.device('cpu')
        self.cfg.once_bind_edge_attributes = True  # explicit legacy comparison, not the new default
        self.cfg.H = self.cfg.W = 24
        self.cfg.once_region_budget = 24
        strength = torch.zeros(1, 1, 24, 24)
        strength[..., 2:22, 6] = 1.; strength[..., 2:22, 17] = 1.
        grad = torch.cat((strength, torch.zeros_like(strength), strength, torch.zeros_like(strength)), 1)
        # Deliberately alternate appearance sharply along each edge. Attributes
        # must NOT split these edges using their own appearance similarity.
        aspect = torch.ones_like(strength) * .8; aspect[..., ::2, :] *= -1
        self.features = {0: grad, 3: aspect, 4: torch.zeros_like(strength)}

    def build(self, features=None):
        gi, gii = g.GmemoryI(), g.GmemoryII(self.cfg)
        report = g.OnceGraphBuilder(self.cfg).build_once(self.features if features is None else features, gi, gii)
        return report, gi

    def test_shares_partition_and_anchor_despite_appearance_changes(self):
        report, gi = self.build()
        parents = {r.region_id: r for r in report.regions if r.modality_id == 0}
        self.assertEqual(len(parents), 2)
        for mid in (3, 4):
            children = [r for r in report.regions if r.modality_id == mid]
            self.assertEqual(len(children), len(parents))
            for r in children:
                parent = parents[r.parent_region_id]
                np.testing.assert_array_equal(r.pixels, parent.pixels)
                self.assertEqual(r.adjacency, parent.adjacency)
                self.assertEqual(r.anchor, parent.anchor)
                self.assertEqual(r.support_dim, parent.support_dim)
        self.assertEqual({n.modality_id for n in gi.nodes.values()}, {0, 3, 4})
        # Features remain modality-specific, not replaced with grad values.
        self.assertTrue(any(float(n.prototype[0]) < 0 for n in gi.nodes.values() if n.modality_id == 3))

    def test_validity_holes_do_not_create_new_regions_or_invalid_samples(self):
        self.features[3][..., 10:14, 6] = float('nan')
        report, _ = self.build()
        children = [r for r in report.regions if r.modality_id == 3]
        self.assertEqual(len(children), 2)
        self.assertTrue(any('edge_attribute_validity_holes' in r.reasons for r in children))
        for r in children:
            self.assertTrue(all(report.supports[3].writable.flat[p] for p in r.samples))
        self.assertTrue((report.labels[3][10:14, 6] == -1).all())

    def test_missing_grad_and_total_budget(self):
        report, gi = self.build({3: self.features[3], 4: self.features[4]})
        self.assertFalse(report.regions)
        self.assertFalse(gi.nodes)
        self.assertEqual(report.rejected['edge_attribute_missing_grad'], 2)
        for budget in (0, 1, 2, 3, 5):
            self.cfg.once_region_budget = budget
            report, _ = self.build()
            self.assertLessEqual(len(report.regions), budget)
            ids = {r.region_id for r in report.regions}
            self.assertEqual(len(ids), len(report.regions))
            for r in report.regions:
                if r.parent_region_id is not None:
                    self.assertIn(r.parent_region_id, ids)


if __name__ == '__main__':
    unittest.main()

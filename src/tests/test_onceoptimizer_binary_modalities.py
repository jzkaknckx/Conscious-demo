"""Conservative filtering must preserve the fine verifier's accepted candidates."""
import copy
import math
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from nns.memorygraphs import graph_memorypool_onceoptimizer as g


class BinaryModalitiesTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = g.MemoryConfig()
        self.cfg.device = torch.device('cpu')
        self.cfg.H = self.cfg.W = 16

    def index(self, mid, prototypes, image):
        gi, gii = g.GmemoryI(), g.GmemoryII(self.cfg)
        for prototype in prototypes:
            gi.add_node(mid, torch.tensor(prototype, dtype=torch.float32))
        provider = g.FeatureResponseCache({mid: image}, gi, self.cfg)
        return g.BinaryCoarseIndex(provider, gii), gi

    def test_rgb_bins_keep_boundary_and_reject_far_colour(self):
        x = torch.full((1, 3, 16, 16), .125 - 1e-6)
        index, gi = self.index(1, [[.125 + 1e-6] * 3, [.95] * 3], x)
        bits = index.query(SimpleNamespace(modality_id=1, pixels=list(range(256))))
        self.assertTrue(bits & (1 << index.node_bits[0]))
        self.assertFalse(bits & (1 << index.node_bits[1]))
        index.views[0] = (None, 3, {0: .8, 1: .2})
        self.assertTrue(index.keep(0, 1))
        self.assertFalse(index.keep(0, 2))
        # Unknown/partial evidence remains possible, not hard full-bit AND.
        index.views[1] = (None, 3, {0: .5, 1: .5})
        self.assertTrue(index.keep(1, 1))

    def test_coarse_upper_covers_real_responses_all_modalities(self):
        rng = torch.Generator().manual_seed(18)
        for mid, channels in [(0, 4), (1, 3), (2, 3), (3, 3), (4, 3)]:
            x = torch.rand(1, channels, 16, 16, generator=rng)
            if mid in (3, 4):
                x = (x - .5) * (math.pi if mid == 4 else 2)
            prototypes = [x[0, :, y, z].tolist() for y, z in [(1, 1), (5, 8), (12, 13)]]
            index, gi = self.index(mid, prototypes, x)
            pixels = list(range(50, 200))
            bits = index.query(SimpleNamespace(modality_id=mid, pixels=pixels))
            for nid, node in gi.nodes.items():
                response = g.SimilarityEngine.calculate_similarity(
                    x, node.prototype.view(1, -1, 1, 1), self.cfg, mid,
                    mask_W=node.mask.view(1, -1))
                if response.flatten()[pixels].max() >= self.cfg.graph_recall_threshold:
                    self.assertTrue(bits & (1 << index.node_bits[nid]), (mid, nid))
        # Circular wrap-around must not become a bin-boundary false negative.
        x = torch.full((1, 1, 16, 16), -math.pi / 2 + .001)
        index, _ = self.index(4, [[math.pi / 2 - .001]], x)
        self.assertTrue(index.query(SimpleNamespace(modality_id=4, pixels=[0])) & 1)

    def test_interval_bound_over_random_narrow_queries(self):
        rng = np.random.default_rng(1818)
        for mid in (1, 2, 3, 4):
            x = torch.zeros(1, 3, 16, 16)
            index, _ = self.index(mid, [[0., 0., 0.]], x)
            for _ in range(30):
                points = rng.uniform(-1.5, 1.5, (1, 3, 1, 1)) + rng.uniform(0, .15, (1, 3, 3, 4))
                proto = rng.uniform(-1.5, 1.5, (1, 3))
                width = self.cfg.graph_coarse_bin_width[mid]
                lo = np.floor(proto / width) * width
                mask = rng.integers(0, 2, (1, 3)).astype(float)
                upper = index._upper(mid, lo, lo + width, mask,
                                     points.min((0, 2, 3)), points.max((0, 2, 3)))[0]
                actual = g.SimilarityEngine.calculate_similarity(
                    torch.tensor(points, dtype=torch.float32),
                    torch.tensor(proto, dtype=torch.float32).view(1, 3, 1, 1), self.cfg, mid,
                    mask_W=torch.tensor(mask, dtype=torch.float32)).max().item()
                self.assertGreaterEqual(upper + self.cfg.graph_coarse_tolerance, actual)

    def test_alias_units_and_all_modality_writing(self):
        x = torch.ones(1, 1, 16, 16)
        named = {'curv': x * .5, 'asp': x * .5, 'ori': x * .2}
        prepared = g.prepare_feature_subspaces(named, self.cfg)
        self.assertEqual(set(prepared), {2, 3, 4})
        torch.testing.assert_close(prepared[4], named['ori'] * (math.pi / 2))
        torch.testing.assert_close(g.prepare_feature_subspaces({4: prepared[4]}, self.cfg)[4], prepared[4])
        # No amplitude heuristic: NaN or a value just beyond one must not change units.
        named['ori'] = x * 1.1
        torch.testing.assert_close(g.prepare_feature_subspaces(named, self.cfg)[4], x * 1.1 * math.pi / 2)
        features = {0: torch.cat([x, x * 0, x, x * 0], 1),
                    1: x.expand(-1, 3, -1, -1) * .5,
                    2: x * .5, 3: x * .5, 4: x * .2}
        cfg = copy.deepcopy(self.cfg)
        cfg.once_thin_gradient_ridges = False
        cfg.once_surface_barrier = 2.  # synthetic constant gradient must not cut every surface edge
        cfg.once_region_budget = 30
        cfg.once_samples_per_region = 4
        gi, gii = g.GmemoryI(), g.GmemoryII(cfg)
        report = g.OnceGraphBuilder(cfg).build_once(features, gi, gii)
        self.assertEqual(set(report.supports), set(range(5)))
        self.assertEqual({n.modality_id for n in gi.nodes.values()}, set(range(5)))

    def test_learning_decisions_unchanged_by_coarse_filter(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.once_region_budget = 8
        cfg.once_samples_per_region = 4
        cfg.once_surface_cover_radius = 32
        coord = g.MultilevelCoordinator(cfg)
        features = {1: torch.full((1, 3, 16, 16), .15)}
        coord.learn_view(features, episode_id='first')
        enabled, disabled = copy.deepcopy(coord), copy.deepcopy(coord)
        disabled.cfg.graph_binary_coarse = False
        # deepcopy preserves shared config references in the controller.
        for value in (.15, .9):
            a, b = copy.deepcopy(enabled), copy.deepcopy(disabled)
            image = {1: torch.full((1, 3, 16, 16), value)}
            ra = a.learn_view(image, episode_id='second')
            rb = b.learn_view(image, episode_id='second')
            for key in ('new_regions', 'updated_regions', 'ambiguous_regions', 'new_entities'):
                self.assertEqual(ra.diagnostics[key], rb.diagnostics[key], key)
            if value == .9:
                self.assertGreater(ra.diagnostics['work_counts'].get('coarse_templates_rejected', 0), 0)


if __name__ == '__main__':
    unittest.main()

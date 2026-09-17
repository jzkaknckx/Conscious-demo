"""Correctness guards for bounded shared responses and necessary-condition filtering.

Run: PYTHONPATH=src python -m unittest discover -s src/tests -p test_onceoptimizer_performance.py
"""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from nns.memorygraphs import graph_memorypool_onceoptimizer as g


class ResponseOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = g.MemoryConfig()
        self.cfg.device = torch.device('cpu')
        self.cfg.H = self.cfg.W = 12
        self.cfg.graph_query_chunk = 19  # exercise multiple chunks
        self.inputs = {1: torch.rand((1, 3, 12, 12), generator=torch.Generator().manual_seed(8))}
        self.gi = g.GmemoryI()
        for i in range(3):
            self.gi.add_node(1, self.inputs[1][0, :, i + 2, i + 3].clone())

    def test_region_masks_share_base_without_leaking(self):
        root = g.FeatureResponseCache(self.inputs, self.gi, self.cfg)
        for column in (6, 9, 6):
            mask = np.zeros((12, 12), bool); mask[:, :column] = True
            local = root.restricted(mask)
            independent = g.FeatureResponseCache(self.inputs, self.gi, self.cfg, mask)
            for nid in self.gi.nodes:
                torch.testing.assert_close(local.get(nid), independent.get(nid), rtol=0, atol=0)
                torch.testing.assert_close(local.evaluable(nid), independent.evaluable(nid))
        self.assertEqual(root.map_computations, 3)
        self.assertLessEqual(root.cache_bytes, self.cfg.graph_shared_response_cache_bytes)
        # Masked entries never reach the shared cache.
        fresh = g.FeatureResponseCache(self.inputs, self.gi, self.cfg)
        torch.testing.assert_close(root.get(0), fresh.get(0), rtol=0, atol=0)

    def test_budget_eviction_does_not_change_values(self):
        self.cfg.graph_shared_response_cache_bytes = 12 * 12 * 4
        root = g.FeatureResponseCache(self.inputs, self.gi, self.cfg)
        local = root.restricted(np.ones((12, 12), bool))
        for nid in self.gi.nodes:
            local.get(nid)
            self.assertLessEqual(root.cache_bytes, self.cfg.graph_shared_response_cache_bytes)
        reference = g.FeatureResponseCache(self.inputs, self.gi, self.cfg)
        torch.testing.assert_close(root.get(0), reference.get(0), rtol=0, atol=0)
        self.assertGreater(root.stats['response_cache_evictions'], 0)

    def test_missing_channels_and_nonfinite_masks(self):
        self.gi.nodes[0].mask.zero_()
        values = self.inputs[1].clone(); values[:, :, 3, 3] = float('nan')
        root = g.FeatureResponseCache({1: values}, self.gi, self.cfg)
        valid = np.ones((12, 12)); valid[0, 0] = np.nan
        local = root.restricted(valid)
        self.assertIsNone(local.get(0))
        self.assertFalse(local.evaluable(1)[0, 0, 3, 3])
        self.assertFalse(local.evaluable(1)[0, 0, 0, 0])
        missing = g.FeatureResponseCache({}, self.gi, self.cfg)
        self.assertIsNone(missing.evaluable(1))

    def test_prefilter_keeps_every_accepted_match_and_assignments(self):
        slots = [{'key': i, 'node_id': i, 'xy': xy, 'group': 0, 'weight': 1.,
                  'group_weight': 1., 'anchor': i == 0}
                 for i, xy in enumerate([(0., 0.), (2., 0.), (0., 2.)])]
        constraints = [{'source': 0, 'target': i, 'delta': xy} for i, xy in [(1, (2., 0.)), (2, (0., 2.))]]
        view = g.StructureView(7, 2, slots, constraints)
        centers = [(x, y) for y in range(-1, 13) for x in range(-1, 13)]
        matcher = g.SpatialStructureMatcher(self.cfg)
        for level in (.2, self.cfg.graph_learn_threshold, .99):
            maps = {i: torch.full((1, 1, 12, 12), level) for i in range(3)}
            # Holes / borders challenge evaluable coverage; different nodes share a modality.
            mask = torch.ones((12, 12)); mask[4:6, 4:6] = 0
            nodes = {i: SimpleNamespace(modality_id=1) for i in range(3)}
            provider = g._StoredMapProvider(maps, nodes=nodes, valid_mask=mask)
            full = matcher.evaluate(view, provider, centers)
            pruned = matcher.evaluate(view, provider, centers, min_score=self.cfg.graph_learn_threshold)
            def accepted(rows):
                return [(m.summary(), m.assignments) for m in rows
                        if m.accepted and m.score >= self.cfg.graph_learn_threshold]
            self.assertEqual(accepted(full), accepted(pruned))
            if level == .2:
                self.assertEqual(pruned, [])
            if level == .99:
                self.assertTrue(accepted(full))

    def test_local_pixel_scoring_preserves_fractional_interpolation(self):
        self.cfg.graph_local_response_fraction = 1.
        points = np.array([[5.25, 5.75], [6., 6.], [6.7, 5.1]])
        mask = np.ones((12, 12), bool); mask[6, 6] = False
        local = g.FeatureResponseCache(self.inputs, self.gi, self.cfg).restricted(mask)
        full = g.FeatureResponseCache(self.inputs, self.gi, self.cfg, mask)
        sampled = local.sample(1, points)
        reference = g.SpatialStructureMatcher._sample(full.get(1), points)
        torch.testing.assert_close(sampled, reference, atol=0, rtol=0)
        self.assertEqual(local.stats['local_response_rectangles'], 1)
        self.assertLess(local.stats['local_response_pixels'], 144)

    def test_progressive_matches_reference_with_holes_and_missing_slots(self):
        features = {1: torch.full((1, 3, 12, 12), .5)}
        for node in self.gi.nodes.values():
            node.prototype.fill_(.5)
        slots = [{'key': i, 'node_id': i, 'xy': xy, 'group': 0, 'weight': i + 1.,
                  'group_weight': 1., 'anchor': i == 0}
                 for i, xy in enumerate([(0., 0.), (2., 0.), (0., 2.)])]
        view = g.StructureView(1, 2, slots, [])
        centers = [(x, y) for y in range(12) for x in range(12)]
        mask = np.ones((12, 12), bool); mask[4:6, 4:6] = False
        matcher = g.SpatialStructureMatcher(self.cfg)
        self.cfg.graph_slot_batch = 1  # exercise intermediate rather than only final bounds
        def accepted(rows):
            return [(m.summary(), m.assignments, m.member_positions) for m in rows
                    if m.accepted and m.score >= self.cfg.graph_learn_threshold]
        for missing in (False, True):
            if missing:
                self.gi.nodes[0].mask.zero_()
            self.cfg.graph_progressive_matching = False
            before = matcher.evaluate(view, g.FeatureResponseCache(features, self.gi, self.cfg, mask),
                                      centers, min_score=self.cfg.graph_learn_threshold)
            self.cfg.graph_progressive_matching = True
            after = matcher.evaluate(view, g.FeatureResponseCache(features, self.gi, self.cfg, mask),
                                     centers, min_score=self.cfg.graph_learn_threshold)
            self.assertTrue(accepted(before))
            self.assertEqual(accepted(before), accepted(after))

    def test_early_bounds_skip_later_slot_scoring(self):
        self.cfg.graph_slot_batch = 1
        self.inputs[1].fill_(.1)
        for node in self.gi.nodes.values():
            node.prototype.fill_(.9)
        slots = [{'key': i, 'node_id': i, 'xy': (float(i), 0.), 'group': 0,
                  'weight': 1., 'group_weight': 1., 'anchor': i == 0} for i in range(3)]
        provider = g.FeatureResponseCache(self.inputs, self.gi, self.cfg)
        rows = g.SpatialStructureMatcher(self.cfg).evaluate(
            g.StructureView(0, 2, slots, []), provider, [(4., 4.)], min_score=.78)
        self.assertEqual(rows, [])
        self.assertEqual(provider.stats['upper_bound_rejected'], 1)
        self.assertLess(provider.stats['progressive_sampled_slot_centers'], 3)

    def test_budget_aborts_before_commit_and_can_retry(self):
        import pickle
        cfg = self.cfg
        cfg.H = cfg.W = 32
        coord = g.MultilevelCoordinator(cfg)
        features = {1: torch.full((1, 3, 32, 32), .5)}
        before = pickle.dumps(coord.state_dict())
        result = coord.learn_view(features, episode_id='episode', search_budget=g.SearchBudget(max_seconds=0))
        self.assertEqual(result.diagnostics['learning_status'], 'UNRESOLVED')
        self.assertFalse(result.diagnostics['observation_committed'])
        self.assertEqual(pickle.dumps(coord.state_dict()), before)
        committed = coord.learn_view(features, episode_id='episode')
        self.assertTrue(committed.diagnostics['observation_committed'])
        self.assertTrue(coord.gmem_i.nodes)
        before = pickle.dumps(coord.state_dict())
        result = coord.learn_view(features, episode_id='episode', search_budget=g.SearchBudget(max_template_pairs=0))
        self.assertEqual(result.diagnostics['learning_status'], 'UNRESOLVED')
        # Compare values; tensor pickle storage identifiers need not be stable.
        self.assertEqual(coord.gmem_iii.observations, 1)
        self.assertTrue(all(node.evidence.support == 1 for node in coord.gmem_ii.semantic_nodes.values()))



if __name__ == '__main__':
    unittest.main()

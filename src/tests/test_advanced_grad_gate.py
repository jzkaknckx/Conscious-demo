"""Run manually: PYTHONPATH=src python -m unittest discover -s src/tests -p test_advanced_grad_gate.py"""
import unittest
import numpy as np
import torch
from nns.memorygraphs import graph_memorypool_onceoptimizer as g


class AdvancedGradGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = g.MemoryConfig()
        self.cfg.device = torch.device('cpu')
        self.cfg.once_bind_edge_attributes = False
        self.cfg.once_advanced_partition_mode = 'grad_gated'
        self.valid = np.ones((8, 8), bool)
        self.grad = torch.zeros(1, 4, 8, 8)
        self.grad[:, 0] = 1
        self.grad[:, 2] = 1

    def supports(self, grad=True, threshold=None):
        self.cfg.once_advanced_grad_continuity = threshold
        inputs = {2: torch.ones(1, 1, 8, 8), 3: torch.ones(1, 1, 8, 8),
                  4: torch.zeros(1, 1, 8, 8)}
        if grad:
            inputs[0] = self.grad
        report = g.BuildReport(1, (8, 8), self.valid)
        return g.FeatureSupportBuilder(self.cfg).build(inputs, self.valid, report)

    def test_missing_grad_blocks_all_advanced_modalities(self):
        for mid, support in self.supports(grad=False).items():
            self.assertFalse(support.writable.any())
            self.assertFalse(support.support.any())
            self.assertFalse(support.seeds)

    def test_strict_threshold_and_nonfinite_grad(self):
        supports = self.supports(threshold=1.)
        for mid in (2, 3, 4):
            self.assertFalse(supports[mid].writable.any())
        self.grad[:, :, 4, 4] = float('nan')
        supports = self.supports()
        for mid in (2, 3, 4):
            self.assertFalse(supports[mid].writable[4, 4])
            self.assertFalse((supports[mid].writable & ~supports[mid].gate_context['grad_eligible'].astype(bool)).any())

    def test_own_curvature_similarity_still_splits_edges(self):
        support = self.supports()[2]
        # Isolate the own-modality affinity decision from seed selection and orientation estimation.
        support.values.fill_(.2)
        support.values[:, :, :, 4:] = .9
        support.support[:] = False
        support.support[4, 1:7] = True
        support.quality[:] = 1
        support.directional[:] = support.support
        support.tangent[:] = (1., 0.)
        report = g.BuildReport(1, (8, 8), self.valid)
        g.LocalAffinityBuilder(self.cfg).build(support, np.zeros((8, 8)), report)
        self.assertGreater(len(support.edges), 0)
        self.assertTrue(all((a % 8 < 4) == (b % 8 < 4) for a, b in support.edges))

    def test_new_checkpoint_contract_rejects_old_partition(self):
        model = g.MultilevelCoordinator(self.cfg)
        state = model.state_dict()
        state['segmentation_contract'] = {'edge_attributes': 'grad'}
        with self.assertRaises(ValueError):
            model.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()

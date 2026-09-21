import unittest
from types import SimpleNamespace
import numpy as np
import torch
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig, HypergraphIndex, Region, BuildReport
from nns.memorygraphs.supervised_data import SupervisedConfig
from nns.memorygraphs.supervised_graph_learning import ObjectRegionSelector, region_family


class RecallCorrectnessTests(unittest.TestCase):
    def test_pose_budget_preserves_distinct_templates(self):
        cfg=MemoryConfig();cfg.graph_candidate_budget=2;cfg.graph_poses_per_node=8
        ranked={0:[((1.,99,99.),(0.,0.)),((1.,98,98.),(20.,20.))],
                1:[((.5,1,1.),(10.,10.))]}
        selected,dropped=HypergraphIndex._limit(ranked,cfg)
        self.assertEqual(set(selected),{0,1})
        self.assertEqual(dropped,1)

    def test_votes_use_all_independent_role_centers(self):
        cfg=MemoryConfig();cfg.graph_pose_bin=10
        events=[(0,(0.,0.),1.),(1,(2.,0.),1.)]
        ranked,_=HypergraphIndex._votes(events,{0:[(9,'a',(0.,0.))],1:[(9,'b',(0.,0.))]},cfg)
        self.assertEqual(ranked[9][0][1],(1.,0.))

    def test_refined_children_share_family_and_require_parent(self):
        points=np.array([5,6,7]);parent=Region(0,0,1,points,points,[],anchor=6,semantic_id=0,completed=True)
        child=Region(1,4,1,points,points,[],anchor=7,semantic_id=1,completed=True,support_parent_region_id=0)
        report=BuildReport(1,(4,4),np.ones((4,4),bool));report.regions=[parent,child]
        obs=SimpleNamespace(valid_mask=torch.ones(4,4,dtype=torch.bool),foreground_probability=None,ambiguous_ownership_mask=None)
        sem=SimpleNamespace(related_node_ids=lambda:[0,1])
        pool=SimpleNamespace(semantic_nodes={0:sem,1:sem})
        selector=ObjectRegionSelector(SupervisedConfig())
        selected,families,diag=selector.select(report,obs,pool)
        self.assertEqual(len(selected),2);self.assertEqual(families,{0:0,1:0})
        self.assertEqual(region_family(child),0)
        parent.budget_truncated=True
        selected,_,diag=selector.select(report,obs,pool)
        self.assertEqual(selected,[]);self.assertEqual(diag['rejected'][1],'missing_grad_parent')

    def test_incomplete_geometry_is_penalized_not_silently_complete(self):
        points=np.array([5,6,7]);r=Region(0,0,1,points,points,[],anchor=6,semantic_id=0,
                                        reasons=['curve_order_unresolved'])
        report=BuildReport(1,(4,4),np.ones((4,4),bool));report.regions=[r]
        obs=SimpleNamespace(valid_mask=torch.ones(4,4,dtype=torch.bool),foreground_probability=None,ambiguous_ownership_mask=None)
        pool=SimpleNamespace(semantic_nodes={0:SimpleNamespace(related_node_ids=lambda:[0,1])})
        selected,_,diag=ObjectRegionSelector(SupervisedConfig()).select(report,obs,pool)
        self.assertEqual(len(selected),1);self.assertLess(diag['quality'][0],1.)
        self.assertFalse(diag['quality_components'][0]['completed'])


if __name__=='__main__':unittest.main()

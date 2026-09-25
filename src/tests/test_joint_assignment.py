import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig, SpatialStructureMatcher


class JointAssignmentTests(unittest.TestCase):
    def setup_case(self, second=.8, shared=False):
        cfg=MemoryConfig();cfg.device=torch.device('cpu');cfg.graph_geometry_radius=1
        cfg.graph_geometry_sigma=2.;cfg.graph_recall_threshold=.5;cfg.graph_verify_threshold=.7
        cfg.graph_evaluable_min=.8;cfg.graph_coverage_min=.8
        response=torch.zeros(1,1,5,6);response[0,0,2,2]=1.;response[0,0,2,3]=second
        provider=NS(nodes={0:NS(modality_id=0),1:NS(modality_id=0)},
            get=lambda nid:response,evaluable=lambda nid:torch.ones_like(response,dtype=torch.bool))
        slots=[dict(key=i,node_id=i,group=0,group_weight=1.,weight=1.,anchor=i==0,
                    xy=np.array([0. if i==0 or shared else 1.,0.])) for i in range(2)]
        view=NS(slots=slots,constraints=[],template_id=1,level=2,version=0)
        return cfg,provider,view

    def test_collision_uses_second_peak_without_relaxing_threshold(self):
        cfg,p,v=self.setup_case()
        cfg.graph_joint_assignment=False
        old=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertEqual(old.rejection_reasons,['evidence_conflict'])
        cfg.graph_joint_assignment=True
        new=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertTrue(new.accepted)
        self.assertEqual(new.assignment_diagnostics['status'],'reassigned')
        self.assertNotEqual(new.assignments[0]['point'],new.assignments[1]['point'])
        self.assertLess(new.score,old.score)

    def test_single_peak_cannot_explain_distinct_roles(self):
        cfg,p,v=self.setup_case(second=0.)
        match=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertFalse(match.accepted)
        self.assertIn('evidence_conflict',match.rejection_reasons)
        self.assertEqual(match.assignment_diagnostics['status'],'no_bounded_solution')

    def test_shared_expected_role_preserves_original_semantics(self):
        cfg,p,v=self.setup_case(shared=True)
        match=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertTrue(match.accepted)
        self.assertFalse(match.assignment_diagnostics)

    def test_slot_budget_retains_unresolved_conflict(self):
        cfg,p,v=self.setup_case();cfg.graph_assignment_max_slots=1
        match=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertFalse(match.accepted)
        self.assertEqual(match.assignment_diagnostics['status'],'slot_budget_exceeded')

    def test_independent_components_use_separate_slot_budgets(self):
        cfg,p,v=self.setup_case();cfg.graph_assignment_max_slots=2
        response=torch.zeros(1,1,5,10)
        response[0,0,2,2]=response[0,0,2,6]=1.
        response[0,0,2,3]=response[0,0,2,7]=.8
        p.get=lambda nid:response
        p.evaluable=lambda nid:torch.ones_like(response,dtype=torch.bool)
        for i,xy in ((2,[4.,0.]),(3,[5.,0.])):
            p.nodes[i]=NS(modality_id=0)
            v.slots.append(dict(v.slots[1],key=i,node_id=i,xy=np.array(xy)))
        match=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertTrue(match.accepted)
        self.assertEqual(match.assignment_diagnostics['component_sizes'],[2,2])

    def test_reassignment_still_checks_geometry(self):
        cfg,p,v=self.setup_case()
        v.constraints=[{'source':0,'target':1,'delta':np.array([20.,0.])}]
        match=SpatialStructureMatcher(cfg).evaluate(v,p,[(2.,2.)])[0]
        self.assertFalse(match.accepted)

if __name__=='__main__':unittest.main()

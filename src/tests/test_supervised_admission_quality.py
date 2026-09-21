import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from nns.memorygraphs import graph_memorypool_onceoptimizer as g
from nns.memorygraphs.supervised_experiment import SupervisedExperiment


class AdmissionQualityTests(unittest.TestCase):
    def test_t_junction_paths_preserve_edges_without_false_bridge(self):
        points=np.array([1,4,5,6,9])
        r=g.Region(0,0,1,points,points,[])
        r.adjacency={5:[(q,1.) for q in (1,4,6,9)],**{q:[(5,1.)] for q in (1,4,6,9)}}
        segments=g.RegionSampler._curve_segments(r)
        self.assertEqual(len(segments),4)
        self.assertTrue(all(len(path)==2 and not closed for path,closed in segments))
        edges=[tuple(sorted(pair)) for path,_ in segments for pair in zip(path,path[1:])]
        self.assertEqual(set(edges),{(1,5),(4,5),(5,6),(5,9)})
        self.assertEqual(len(edges),len(set(edges)))

    def test_cycle_and_isolated_pixel_survive_path_decomposition(self):
        points=np.array([0,1,2,3,9]);r=g.Region(0,0,1,points,points,[])
        r.adjacency={0:[(1,1.),(3,1.)],1:[(0,1.),(2,1.)],2:[(1,1.),(3,1.)],3:[(0,1.),(2,1.)]}
        segments=g.RegionSampler._curve_segments(r)
        self.assertEqual(sum(closed for _,closed in segments),1)
        self.assertIn(([9],False),segments)
        self.assertEqual(set(p for path,_ in segments for p in path),set(points))

    def test_family_weight_is_quality_normalized_not_fragment_count(self):
        evidence=NS(weight=lambda cfg:1.)
        def member(q,root=False):return dict(sem_id=1,dx=0.,dy=0.,evidence=evidence,family_id=4,reliability=q,is_family_root=root)
        entity=NS(component_edges={0:member(.5,True),1:member(1.)},relation_edges={},node_id=0,version=1)
        view=NS(slots=[{'key':0,'xy':(0.,0.)}],constraints=[])
        with patch.object(g,'region_view',return_value=view):
            before=g.entity_view(entity,NS(semantic_nodes={1:None}),g.MemoryConfig())
            entity.component_edges[2]=member(1.)
            after=g.entity_view(entity,NS(semantic_nodes={1:None}),g.MemoryConfig())
        self.assertAlmostEqual(sum(x['group_weight'] for x in before.slots),.5)
        self.assertAlmostEqual(sum(x['group_weight'] for x in after.slots),.5)
        self.assertLess(after.slots[0]['group_weight'],after.slots[1]['group_weight'])

    def test_training_audit_separates_not_committed_and_target_recall(self):
        exp=SupervisedExperiment.__new__(SupervisedExperiment)
        exp.image_window=lambda *args:['image']
        a=NS(image_id='image',object_id='0',class_id='cat');o=NS(ledger_key='committed',valid_mask=None)
        b=NS(image_id='image',object_id='1',class_id='cat');ob=NS(ledger_key='missing',valid_mask=None)
        exp._objects=lambda *args:iter([(a,o),(b,ob)])
        model=NS(gmem_iii=NS(entity_nodes={3:NS(supervision={'label_id':'cat'})}),
                 query_hierarchy=lambda *args,**kwargs:NS(entities=[NS(template_id=3)],diagnostics={}))
        exp.learner=NS(graph_version=2,ledger={'committed':{'entity_id':3}},model=model)
        exp.encoder=lambda observation:{}
        result=exp.audit_memory_recall()
        self.assertEqual(result['status_counts'],{'TARGET_HIT':1,'NOT_COMMITTED':1})
        self.assertEqual(exp.learner.graph_version,2)
        self.assertEqual(len(exp.learner.ledger),1)

if __name__=='__main__':unittest.main()

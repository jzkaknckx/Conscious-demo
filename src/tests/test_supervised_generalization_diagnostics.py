import copy
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from nns.memorygraphs.pending_associations import normalize_pending, register_outcome, review_regions
from nns.memorygraphs.experiment_diagnostics import weak_class_features
from nns.memorygraphs.supervised_readout import ClassEvidenceReadout, ClassifierTrainer, detection_metrics
from nns.memorygraphs.supervised_data import SupervisedConfig
from nns.memorygraphs.familiarity_probe import ProbeConfig, pilot_verdict, perturb_observation


class GeneralizationDiagnosticsTests(unittest.TestCase):
    def test_empty_repair_history_is_not_competing_evidence(self):
        e=NS(supervision={'pending_reuse':{'source':{'regions':{0:{'candidates':[]}},'entity_candidates':[]}}})
        self.assertTrue(normalize_pending(e))
        self.assertFalse(e.supervision['reuse_unresolved'])
        self.assertEqual(e.supervision['pending_reuse']['source']['regions'][0]['status'],'resolved_local')
        self.assertFalse(normalize_pending(e))
        e.supervision['pending_reuse']['source']['regions'][1]={'candidates':[{'semantic_id':3}]}
        normalize_pending(e)
        self.assertTrue(e.supervision['reuse_unresolved'])

    def test_independent_evidence_and_unknown(self):
        c={'status':'pending'}
        register_outcome(c,'image:object',None,2)
        register_outcome(c,'image:object',True,2)
        register_outcome(c,'image:object',True,2)
        self.assertEqual(c['positive_episodes'],['image:object'])
        self.assertEqual(c['status'],'pending')
        register_outcome(c,'other:object',True,2)
        self.assertEqual(c['status'],'confirmed')
        register_outcome(c,'third:object',False,2)
        self.assertEqual(c['status'],'pending')  # conflict stays unresolved

    def test_changed_template_invalidates_confirmed_region(self):
        c={'semantic_id':3,'version':1,'status':'confirmed','positive_episodes':['a','b']}
        item={'status':'confirmed','local_semantic_id':2,'confirmed_semantic_id':3,'candidates':[c]}
        e=NS(supervision={'pending_reuse':{'source':{'regions':{0:item},'entity_candidates':[]}}},
             component_edges={0:{'sem_id':2,'dx':0.,'dy':0.}})
        cfg=NS(graph_stable_support=2,graph_evaluable_min=.5,graph_learn_threshold=.9,graph_recall_threshold=.7)
        gii=NS(semantic_nodes={2:NS(version=1),3:NS(version=2)})
        with patch('nns.memorygraphs.pending_associations.region_view',return_value=None), patch(
                'nns.memorygraphs.pending_associations.SpatialStructureMatcher') as matcher:
            matcher.return_value.evaluate.return_value=[NS(accepted=False)]
            review_regions(e,gii,None,(0.,0.),'new',cfg)
        self.assertEqual(c['status'],'stale')
        self.assertEqual(c['positive_episodes'],[])
        self.assertNotIn('confirmed_semantic_id',item)
        self.assertTrue(e.supervision['reuse_unresolved'])

    def test_readout_failure_reasons_are_distinct(self):
        readout=ClassEvidenceReadout(('cat','dog'),SupervisedConfig())
        base={'best':{},'scores':np.zeros(2),'search_complete':False,
              'assignment_search_complete':False,'graph_version':1}
        self.assertEqual(readout.predict(base,None)['rejection_reason'],'no_accepted_entity')
        base['best']={0:NS(template_id=1,point=(1,2))}
        self.assertEqual(readout.predict(base,None)['rejection_reason'],'class_score_below_threshold')
        base['scores']=np.array([1.,1.])
        self.assertEqual(readout.predict(base,None)['rejection_reason'],'class_margin_below_threshold')
        base['scores']=np.array([1.,0.])
        self.assertIsNone(readout.predict(base,None)['rejection_reason'])

    def test_empty_detections_count_misses(self):
        m=detection_metrics([],{'image':[NS(class_id='cat',bbox=(0,0,10,10))]},('cat','dog'))
        self.assertEqual(m['counts']['cat'],{'truth':1,'tp':0,'fp':0,'fn':1,'recall':0.})
        self.assertIsNone(m['counts']['dog']['recall'])

    def test_rejected_hypothesis_retains_weak_evidence_without_mutation(self):
        best=dict(score=.7,evaluable=1.,coverage=.6,geometry_error=2.,accepted=False)
        q=NS(entities=[],diagnostics={'trace':[{'level':3,'template_id':3,'best':best}],'search_complete':False})
        model=NS(gmem_iii=NS(entity_nodes={3:NS(supervision={'label_id':'cat'})}),cfg=NS(graph_geometry_sigma=2.))
        before=copy.deepcopy(q.diagnostics)
        features=weak_class_features(q,model,('cat','dog')).reshape(2,6)
        np.testing.assert_allclose(features[0],[.7,1.,.6,1.,1.,1.])
        self.assertEqual(q.diagnostics,before)
        self.assertEqual(q.entities,[])

    def test_weak_classifier_checkpoint_preserves_evidence_contract(self):
        cfg=SupervisedConfig(epochs=1)
        trainer=ClassifierTrainer(('cat','dog'),2,cfg,device='cpu')
        x=torch.arange(24,dtype=torch.float32).reshape(2,12)/24
        batch=dict(features=x,targets=torch.tensor([0,1]),classes=('cat','dog'),
                   graph_version=2,feature_source='weak')
        fit=dict(batch,split='readout_fit',source_ids=['fit1','fit2'])
        val=dict(batch,split='validation',source_ids=['val1','val2'])
        trainer.fit(fit,val,['memory'])
        restored=ClassifierTrainer(('cat','dog'),2,cfg,device='cpu')
        restored.load_state_dict(trainer.state_dict())
        self.assertEqual(restored.feature_source,'weak')
        self.assertTrue(torch.allclose(restored.predict(x,2),trainer.predict(x,2)))
        with self.assertRaisesRegex(ValueError,'evidence sources differ'):
            trainer.fit(fit,dict(val,feature_source='accepted'),['memory'])

    def test_normalization_uses_fit_only_and_baseline_is_recorded(self):
        cfg=SupervisedConfig(epochs=1)
        trainer=ClassifierTrainer(('cat','dog'),2,cfg,device='cpu')
        x=torch.arange(24,dtype=torch.float32).reshape(2,12)
        x[:,0]=4.
        batch=dict(features=x,targets=torch.tensor([0,1]),classes=('cat','dog'),graph_version=2)
        fit=dict(batch,split='readout_fit',source_ids=['fit1','fit2'])
        val=dict(batch,features=x+100.,split='validation',source_ids=['val1','val2'])
        history=trainer.fit(fit,val,['memory'])
        torch.testing.assert_close(trainer.feature_mean,x.mean(0))
        self.assertEqual(float(trainer.feature_scale[0]),1.)
        self.assertFalse(trainer.feature_active[0])
        altered=x.clone();altered[:,0]+=1000.
        torch.testing.assert_close(trainer.predict(x,2),trainer.predict(altered,2))
        self.assertEqual(history[0]['epoch'],-1)
        self.assertEqual(history[0]['updates'],0)
        self.assertIn('validation_predicted_counts',history[0])
        self.assertEqual(history[1]['updates'],1)

    def test_pilot_does_not_ignore_failed_scale_and_occlusion(self):
        cfg=ProbeConfig()
        rows=[dict(variant=v,anchor_families=3,anchor_span=.5,candidate_ambiguous=False,
                   candidate_id=1,target_entity_id=1,identity_verified=True,action='reinforce_known') for v in cfg.variants]
        rows[-1].update(candidate_id=2,action='insufficient_correspondence')
        self.assertEqual(pilot_verdict(rows,cfg),'continue_readonly_validation')
        rows[1]['identity_verified']=False
        self.assertEqual(pilot_verdict(rows,cfg),'fix_local_correspondence_first')
        rows[1]['identity_verified']=True
        rows[2]['anchor_families']=0
        self.assertEqual(pilot_verdict(rows,cfg),'fix_scale_and_partial_correspondence_before_learning')
        self.assertEqual(pilot_verdict(rows[:2],cfg),'insufficient_controls')

    def test_translation_masks_padding_and_occlusion_not_oracle(self):
        from dataclasses import make_dataclass
        Obs=make_dataclass('Obs',['view_tensor','image_valid_mask','object_roi_mask','augmentation_id'])
        o=Obs(torch.rand(1,3,12,12),torch.ones(12,12,dtype=torch.bool),torch.ones(12,12,dtype=torch.bool),'')
        moved,_=perturb_observation(o,'translation',ProbeConfig(shift_pixels=4))
        self.assertFalse(moved.image_valid_mask[:,:4].any())
        self.assertTrue(moved.image_valid_mask[:,4:].all())
        covered,mask=perturb_observation(o,'occlusion',ProbeConfig())
        self.assertTrue(mask.any())
        self.assertTrue(torch.equal(covered.image_valid_mask,o.image_valid_mask))
        self.assertTrue(torch.equal(covered.object_roi_mask,o.object_roi_mask))

if __name__=='__main__':unittest.main()

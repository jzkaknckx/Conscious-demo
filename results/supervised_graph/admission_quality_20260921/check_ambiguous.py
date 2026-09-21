"""Controlled tie injection tests admission/transaction isolation, not similarity accuracy."""
import torch,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,MultilevelCoordinator,GraphConsolidationOptimizer
from nns.memorygraphs.supervised_data import AnnotationDataset,SupervisedConfig,ObjectViewTransform
from nns.memorygraphs.supervised_graph_learning import SupervisedGraphOptimizer
from nns.memorygraphs.supervised_readout import FixedCNNEncoder

torch.set_num_threads(2)
cfg=MemoryConfig();cfg.device=torch.device('cpu');c=SupervisedConfig()
p=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007');d=AnnotationDataset(p/'JPEGImages',p/'Annotations','1based_inclusive',dataset_version='VOC2007')
m=MultilevelCoordinator(cfg);l=SupervisedGraphOptimizer(m,('cat','dog','car'),c);encoder=FixedCNNEncoder(cfg,c)
def inputs(iid,oid):
 im,anns=d.load_image_objects(iid);a=next(a for a in anns if a.object_id==oid)
 o=ObjectViewTransform(c).make_view(im,a);return o,encoder(o)
o,f=inputs('003671','0');assert l.learn_object(o,f)['status']=='ANNOTATED_SEED_CREATED'
before={sid:sem.evidence.support for sid,sem in m.gmem_ii.semantic_nodes.items()}
def tied(self,region,report,gi,gii,provider,prior,budget=None,coarse=None,candidate_sink=None):
 ids=[sid for sid,sem in gii.semantic_nodes.items() if sem.kind=='region' and gi.nodes[sem.anchor_id].modality_id==region.modality_id]
 if len(ids)<2:return None,False
 candidate_sink.extend(NS(template_id=sid,score=.9,point=(region.anchor%report.shape[1],region.anchor//report.shape[1]),version=gii.semantic_nodes[sid].version) for sid in ids[:2])
 return None,True
o,f=inputs('007490','0')
with patch.object(GraphConsolidationOptimizer,'_region_match',tied):r=l.learn_object(o,f)
assert r['status']=='ANNOTATED_SEED_CREATED',r
assert r['reuse_unresolved']
assert all(m.gmem_ii.semantic_nodes[sid].evidence.support==v for sid,v in before.items())
assert len(l.ledger)==2
out={'controlled_region_tie':True,'status':r['status'],'reuse_unresolved':r['reuse_unresolved'],'competitor_support_unchanged':True,'ledger_count':len(l.ledger)}
Path('results/supervised_graph/admission_quality_20260921/ambiguous_summary.json').write_text(json.dumps(out,indent=2));print(out)

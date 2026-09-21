import json,torch
from pathlib import Path
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,MultilevelCoordinator
from nns.memorygraphs.supervised_data import AnnotationDataset,SupervisedConfig,ObjectViewTransform
from nns.memorygraphs.supervised_graph_learning import SupervisedGraphOptimizer
from nns.memorygraphs.supervised_readout import FixedCNNEncoder

torch.set_num_threads(2)
cfg=MemoryConfig();cfg.device=torch.device('cpu');c=SupervisedConfig()
p=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007')
d=AnnotationDataset(p/'JPEGImages',p/'Annotations','1based_inclusive',dataset_version='VOC2007')
m=MultilevelCoordinator(cfg);l=SupervisedGraphOptimizer(m,('cat','dog','car'),c);enc=FixedCNNEncoder(cfg,c)
rows=[]
for iid,oid in [('003671','0'),('007490','0'),('007530','1')]:
 im,anns=d.load_image_objects(iid);a=next(a for a in anns if a.object_id==oid)
 o=ObjectViewTransform(c).make_view(im,a);f=enc(o)
 r=l.learn_object(o,f,capture_debug=True)
 print(iid,a.class_id,r['status'],r.get('reason'),flush=True)
 assert r['status'] in ('ANNOTATED_SEED_CREATED','MATCHED_UPDATED'),r.get('reason')
 audit=l.diagnose_write_recall(o,f,l.last_debug['write_witness'])
 print('recall',audit['ordinary_target_hit'],flush=True)
 before=(len(l.ledger),len(m.gmem_i.nodes),[e.evidence.support for e in m.gmem_iii.entity_nodes.values()])
 assert l.learn_object(o,f)['status']=='ALREADY_OBSERVED'
 assert before==(len(l.ledger),len(m.gmem_i.nodes),[e.evidence.support for e in m.gmem_iii.entity_nodes.values()])
 rows.append({'image_id':iid,'class':a.class_id,'status':r['status'],'reuse_unresolved':r['reuse_unresolved'],
 'decisions':r['reuse_decisions'],'recall':audit['ordinary_target_hit'],'known_pose':audit['known_pose_entity']})
Path('results/supervised_graph/admission_quality_20260921/summary.json').write_text(json.dumps(rows,indent=2,default=str))

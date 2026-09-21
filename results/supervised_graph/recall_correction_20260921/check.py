import torch, json
from collections import Counter
from pathlib import Path
from nns.memorygraphs.graph_memorypool_onceoptimizer import *
from nns.memorygraphs.supervised_data import *
from nns.memorygraphs.supervised_graph_learning import *
from nns.memorygraphs.supervised_readout import FixedCNNEncoder
torch.set_num_threads(2)
cfg=MemoryConfig();cfg.device=torch.device('cpu');cfg.once_advanced_partition_mode='grad_refined'
c=SupervisedConfig();root=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007')
d=AnnotationDataset(root/'JPEGImages',root/'Annotations','1based_inclusive',dataset_version='VOC2007')
m=MultilevelCoordinator(cfg);l=SupervisedGraphOptimizer(m,('cat','dog','car'),c);enc=FixedCNNEncoder(cfg,c)
rows=[]
for iid in ('007530','000209'):
 im,anns=d.load_image_objects(iid);a=next(a for a in anns if a.class_id=='cat');o=ObjectViewTransform(c).make_view(im,a);f=enc(o);r=l.learn_object(o,f,capture_debug=True);print(iid,r['status'],flush=True)
 if 'entity_id' not in r:continue
 audit=l.diagnose_write_recall(o,f,l.last_debug['write_witness'])
 assert audit['ordinary_target_hit'], audit['ordinary_entities']
 assert audit['known_pose_entity'][0]['accepted']
 counts=(len(m.gmem_i.nodes),len(m.gmem_ii.semantic_nodes),len(m.gmem_iii.entity_nodes))
 supports=[e.evidence.support for e in m.gmem_iii.entity_nodes.values()]
 assert l.learn_object(o,f)['status']=='ALREADY_OBSERVED'
 assert counts==(len(m.gmem_i.nodes),len(m.gmem_ii.semantic_nodes),len(m.gmem_iii.entity_nodes))
 assert supports==[e.evidence.support for e in m.gmem_iii.entity_nodes.values()]
 rows.append({'image_id':iid,'learning_status':r['status'],'modality_counts':r['modality_counts'],
     'target_hit':audit['ordinary_target_hit'],'known_pose':audit['known_pose_entity'],
     'ordinary_entities':audit['ordinary_entities'],
     'structural_fallback':audit['query']['structural_fallback'],'replay_unchanged':True})
 print(rows[-1],flush=True)
Path('results/supervised_graph/recall_correction_20260921/summary.json').write_text(json.dumps(rows,indent=2,default=str))

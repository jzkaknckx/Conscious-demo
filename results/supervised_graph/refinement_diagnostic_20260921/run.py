import json,time,copy
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,MultilevelCoordinator,OnceGraphBuilder,GmemoryI,GmemoryII,region_shape_descriptors,advanced_partition_contract
from nns.memorygraphs.supervised_data import AnnotationDataset,SupervisedConfig,ObjectViewTransform
from nns.memorygraphs.supervised_graph_learning import SupervisedGraphOptimizer
from nns.memorygraphs.supervised_readout import FixedCNNEncoder,ClassEvidenceReadout

torch.set_num_threads(2)
out=Path('results/supervised_graph/refinement_diagnostic_20260921')
root=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007')
dataset=AnnotationDataset(root/'JPEGImages',root/'Annotations','1based_inclusive',dataset_version='VOC2007')
cfg=MemoryConfig();cfg.device=torch.device('cpu')
config=SupervisedConfig();encoder=FixedCNNEncoder(cfg,config)
model=MultilevelCoordinator(cfg);learner=SupervisedGraphOptimizer(model,('cat','dog','car'),config)
rows=[]
for image_id in ('003671','007490'):
 image,objects=dataset.load_image_objects(image_id)
 annotation=next(a for a in objects if a.class_id in learner.classes and a.flags.get('difficult')!='1')
 observation=ObjectViewTransform(config).make_view(image,annotation)
 features=encoder(observation)
 for mode in ('grad_gated','grad_refined'):
  other=copy.deepcopy(cfg);other.once_advanced_partition_mode=mode
  report=OnceGraphBuilder(other).build_once(features,GmemoryI(),GmemoryII(other),valid_mask=observation.valid_mask)
  regions=[]
  for mid in range(5):
   rs=[r for r in report.regions if r.modality_id==mid]
   outside=sum(np.count_nonzero(report.labels[0].flat[r.pixels]<0) for r in rs) if mid in (2,3,4) else 0
   if mode=='grad_refined' and mid in (2,3,4):
    assert outside==0
    assert all(np.all(report.labels[0].flat[r.pixels]==r.support_parent_region_id) for r in rs)
   regions.append({'modality':mid,'regions':len(rs),'pixels':sum(len(r.pixels) for r in rs),'truncated':sum(r.budget_truncated for r in rs),'outside_grad_partition':int(outside)})
  rows.append({'image':image_id,'mode':mode,'regions':regions,'rgb_shapes':region_shape_descriptors(report)})
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,5,figsize=(15,3))
  cmap=copy.copy(plt.get_cmap('tab20'));cmap.set_bad('black')
  for mid,ax in enumerate(axes):
   lab=report.labels[mid];ax.imshow(np.ma.masked_where(lab<0,lab),cmap=cmap,interpolation='nearest');ax.set_title(f'{mid}: {regions[mid]["regions"]} regions');ax.axis('off')
  fig.suptitle(f'{image_id}: {mode}');fig.tight_layout();fig.savefig(out/f'{image_id}_{mode}.png');plt.close(fig)
 start=time.perf_counter()
 result=learner.learn_object(observation,features,capture_debug=True,profile_stages=True)
 print(image_id,result['status'],result.get('reason'),time.perf_counter()-start,flush=True)
 query=model.query_hierarchy(features,valid_mask=observation.valid_mask)
 readout=ClassEvidenceReadout(learner.classes,config)
 pred=readout.predict(readout.encode(query,model,learner.graph_version),model,observation)
 rows.append({'image':image_id,'learning':result,'self_query':pred})
 if result['status'] in ('ANNOTATED_SEED_CREATED','MATCHED_UPDATED'):
  count=len(model.gmem_iii.entity_nodes);support=[e.evidence.support for e in model.gmem_iii.entity_nodes.values()]
  replay=learner.learn_object(observation,features)
  assert replay['status']=='ALREADY_OBSERVED'
  assert count==len(model.gmem_iii.entity_nodes) and support==[e.evidence.support for e in model.gmem_iii.entity_nodes.values()]
(out/'summary.json').write_text(json.dumps({'device':'cpu','contract':advanced_partition_contract(cfg),'rows':rows},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
print('Saved diagnostics',out)

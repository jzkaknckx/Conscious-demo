import torch,json,time
from pathlib import Path
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,MultilevelCoordinator
from nns.memorygraphs.supervised_data import SupervisedConfig,AnnotationDataset,ObjectViewTransform
from nns.memorygraphs.supervised_graph_learning import SupervisedGraphOptimizer
from nns.memorygraphs.supervised_readout import FixedCNNEncoder,ClassEvidenceReadout

torch.set_num_threads(2)
rundir=Path('results/supervised_graph/voc2007_20260921_231411_853393')
s=torch.load(rundir/'memory.pt',map_location='cpu',weights_only=False)
cfg=MemoryConfig()
for k,v in s['memory_config'].items():setattr(cfg,k,v)
cfg.device=torch.device('cpu');config=SupervisedConfig(**s['config']);model=MultilevelCoordinator(cfg);l=SupervisedGraphOptimizer(model,s['classes'],config);l.load_state_dict(s)
encoder=FixedCNNEncoder(cfg,config);readout=ClassEvidenceReadout(l.classes,config)
p=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007');d=AnnotationDataset(p/'JPEGImages',p/'Annotations','1based_inclusive',dataset_version='VOC2007')
log=[json.loads(x) for x in open(rundir/'learning.jsonl')];out=[]
for idx in (0,16,33):
 row=log[idx];im,anns=d.load_image_objects(row['image_id']);a=next(a for a in anns if a.object_id==row['object_id']);o=ObjectViewTransform(config).make_view(im,a)
 f=encoder(o);start=time.perf_counter();q=model.query_hierarchy(f,valid_mask=o.valid_mask);pred=readout.predict(readout.encode(q,model,l.graph_version),model,o)
 result={'index':idx,'image_id':a.image_id,'object_id':a.object_id,'class':a.class_id,'target':row['entity_id'],'hits':[m.template_id for m in q.entities],'scores':[m.score for m in q.entities],'target_hit':any(m.template_id==row['entity_id'] for m in q.entities),'prediction':pred['class_id'],'status':pred['status'],'fallback':q.diagnostics.get('structural_fallback'),'seconds':time.perf_counter()-start}
 out.append(result);print(json.dumps(result,default=str),flush=True)
 Path('results/supervised_graph/scale_readiness_20260921/summary.json').write_text(json.dumps({'source_run':rundir.name,'device':'cpu','graph_version':l.graph_version,'rows':out},indent=2,default=str))
assert l.graph_version==34 and len(l.ledger)==34

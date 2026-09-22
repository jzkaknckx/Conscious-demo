import torch,json,time
from pathlib import Path
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,MultilevelCoordinator
from nns.memorygraphs.supervised_data import SupervisedConfig,AnnotationDataset
from nns.memorygraphs.supervised_graph_learning import SupervisedGraphOptimizer
from nns.memorygraphs.supervised_readout import FixedCNNEncoder
from nns.memorygraphs.supervised_experiment import SupervisedExperiment
from nns.memorygraphs.familiarity_probe import run_familiarity_pilot,ProbeConfig
from nns.memorygraphs.experiment_diagnostics import save_json

torch.set_num_threads(2)
source=Path('results/supervised_graph/voc2007_20260921_231411_853393')
s=torch.load(source/'memory.pt',map_location='cpu',weights_only=False)
cfg=MemoryConfig()
for k,v in s['memory_config'].items():setattr(cfg,k,v)
cfg.device=torch.device('cpu');sc=SupervisedConfig(**s['config'])
m=MultilevelCoordinator(cfg);l=SupervisedGraphOptimizer(m,s['classes'],sc);l.load_state_dict(s)
root=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007')
class Dataset(AnnotationDataset):
 def annotations(self,image_id):return [a for a in super().annotations(image_id) if a.flags.get('difficult')!='1']
d=Dataset(root/'JPEGImages',root/'Annotations','1based_inclusive',dataset_version='VOC2007')
out=Path('results/supervised_graph/generalization_pilot_20260922')
save_json(out/'maintenance.json',l.maintain_pending())
e=SupervisedExperiment(l,FixedCNNEncoder(cfg,sc),d,s['split_manifest'])
r=run_familiarity_pilot(e,out/'pilot',ProbeConfig(max_objects=1),on_result=lambda x:print(json.dumps({k:x.get(k) for k in ('variant','target_hit','candidate_id','anchor_families','familiar','difference','unknown','action')}),flush=True))
print('VERDICT',r['verdict'],flush=True)
cache=e.query_split('validation',max_images=1,output_dir=out,on_result=lambda x:print('validation',x['image_id'],x['prediction']['rejection_reason'],x.get('accepted_regions'),x.get('entity_candidates'),flush=True))
print('METRICS',e.evaluate(cache),flush=True)

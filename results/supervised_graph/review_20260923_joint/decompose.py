"""Frozen, known-transform loss decomposition; no learning or threshold changes."""
import json,copy,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,MultilevelCoordinator,FeatureResponseCache,SpatialStructureMatcher,entity_view
from nns.memorygraphs.supervised_data import SupervisedConfig,AnnotationDataset
from nns.memorygraphs.supervised_graph_learning import SupervisedGraphOptimizer
from nns.memorygraphs.supervised_readout import FixedCNNEncoder
from nns.memorygraphs.supervised_experiment import SupervisedExperiment
from nns.memorygraphs.familiarity_probe import perturb_observation,ProbeConfig
from nns.memorygraphs.experiment_diagnostics import save_json

torch.set_num_threads(2)
source=Path('results/supervised_graph/voc2007_20260923_171046_543975')
state=torch.load(source/'memory.pt',map_location='cpu',weights_only=False)
cfg=MemoryConfig()
for k,v in state['memory_config'].items():setattr(cfg,k,v)
cfg.device=torch.device('cpu');sc=SupervisedConfig(**state['config'])
model=MultilevelCoordinator(cfg);learner=SupervisedGraphOptimizer(model,state['classes'],sc);learner.load_state_dict(state)
root=Path('/home/p/code/datasets/VOC2007/VOCdevkit/VOC2007')
class Dataset(AnnotationDataset):
 def annotations(self,image_id):return [a for a in super().annotations(image_id) if a.flags.get('difficult')!='1']
e=SupervisedExperiment(learner,FixedCNNEncoder(cfg,sc),Dataset(root/'JPEGImages',root/'Annotations','1based_inclusive',dataset_version='VOC2007'),state['split_manifest'])
pilot=json.loads((source/'familiarity_pilot/summary.json').read_text())
baselines={(r['image_id'],r['object_id']):r for r in pilot['rows'] if r['variant']=='identity'}
rows=[]
for a,obs in e._objects('memory_build'):
 key=(a.image_id,a.object_id)
 if key not in baselines:continue
 base=baselines.pop(key)
 for variant in ('scale','occlusion'):
  o,changed=perturb_observation(obs,variant,ProbeConfig());features=e.encoder(o)
  provider=FeatureResponseCache(features,model.gmem_i,cfg,o.valid_mask)
  view=copy.deepcopy(entity_view(model.gmem_iii.entity_nodes[base['target_entity_id']],model.gmem_ii,cfg))
  h,w=provider.shape;scale=.95 if variant=='scale' else 1.;center=np.array([(w-1)/2,(h-1)/2]);pose=center+scale*(np.asarray(base['candidate_pose'])-center)
  for s in view.slots:
   s['xy']=np.asarray(s['xy'])*scale
   if 'member_xy' in s:s['member_xy']=np.asarray(s['member_xy'])*scale
  for edge in view.constraints:edge['delta']=np.asarray(edge['delta'])*scale
  match=SpatialStructureMatcher(cfg).evaluate(view,provider,[pose])[0]
  groups=defaultdict(list)
  for s in view.slots:groups[s['group']].append(s)
  gw={g:ss[0]['group_weight'] for g,ss in groups.items()};total=sum(gw.values())
  weights={s['key']:gw[g]/total*s['weight']/sum(i['weight'] for i in ss) for g,ss in groups.items() for s in ss}
  shared=Counter((s['node_id'],tuple(round(v,6) for v in s['xy'])) for s in view.slots)
  for s in view.slots:weights[s['key']]/=shared[(s['node_id'],tuple(round(v,6) for v in s['xy']))]
  norm=sum(weights.values());weights={k:v/norm for k,v in weights.items()}
  mods=defaultdict(lambda:dict(evaluable_mass=0.,loss=0.,low_mass=0.,matched_mass=0.,slots=0))
  for s in view.slots:
   assigned=match.assignments.get(s['key'])
   if assigned is None:continue
   mid=model.gmem_i.nodes[s['node_id']].modality_id;v=mods[mid];weight=weights[s['key']]/match.evaluable_coverage;r=assigned['score']
   v['evaluable_mass']+=weight;v['loss']-=weight*math.log(max(r,1e-8));v['low_mass']+=weight*(r<1e-4);v['matched_mass']+=weight*(r>=cfg.graph_recall_threshold);v['slots']+=1
  loss=sum(v['loss'] for v in mods.values())
  row={'image_id':a.image_id,'object_id':a.object_id,'variant':variant,'score':match.score,'coverage':match.matched_coverage,'evaluable':match.evaluable_coverage,'reasons':match.rejection_reasons,'modalities':dict(mods),'loss':loss,'low_mass':sum(v['low_mass'] for v in mods.values())}
  rows.append(row);print(a.image_id,variant,round(match.score,4),'low_mass',round(row['low_mass'],4),'loss_share',{mid:round(v['loss']/loss,3) for mid,v in mods.items()},flush=True)
 if not baselines:break
save_json(Path(__file__).with_name('loss_decomposition.json'),{'scope':'CPU frozen oracle diagnostic; normal retrieval unchanged','source':str(source),'rows':rows})

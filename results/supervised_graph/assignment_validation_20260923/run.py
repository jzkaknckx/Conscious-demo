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
out=Path('results/supervised_graph/assignment_validation_20260923')
save_json(out/'maintenance.json',l.maintain_pending())
e=SupervisedExperiment(l,FixedCNNEncoder(cfg,sc),d,s['split_manifest'])

from nns.memorygraphs.familiarity_probe import perturb_observation, FamiliarityProbe
from nns.memorygraphs.graph_memorypool_onceoptimizer import FeatureResponseCache,SpatialStructureMatcher,entity_view
annotation,obs=next((a,o) for a,o in e._objects('memory_build') if a.image_id=='007490')
target=l.ledger[obs.ledger_key]['entity_id']
prior=Path('results/supervised_graph/voc2007_20260922_222716_908742/familiarity_pilot')
results=[]
for variant in ('identity','translation','brightness','layout_negative'):
    old=json.loads(next(prior.glob('007490_*_'+variant+'.json')).read_text())
    traces=old['query']['trace']
    target_trace=next(r for r in traces if r['level']==3 and r['template_id']==target)
    pose=target_trace['best']['point']
    perturbed,_=perturb_observation(obs,variant,ProbeConfig())
    features=e.encoder(perturbed)
    provider=FeatureResponseCache(features,m.gmem_i,cfg,perturbed.valid_mask)
    row={'variant':variant,'target':target,'pose':pose}
    for enabled in (False,True):
        cfg.graph_joint_assignment=enabled
        start=time.perf_counter()
        match=SpatialStructureMatcher(cfg).evaluate(entity_view(m.gmem_iii.entity_nodes[target],m.gmem_ii,cfg),provider,[pose])[0]
        row[str(enabled)]={'seconds':time.perf_counter()-start,**match.summary()}
    results.append(row)
    print(variant,[(k,row[k]['accepted'],row[k]['rejection_reasons'],row[k]['assignment_diagnostics'].get('status'),row[k]['assignment_diagnostics'].get('initial_conflicts')) for k in ('False','True')],flush=True)
    save_json(out/'known_pose_comparison.json',results)
# One real retrieval, not an oracle result.
perturbed,_=perturb_observation(obs,'translation',ProbeConfig())
features=e.encoder(perturbed)
q=m.query_hierarchy(features,valid_mask=perturbed.valid_mask,trace=True)
decision,_=FamiliarityProbe(m).analyze(features,perturbed.valid_mask,q)
save_json(out/'ordinary_translation.json',{'target_hit':any(a.template_id==target for a in q.entities),'decision':decision})
print('ordinary',any(a.template_id==target for a in q.entities),decision['action'],flush=True)

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

from nns.memorygraphs.familiarity_probe import known_transform_diagnostic
from nns.memorygraphs.graph_memorypool_onceoptimizer import HierarchyQuery
identity=json.loads(next(prior.glob('007490_*_identity.json')).read_text())
pose=next(r['best']['point'] for r in identity['query']['trace'] if r['level']==3 and r['template_id']==target)
rows=[]
for variant in ('identity','translation','scale'):
    transformed,_=perturb_observation(obs,variant,ProbeConfig())
    features=e.encoder(transformed)
    diagnostic=known_transform_diagnostic(m,features,transformed.valid_mask,HierarchyQuery(),target,pose,variant,ProbeConfig())
    rows.append({'variant':variant,'scope':'known transform only; ordinary query intentionally not run',**diagnostic})
    print(variant,diagnostic['known_center_original_geometry']['accepted'],diagnostic['known_center_transformed_geometry']['accepted'],flush=True)
save_json(out/'oracle_geometry_check.json',rows)

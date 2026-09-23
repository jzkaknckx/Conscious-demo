"""Read-only diagnostics: no graph learning, head fitting, or parameter selection."""
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[3]
source=root/'results/supervised_graph/voc2007_20260922_222716_908742'
torch.manual_seed(0)
initial=torch.nn.Linear(18,3)

def metrics(y,pred):
    cm=torch.zeros(3,3,dtype=torch.long)
    for a,b in zip(y,pred):cm[a,b]+=1
    tp=cm.diag().float();recall=tp/cm.sum(1).clamp_min(1)
    f1=2*tp/(cm.sum(0)+cm.sum(1)).clamp_min(1)
    return dict(accuracy=float((y==pred).float().mean()),macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()),confusion=cm.tolist(),prediction_counts=torch.bincount(pred,minlength=3).tolist())

result={'source':str(source.relative_to(root)),'classes':['cat','dog','car'],'scope':'posthoc cached evidence analysis; no model selection','splits':{}}
for split in ('readout_fit','validation','test'):
    cache=torch.load(source/f'{split}_cache.pt',map_location='cpu',weights_only=False)
    x=cache['weak_features'];y=cache['targets'];v=x.reshape(-1,3,6)
    predictions={
        'untrained_seed0_head':initial(x).argmax(1),
        'always_car':torch.full_like(y,2),
        'always_dog_fit_majority':torch.full_like(y,1),
        'weak_score_argmax':v[:,:,0].argmax(1),
        'weak_score_times_coverage':(v[:,:,0]*v[:,:,2]).argmax(1),
        'weak_score_times_coverage_times_evaluable':(v[:,:,0]*v[:,:,2]*v[:,:,1]).argmax(1)}
    result['splits'][split]={'samples':len(y),'baselines':{k:metrics(y,p) for k,p in predictions.items()},
        'constant_feature_columns':torch.where(x.amax(0)==x.amin(0))[0].tolist(),
        'candidate_class_mean_features':v.mean(0).tolist()}
path=Path(__file__).with_name('cached_evidence_analysis.json');path.write_text(json.dumps(result,indent=2))
for name,split in result['splits'].items():
    print(name,{k:{m:round(v[m],4) for m in ('accuracy','macro_f1')} for k,v in split['baselines'].items()})

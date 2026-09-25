"""Fixed-seed readout diagnostic on the previous frozen caches; no graph or threshold tuning."""
from pathlib import Path
import torch
from dataclasses import replace
from nns.memorygraphs.supervised_readout import ClassifierTrainer, classification_metrics
from nns.memorygraphs.supervised_data import SupervisedConfig
from nns.memorygraphs.experiment_diagnostics import save_json
import json

torch.set_num_threads(2)
source=Path('results/supervised_graph/voc2007_20260922_222716_908742')
caches={name:torch.load(source/f'{name}_cache.pt',map_location='cpu',weights_only=False) for name in ('readout_fit','validation','test')}
for c in caches.values():c['features']=c['weak_features'];c['feature_source']='weak'
splits=json.loads((source/'split.json').read_text());results={}
for seed in (0,1,2):
    cfg=SupervisedConfig(seed=seed)
    trainer=ClassifierTrainer(('cat','dog','car'),35,cfg,device='cpu')
    history=trainer.fit(caches['readout_fit'],caches['validation'],splits['splits']['memory_build'])
    metrics={}
    for name in ('validation','test'):
        c=caches[name];pred=trainer.predict(c['features'],35).argmax(1)
        metrics[name]=classification_metrics([trainer.classes[int(y)] for y in c['targets']],
             [{'class_id':trainer.classes[int(y)],'status':'PREDICTED'} for y in pred],trainer.classes)
    results[seed]={'best_epoch':trainer.best_epoch,'history':history,'metrics':metrics}
    print(seed,trainer.best_epoch,{k:(v['accuracy'],v['macro_f1']) for k,v in metrics.items()},flush=True)
save_json(Path(__file__).with_name('cached_head_comparison.json'),{'cache_source':str(source),'scope':'old retrieval features; fixed seeds, no test selection','seeds':results})

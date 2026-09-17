import copy, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
from nns.memorygraphs import graph_memorypool_onceoptimizer as new
from nns.cnns.features import RetinaModel
spec=importlib.util.spec_from_file_location('once_baseline',Path(__file__).with_name('before.py'))
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
torch.set_num_threads(2)
cfg=new.MemoryConfig();cfg.device=torch.device('cuda:0');cfg.H=cfg.W=256
manifest=json.loads((root/'results/onceoptimizer/dataset_20260915_224255_911963/manifest.json').read_text())
for k,v in manifest['params'].items():setattr(cfg,k,v)
retina=RetinaModel(cropped_size=256,output_size=256,edge_apply_gaussian=True,edge_gauss_kernel_size=5,edge_gauss_sigma=1.).eval().to(cfg.device)
@torch.no_grad()
def encode(path):
 with Image.open(path) as image:
  image=image.convert('RGB');image.thumbnail((256,256))
  x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().to(cfg.device)/255.
 _,grad,_,_,_,rgb=retina(x,center_x=-1,center_y=-1)
 valid=retina.preprocess(torch.ones_like(x[:,:1]),-1,-1)[0,0]>.999
 return {'grad':grad,'rgb':rgb},valid

def timed(fn):
 torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
 return result,time.perf_counter()-start

def matches(result):return [x.summary() for x in result.regions],[x.summary() for x in result.entities]

with torch.no_grad():
 first,valid1=encode(manifest['train'][0]);second,valid2=encode(manifest['train'][1])
 coordinator=old.MultilevelCoordinator(cfg)
 _,build_seconds=timed(lambda:coordinator.learn_view(first,valid_mask=valid1,episode_id='first'))
 gi,gii,giii=coordinator.gmem_i,coordinator.gmem_ii,coordinator.gmem_iii
 old_query,t_old=timed(lambda:old.HierarchyRetriever(cfg).query(second,gi,gii,giii,valid2))
 new_query,t_new=timed(lambda:new.HierarchyRetriever(cfg).query(second,gi,gii,giii,valid2))
 assert matches(old_query)==matches(new_query),'query output differs'
 print('query',t_old,t_new,flush=True)
 oi,oii=old.GmemoryI(),old.GmemoryII(cfg)
 report=old.OnceGraphBuilder(cfg).build_once(second,oi,oii,valid2)
 # Fixed bounded sample, including surface and edge regions of differing sizes.
 regions=[r for r in report.regions if r.semantic_id is not None and len(r.pixels)<=512][:12]
 assert regions
 out={'first_build_seconds':build_seconds,'query_old_seconds':t_old,'query_new_seconds':t_new,
      'query_equivalent':True,'regions':[{'id':r.region_id,'pixels':len(r.pixels)} for r in regions]}
 decisions=[]
 for label,mod,prior in [('old',old,old_query),('new',new,new_query)]:
  optimizer=mod.GraphConsolidationOptimizer(cfg)
  provider=mod.FeatureResponseCache(second,gi,cfg,report.valid_mask)
  def run():return [optimizer._region_match(r,report,gi,gii,provider,prior) for r in regions]
  found,elapsed=timed(run)
  compact=[(m.summary() if m else None,ambiguous) for m,ambiguous in found]
  decisions.append(compact)
  out[label+'_region_seconds']=elapsed
  out[label+'_work_counts']=dict(optimizer.work_counts)
  if label=='new':out['shared_cache_stats']=dict(provider.stats)
  print(label,'region seconds',elapsed,flush=True)
 assert decisions[0]==decisions[1], 'region decisions differ'
 out['region_decisions_equivalent']=True
 out['query_speedup']=t_old/t_new;out['region_speedup']=out['old_region_seconds']/out['new_region_seconds']
 Path(__file__).with_name('comparison.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))

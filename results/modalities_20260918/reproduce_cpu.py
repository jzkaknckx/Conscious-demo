import json,torch,numpy as np
from PIL import Image
from nns.cnns.features import RetinaModel,MultiScaleFeatureBank
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig,prepare_feature_subspaces,OnceGraphBuilder,GmemoryI,GmemoryII
from collections import Counter

torch.set_num_threads(2)
path='/home/p/code/ILSVRC/Data/DET/train/ILSVRC2014_train_0006/ILSVRC2014_train_00060030.JPEG'
im=Image.open(path).convert('RGB');im.thumbnail((256,256))
x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)[None].float()/255
retina=RetinaModel(cropped_size=256,output_size=256,edge_apply_gaussian=True,edge_gauss_kernel_size=5,edge_gauss_sigma=1.).eval()
bank=MultiScaleFeatureBank([1,2,4]).eval()
def stats(t):
 a=t.detach().cpu().numpy();f=a[np.isfinite(a)]
 return {'shape':list(a.shape),'nan':int(np.isnan(a).sum()),'inf':int(np.isinf(a).sum()),'finite_fraction':float(np.isfinite(a).mean()),'quantiles':np.quantile(f,[0,.01,.5,.9,.99,1]).tolist() if len(f) else []}
with torch.no_grad():
 _,grad,_,_,_,cropped=retina(x,center_x=-1,center_y=-1)
 valid=retina.preprocess(torch.ones_like(x[:,:1]),-1,-1)[0,0]>.999
 curv,aps,ori=bank(cropped,precomputed_derivs={'grad':grad})
 cfg=MemoryConfig();cfg.device=torch.device('cpu');cfg.H=cfg.W=256
 inputs=prepare_feature_subspaces({'grad':grad,'rgb':cropped,'curvature':curv,'aspect':aps,'orientation':ori},cfg)
 out={'image':path,'device':'cpu','raw':{n:[stats(t[:,i:i+1]) for i in range(t.shape[1])] for n,t in [('curv',curv),('aps',aps),('ori',ori)]}}
 print(json.dumps(out),flush=True)
 gi,gii=GmemoryI(),GmemoryII(cfg)
 report=OnceGraphBuilder(cfg).build_once(inputs,gi,gii,valid)
 out['supports']={m:{'writable':int(s.writable.sum()),'support':int(s.support.sum()),'seeds':len(s.seeds),'events':int(s.events.sum()),'quality':stats(torch.from_numpy(s.quality)),'regions':sum(r.modality_id==m for r in report.regions),'nodes':sum(n.modality_id==m for n in gi.nodes.values())} for m,s in report.supports.items()}
 out['rejected']=dict(report.rejected)
 print(json.dumps(out['supports']),flush=True)
 # First-scale tensor should be PSD rank one; cancellation can make lambda2 negative.
 dx,dy=grad[...,0],grad[...,1]
 Jxx,Jyy,Jxy=dx*dx,dy*dy,dx*dy
 trace=Jxx+Jyy;disc=torch.clamp(trace*trace*.25-(Jxx*Jyy-Jxy*Jxy),min=0)
 l2=.5*trace-torch.sqrt(disc)
 out['aspect_rank_one']={'lambda2':stats(l2),'negative':int((l2<0).sum())}
 import nns.cnns.features as mod
 out['aspect_eps']=mod._EPS
 for name,v in [('zero',torch.zeros_like(cropped)),('constant',torch.ones_like(cropped)*.5)]:
  a,b,c=bank(v)
  out[name]={k:stats(t) for k,t in [('curv',a),('aps',b),('ori',c)]}
 open('results/modalities_20260918/diagnostics_cpu.json','w').write(json.dumps(out,indent=2))

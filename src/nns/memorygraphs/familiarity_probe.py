"""Frozen familiarity/difference pilot. Labels and write witnesses are evaluation-only."""
from dataclasses import dataclass, replace
from pathlib import Path
from collections import defaultdict, Counter
import math
import numpy as np
import torch
import torch.nn.functional as F
from .graph_memorypool_onceoptimizer import FeatureResponseCache, SpatialStructureMatcher, entity_view
from .experiment_diagnostics import save_json, query_audit


@dataclass(frozen=True)
class ProbeConfig:
    max_objects: int = 3
    shift_pixels: int = 4
    scale: float = .95
    brightness: float = .85
    min_anchor_families: int = 2
    min_anchor_span: float = .1  # diagonal fraction, not pixel count
    variants: tuple = ('identity', 'translation', 'scale', 'brightness', 'occlusion', 'layout_negative')


class FamiliarityProbe:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or ProbeConfig()

    @torch.no_grad()
    def analyze(self, features, valid_mask, query):
        model, cfg = self.model, self.model.cfg
        provider = FeatureResponseCache(features, model.gmem_i, cfg, valid_mask)
        h,w = provider.shape
        traces = query.diagnostics.get('trace', []) + query.diagnostics.get('initial_sparse', {}).get('trace', [])
        hypotheses = {}
        for row in traces:
            if row['level'] == 3 and row.get('best'):
                best = row['best']; rank=best['score']*best['coverage']
                if row['template_id'] not in hypotheses or rank > hypotheses[row['template_id']][0]:
                    hypotheses[row['template_id']] = (rank, best)
        candidates = sorted(hypotheses, key=lambda eid:(-hypotheses[eid][0],eid))
        maps = np.zeros((3,h,w),np.float32);maps[2]=1.
        if not candidates:
            return {'action':'insufficient_correspondence','reason':'no_L3_hypothesis',
                    'anchor_families':0,'candidate_id':None,'familiar':0.,'difference':0.,'unknown':1.},maps
        eid=candidates[0];best=hypotheses[eid][1]
        entity=model.gmem_iii.entity_nodes[eid];view=entity_view(entity,model.gmem_ii,cfg)
        match=SpatialStructureMatcher(cfg).evaluate(view,provider,[best['point']])[0]
        families=defaultdict(list)
        for slot in view.slots:
            role=slot['group'];member=entity.component_edges[role]
            families[member.get('family_id',role)].append(slot)
        ambiguous=(len(candidates)>1 and hypotheses[eid][0]-hypotheses[candidates[1]][0]<cfg.graph_match_margin)
        rows=[];anchors=[];numerator=np.zeros((3,h,w));denominator=np.zeros((h,w))
        radius=max(1,int(cfg.graph_geometry_radius))
        for family,slots in families.items():
            weights=np.array([max(float(s.get('weight',1.)),0.) for s in slots]);weights/=max(weights.sum(),1e-12)
            known=np.array([s['key'] in match.assignments for s in slots])
            coverage=float(weights[known].sum())
            similarity=sum(weight*match.assignments[s['key']]['score'] for s,weight,k in zip(slots,weights,known) if k)/max(coverage,1e-12)
            roles={s['group'] for s in slots}
            quality=min(entity.component_edges[r].get('reliability',1.) for r in roles)
            ambiguity=.5 if ambiguous else 0.
            comparable=quality*coverage*(1-ambiguity)
            familiar=comparable*similarity;difference=comparable*(1-similarity);unknown=1-comparable
            points=[tuple(np.asarray(best['point'])+s['xy']) for s in slots]
            center=np.mean(points,axis=0)
            if similarity>=cfg.graph_learn_threshold and coverage>=cfg.graph_evaluable_min:
                anchors.append(center)
            rows.append({'family':str(family),'familiar':familiar,'difference':difference,'unknown':unknown,
                         'similarity':similarity,'evaluable':coverage,'center':center.tolist()})
            # Each family votes once per pixel, even if its modalities overlap there.
            mask=np.zeros((h,w),bool)
            for px,py in points:
                x,y=int(round(px)),int(round(py))
                if 0<=x<w and 0<=y<h:mask[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]=True
            denominator[mask]+=1
            for k,val in enumerate((familiar,difference,unknown)):numerator[k,mask]+=val
        active=denominator>0
        maps[:,active]=(numerator[:,active]/denominator[active]).astype(np.float32)
        span=max((np.linalg.norm(a-b) for a in anchors for b in anchors),default=0.)/max(math.hypot(h,w),1.)
        mean={k:float(np.mean([r[k] for r in rows])) for k in ('familiar','difference','unknown')}
        sufficient=len(anchors)>=self.config.min_anchor_families and span>=self.config.min_anchor_span and not ambiguous
        if not sufficient:action='insufficient_correspondence'
        elif mean['unknown']>1-cfg.graph_evaluable_min:action='request_more_observation'
        elif mean['familiar']>=cfg.graph_learn_threshold and mean['difference']<=1-cfg.graph_learn_threshold:action='reinforce_known'
        elif mean['familiar']>=cfg.graph_coverage_min:action='consider_local_change'
        else:action='record_new_combination_candidate'
        return {'candidate_id':eid,'candidate_pose':best['point'],'action':action,'diagnostic_only':True,
                'anchor_families':len(anchors),'anchor_span':span,'candidate_ambiguous':ambiguous,
                'families':rows,'map_coverage':float(active.mean()),**mean},maps


def perturb_observation(observation, variant, cfg):
    """Synthetic controls on canonical inputs; masks describe validity, never the known answer."""
    image=observation.view_tensor.clone();h,w=image.shape[-2:]
    changed=torch.zeros((h,w),dtype=torch.bool,device=image.device)
    image_valid=observation.image_valid_mask.clone();roi=observation.object_roi_mask.clone()
    if variant in ('translation','scale'):
        scale=1. if variant=='translation' else cfg.scale
        dx=cfg.shift_pixels if variant=='translation' else 0
        matrix=torch.tensor([[[1/scale,0.,-2*dx/w],[0.,1/scale,0.]]],device=image.device,dtype=image.dtype)
        grid=F.affine_grid(matrix,image.shape,align_corners=False)
        image=F.grid_sample(image,grid,align_corners=False)
        def warp(mask):
            return F.grid_sample(mask.to(image.device).float()[None,None],grid,mode='nearest',align_corners=False)[0,0]>.5
        image_valid=warp(image_valid);roi=warp(roi)
    elif variant=='brightness':image=(image*cfg.brightness).clamp(0,1);changed[:]=True
    elif variant=='occlusion':
        changed[h//3:h//2,w//3:w//2]=True
        image[:,:,changed]=image.mean(dim=(-2,-1),keepdim=False)[:,:,None]
        # Do not give the algorithm an oracle occlusion mask.
    elif variant=='layout_negative':
        # Roll every other quadrant: exact same local pixels, deliberately changed layout.
        a=image[:,:,:h//2,:w//2].clone();b=image[:,:,-(h//2):,-(w//2):].clone()
        image[:,:,:h//2,:w//2]=b;image[:,:,-(h//2):,-(w//2):]=a
        changed[:h//2,:w//2]=True;changed[-(h//2):,-(w//2):]=True
    elif variant!='identity':raise ValueError(variant)
    return replace(observation,view_tensor=image,image_valid_mask=image_valid,object_roi_mask=roi,
                   augmentation_id='probe:'+variant),changed.cpu().numpy()


def run_familiarity_pilot(experiment, output_dir, config=None, on_result=None):
    config=config or ProbeConfig();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    learner=experiment.learner;version=learner.graph_version;ledger_size=len(learner.ledger)
    probe=FamiliarityProbe(learner.model,config);rows=[];chosen=[]
    # Prefer distinct classes, then fill. Source labels select the evaluation cohort only.
    seen=set();fallback=[]
    for annotation,obs in experiment._objects('memory_build'):
        if obs.ledger_key not in learner.ledger:continue
        if annotation.class_id not in seen and len(chosen)<config.max_objects:
            chosen.append((annotation,obs));seen.add(annotation.class_id)
        else:fallback.append((annotation,obs))
        if len(chosen)>=config.max_objects:break
    chosen=(chosen+fallback)[:config.max_objects]
    for annotation,obs in chosen:
        target=learner.ledger[obs.ledger_key]['entity_id']
        for variant in config.variants:
            perturbed,changed=perturb_observation(obs,variant,config)
            features=experiment.encoder(perturbed)
            query=learner.model.query_hierarchy(features,valid_mask=perturbed.valid_mask,trace=True)
            decision,maps=probe.analyze(features,perturbed.valid_mask,query)
            prediction=experiment.readout.predict(experiment.readout.encode(query,learner.model,version),learner.model)
            row={'image_id':annotation.image_id,'object_id':annotation.object_id,'variant':variant,
                 'target_entity_id':target,'target_hit':any(m.template_id==target for m in query.entities),
                 'class_id':annotation.class_id,'prediction':prediction,'graph_version':version,**decision}
            active=(maps[0]+maps[1])>0
            for label,mask in (('changed',changed & active),('unchanged',~changed & active)):
                row['difference_'+label]=float(maps[1,mask].mean()) if mask.any() else None
                row['unknown_'+label]=float(maps[2,mask].mean()) if mask.any() else None
            stem=f'{annotation.image_id}_{annotation.object_id}_{variant}'
            np.savez_compressed(output_dir/(stem+'.npz'),familiar=maps[0],difference=maps[1],unknown=maps[2],changed_mask=changed)
            save_json(output_dir/(stem+'.json'),{'result':row,'query':query_audit(query,prediction)})
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,4,figsize=(12,3))
            axes[0].imshow(perturbed.view_tensor[0].detach().cpu().permute(1,2,0).clamp(0,1));axes[0].set_title(variant)
            for ax,values,title in zip(axes[1:],maps,('Familiar','Reliable difference','Unknown')):
                ax.imshow(values,vmin=0,vmax=1,cmap='magma');ax.set_title(title)
            for ax in axes:ax.axis('off')
            fig.tight_layout();fig.savefig(output_dir/(stem+'.png'));plt.close(fig)
            rows.append(row);save_json(output_dir/'progress.json',rows)
            if on_result:on_result(row)
            if learner.graph_version!=version or len(learner.ledger)!=ledger_size:raise RuntimeError('Probe mutated learning state')
    verdict = pilot_verdict(rows, config)
    result={'graph_version':version,'cohort_objects':len(chosen),'rows':rows,'verdict':verdict,
            'independent_learning_evidence_added':0,'limitations':[
                'Frozen diagnostic, not an implemented learning algorithm.',
                'Layout scrambling is a structural control, not a guaranteed different semantic class.',
                'Three maps cover predicted template support only; unexplained foreground needs segmentation association.',
                'Occlusion mask is evaluation-only; flat occluders may remain confidently different, not unknown.']}
    save_json(output_dir/'summary.json',result);return result


def pilot_verdict(rows, config):
    """Conservative investment gate: identity success alone does not support learning."""
    def anchored(row):
        return (row['anchor_families'] >= config.min_anchor_families
                and row.get('anchor_span', 0.) >= config.min_anchor_span
                and not row.get('candidate_ambiguous', True)
                and row['candidate_id'] == row['target_entity_id'])
    by_variant = {name: [r for r in rows if r['variant'] == name] for name in config.variants}
    if any(not by_variant.get(name) for name in
           ('identity', 'translation', 'scale', 'brightness', 'occlusion', 'layout_negative')):
        return 'insufficient_controls'
    if not all(anchored(r) for name in ('identity', 'translation') for r in by_variant[name]):
        return 'fix_local_correspondence_first'
    if not all(anchored(r) for name in ('scale', 'brightness', 'occlusion') for r in by_variant[name]):
        return 'fix_scale_and_partial_correspondence_before_learning'
    if any(r['action'] in ('consider_local_change', 'reinforce_known') and
           r['candidate_id'] == r['target_entity_id'] for r in by_variant['layout_negative']):
        return 'difference_decision_not_discriminative'
    return 'continue_readonly_validation'

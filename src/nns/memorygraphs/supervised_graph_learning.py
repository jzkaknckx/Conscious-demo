"""Box-supervised transactions on the existing three-layer memory graph.

No CNN or graph-gradient training is performed here. Search budgets are optional,
cooperative, and abort a whole observation rather than commit partial evidence.
"""
import copy
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .supervised_data import SupervisedConfig
from .graph_memorypool_onceoptimizer import (
    GmemoryI, GmemoryII, GraphConsolidationOptimizer, FeatureResponseCache,
    BinaryCoarseIndex, HierarchyQuery, SearchBudget, SearchBudgetExceeded,
    SpatialStructureMatcher, region_view, entity_view, prepare_feature_subspaces,
)


def minimum_assignment(cost):
    """Rectangular Hungarian assignment, O(rows**2 * columns), no SciPy dependency."""
    a = np.asarray(cost, dtype=float)
    n, m = a.shape
    if n > m or not np.isfinite(a).all():
        raise ValueError('Assignment requires finite costs and columns >= rows')
    u, v, p, way = np.zeros(n+1), np.zeros(m+1), np.zeros(m+1, int), np.zeros(m+1, int)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        dist, used = np.full(m+1, np.inf), np.zeros(m+1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m+1):
                if not used[j]:
                    cur = a[i0-1, j-1] - u[i0] - v[j]
                    if cur < dist[j]:
                        dist[j], way[j] = cur, j0
                    if dist[j] < delta:
                        delta, j1 = dist[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    dist[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return {p[j]-1: j-1 for j in range(1, m+1) if p[j]}


class ObjectRegionSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, report, observation, observation_ii):
        roi = observation.valid_mask.detach().cpu().numpy().reshape(-1)
        fg = (observation.foreground_probability.detach().cpu().numpy().reshape(-1)
              if observation.foreground_probability is not None else np.ones(len(roi)))
        ownership = (observation.ambiguous_ownership_mask.detach().cpu().numpy().reshape(-1)
                     if observation.ambiguous_ownership_mask is not None else np.zeros(len(roi)))
        groups, rejected, quality = defaultdict(list), {}, {}
        for r in report.regions:
            if r.semantic_id is None or r.anchor is None:
                rejected[r.region_id] = 'unwritable'
                continue
            inside = float(roi[r.pixels].mean()) if len(r.pixels) else 0.
            if inside < self.cfg.min_inside_fraction:
                rejected[r.region_id] = 'outside_roi'
                continue
            quality[r.region_id] = inside * float(fg[r.pixels].mean()) * float((1-.5*ownership[r.pixels]).mean())
            if quality[r.region_id] <= 0:
                rejected[r.region_id] = 'zero_foreground_support'
                continue
            family = r.parent_region_id if r.parent_region_id is not None else r.region_id
            groups[family].append(r)
        selected, families, leaves, used_cells = [], {}, 0, set()
        h, w = report.shape
        candidates = list(groups.items())
        while candidates and len(set(families.values())) < self.cfg.family_budget:
            def rank(item):
                fid, rs = item
                root = next((r for r in rs if r.modality_id == 0), rs[0])
                y, x = divmod(root.anchor, w)
                cell = (x*self.cfg.grid_size//w, y*self.cfg.grid_size//h)
                return (cell not in used_cells, max(quality[r.region_id] for r in rs)*math.log1p(len(root.pixels)), -fid)
            fid, rs = max(candidates, key=rank)
            candidates.remove((fid, rs))
            # A bound aps/ori child without its selected grad parent is not a standalone group.
            if any(r.parent_region_id is not None for r in rs) and not any(r.region_id == fid for r in rs):
                rejected.update({r.region_id: 'missing_grad_parent' for r in rs})
                continue
            count = sum(len(observation_ii.semantic_nodes[r.semantic_id].related_node_ids()) for r in rs)
            if len(selected)+len(rs) > self.cfg.member_budget or leaves+count > self.cfg.leaf_budget:
                rejected.update({r.region_id: 'selection_budget' for r in rs})
                continue
            selected.extend(rs)
            families.update({r.region_id: fid for r in rs})
            leaves += count
            y, x = divmod(rs[0].anchor, w)
            used_cells.add((x*self.cfg.grid_size//w, y*self.cfg.grid_size//h))
        selected.sort(key=lambda r: (r.modality_id != 0,
            (r.anchor % w-w/2)**2+(r.anchor//w-h/2)**2, r.region_id))
        return selected, families, {'selected_leaves': leaves, 'quality': quality, 'rejected': rejected,
                                   'foreground_source': 'soft_mask' if observation.foreground_probability is not None else 'bbox_only'}


class EntityMatcher:
    """Appearance correspondence followed by one-to-one roles and full geometric verification."""
    def __init__(self, cfg):
        self.cfg, self.matcher = cfg, SpatialStructureMatcher(cfg)

    def match_object(self, entity, selected, positions, provider, gii, budget):
        roles = sorted(entity.component_edges)
        costs = np.full((len(roles), len(selected)), 1e6)
        possible_centers = set()
        for i, role in enumerate(roles):
            member = entity.component_edges[role]
            sem = gii.semantic_nodes[member['sem_id']]
            modality = provider.nodes[sem.anchor_id].modality_id
            view = region_view(sem, self.cfg)
            for j, region in enumerate(selected):
                if region.modality_id != modality:
                    continue
                budget.check(template=True)
                matches = self.matcher.evaluate(view, provider, [positions[region.region_id]], budget=budget)
                if matches and matches[0].accepted and matches[0].score >= self.cfg.graph_learn_threshold:
                    costs[i, j] = 1-matches[0].score
                    x, y = positions[region.region_id]
                    possible_centers.add((x-member['dx'], y-member['dy']))
        best = None
        view = entity_view(entity, gii, self.cfg)
        for center in sorted(possible_centers):
            budget.check()
            distance = np.array([[math.dist(positions[r.region_id],
                (center[0]+entity.component_edges[role]['dx'], center[1]+entity.component_edges[role]['dy']))
                for r in selected] for role in roles])
            allowed = (costs < 1e6) & (distance <= self.cfg.graph_geometry_update_limit)
            score_cost = np.where(allowed, costs + .25*distance/max(1., self.cfg.graph_geometry_update_limit), 1e6)
            assignment = minimum_assignment(np.concatenate((score_cost, np.full((len(roles), len(roles)), 1.)), axis=1))
            mapping = {roles[i]: selected[j].region_id for i, j in assignment.items()
                       if j < len(selected) and allowed[i, j]}
            if min(len(mapping)/max(1, len(roles)), len(mapping)/max(1, len(selected))) < self.cfg.graph_coverage_min:
                continue
            # Preserve parent/attribute family correspondence as well as modality.
            observed_families = {r.region_id: r.parent_region_id if r.parent_region_id is not None else r.region_id for r in selected}
            family_targets = defaultdict(set)
            for role, rid in mapping.items():
                family_targets[entity.component_edges[role].get('family_id', role)].add(observed_families[rid])
            if any(len(ids) > 1 for ids in family_targets.values()):
                continue
            verified = self.matcher.evaluate(view, provider, [center], budget=budget)
            if verified and verified[0].accepted and verified[0].score >= self.cfg.graph_learn_threshold:
                value = verified[0].score * min(len(mapping)/len(roles), len(mapping)/len(selected))
                if best is None or value > best[0]:
                    best = (value, center, mapping)
        return best


class SupervisedGraphOptimizer:
    def __init__(self, coordinator, classes, config=None):
        self.model, self.cfg = coordinator, coordinator.cfg
        self.config = config or SupervisedConfig()
        self.classes = tuple(classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('A nonempty fixed, unique class vocabulary is required')
        self.ledger, self.graph_version, self.split_manifest = {}, 0, None
        self.selector, self.matcher = ObjectRegionSelector(self.config), EntityMatcher(self.cfg)
        self.last_debug = None  # transient observation, never serialized

    @torch.no_grad()
    def learn_object(self, observation, features, search_budget=None, capture_debug=False):
        self.last_debug = None
        started = time.perf_counter()
        def result(status, **kwargs):
            return {'status': status, 'graph_version': self.graph_version,
                    'seconds': time.perf_counter()-started, **kwargs}
        a = observation.annotation
        try:
            a.validate(*observation.original_shape)
        except ValueError as exc:
            return result('INVALID_ANNOTATION', reason=str(exc))
        if a.class_id not in self.classes or observation.preprocessing_version != self.config.preprocessing_version:
            return result('INVALID_ANNOTATION', reason='class or preprocessing contract mismatch')
        key = observation.ledger_key
        if any(k[0] == a.episode_id and v['class_id'] != a.class_id for k, v in self.ledger.items()):
            return result('INVALID_ANNOTATION', reason='Conflicting source label: withdraw the previous template first')
        if key in self.ledger:
            entry = self.ledger[key]
            if entry['class_id'] != a.class_id or entry['bbox'] != a.bbox or entry['digest'] != observation.source_digest:
                return result('INVALID_ANNOTATION', reason='Changed annotation/source requires explicit withdrawal and version update')
            return result('ALREADY_OBSERVED', entity_id=entry['entity_id'])
        budget = search_budget or SearchBudget(self.cfg.graph_search_budget_seconds, self.cfg.graph_search_budget_templates)
        budget.start()
        inputs = prepare_feature_subspaces(features, self.cfg)
        if not inputs or any(tuple(x.shape[-2:]) != tuple(observation.valid_mask.shape) for x in inputs.values()):
            return result('INVALID_ANNOTATION', reason='feature/ROI shape mismatch or empty input')
        if self.cfg.once_node_reuse_mode != 'sample_instance':
            raise ValueError('Supervised learning requires sample_instance geometry')
        base = GraphConsolidationOptimizer(self.cfg)
        oi, oii = GmemoryI(), GmemoryII(self.cfg)
        report = base.builder.build_once(inputs, oi, oii, valid_mask=observation.valid_mask)
        selected, families, diagnostics = self.selector.select(report, observation, oii)
        if capture_debug:
            self.last_debug = {'report': report, 'selected_ids': [r.region_id for r in selected],
                               'families': dict(families), 'selection': diagnostics}
        if len(selected) < self.config.min_members:
            return result('INSUFFICIENT_STRUCTURE', selection=diagnostics)
        if any(r.budget_truncated for r in selected):
            return result('UNRESOLVED', reason='selected region sampling truncated', selection=diagnostics)
        gi, gii, giii = self.model.gmem_i, self.model.gmem_ii, self.model.gmem_iii
        provider = FeatureResponseCache(inputs, gi, self.cfg, observation.valid_mask)
        coarse = BinaryCoarseIndex(provider, gii) if self.cfg.graph_binary_coarse else None
        positions = {r.region_id: (r.anchor % report.shape[1], r.anchor // report.shape[1]) for r in selected}
        decisions, choices = {}, []
        try:
            for r in selected:
                region_view(oii.semantic_nodes[r.semantic_id], self.cfg)
                match, uncertain = base._region_match(r, report, gi, gii, provider, HierarchyQuery(), budget, coarse)
                if uncertain:
                    return result('UNRESOLVED', reason='ambiguous region', selection=diagnostics)
                decisions[r.region_id] = match
            for eid, entity in sorted(giii.entity_nodes.items()):
                metadata = getattr(entity, 'supervision', None)
                if not metadata or metadata['label_id'] != a.class_id:
                    continue
                view = entity_view(entity, gii, self.cfg)
                if any(provider.evaluable(s['node_id']) is None for s in view.slots):
                    return result('UNRESOLVED', reason='same-class template not evaluable', entity_id=eid)
                alignment = self.matcher.match_object(entity, selected, positions, provider, gii, budget)
                if alignment:
                    choices.append((alignment[0], eid, alignment))
            choices.sort(key=lambda item: (-item[0], item[1]))
            if len(choices) > 1 and choices[0][0]-choices[1][0] < self.cfg.graph_match_margin:
                return result('UNRESOLVED', reason='ambiguous same-class entity')
            budget.check()
        except SearchBudgetExceeded as exc:
            return result('UNRESOLVED', reason=str(exc), retry_required=True)
        except ValueError as exc:
            return result('UNRESOLVED', reason=f'invalid structure: {exc}')
        # A single commit owns graph mutations, provenance and the replay ledger.
        wi, wii, wiii = copy.deepcopy((gi, gii, giii))
        mapped, node_mapping = {}, {}
        episode = a.episode_id
        wiii.observations += 1
        for r in selected:
            match = decisions[r.region_id]
            if match is None:
                sem = base._copy_region(oii.semantic_nodes[r.semantic_id], oi, wi, wii, node_mapping)
            else:
                sem = wii.semantic_nodes[match.template_id]
                positions[r.region_id] = match.point
            base._update_region(sem, match, episode)
            mapped[r.region_id] = sem.node_id
        base._relations_and_proposals(report, mapped, positions, wii, episode, generate_proposals=False)
        if choices:
            _, eid, alignment = choices[0]
            entity = wiii.entity_nodes[eid]
            # Existing role identity is retained, even when current GmemII IDs differ.
            # The pre-match positions must remain in the same observation frame.
            original_positions = {r.region_id: (r.anchor % report.shape[1], r.anchor // report.shape[1]) for r in selected}
            local_cfg = copy.copy(self.cfg)
            local_cfg.graph_max_members = self.config.member_budget
            updater = GraphConsolidationOptimizer(local_cfg)
            updater._working_ii, updater._working_iii = wii, wiii
            updater._update_entity(entity, alignment, tuple(mapped), mapped, original_positions,
                                   FeatureResponseCache(inputs, wi, self.cfg, observation.valid_mask), episode, wiii.observations)
            # Transfer observed families through established role correspondences.
            observed_to_stored = {}
            for role, rid in alignment[2].items():
                if role in entity.component_edges:
                    observed_to_stored[families[rid]] = entity.component_edges[role].get('family_id', role)
            for role, member in entity.component_edges.items():
                if 'family_id' not in member:
                    candidates = [r for r in selected if mapped[r.region_id] == member['sem_id']]
                    if not candidates:
                        return result('UNRESOLVED', reason='promoted member has no current family witness')
                    r = min(candidates, key=lambda r: math.dist(original_positions[r.region_id],
                        (alignment[1][0]+member['dx'], alignment[1][1]+member['dy'])))
                    fid = families[r.region_id]
                    family = observed_to_stored.setdefault(fid, ('source_family', episode, fid))
                    member['family_id'] = family
                    entity.supervision['member_foreground_priors'][role] = diagnostics['quality'][r.region_id]
            modalities_by_family = defaultdict(set)
            for member in entity.component_edges.values():
                mid = wi.nodes[wii.semantic_nodes[member['sem_id']].anchor_id].modality_id
                modalities_by_family[member['family_id']].add(mid)
            if self.cfg.once_bind_edge_attributes and any(
                    mids & {3, 4} and 0 not in mids for mids in modalities_by_family.values()):
                return result('UNRESOLVED', reason='membership update would detach edge attributes from grad')
            if (sum(len(wii.semantic_nodes[m['sem_id']].related_node_ids()) for m in entity.component_edges.values()) > self.config.leaf_budget
                    or len({m['family_id'] for m in entity.component_edges.values()}) > self.config.family_budget):
                return result('UNRESOLVED', reason='updated entity exceeds family/leaf budget')
            status = 'MATCHED_UPDATED'
        else:
            entity = base._new_entity(tuple(r.region_id for r in selected), mapped, positions, wii, wiii, episode, wiii.observations)
            root = positions[selected[0].region_id]
            entity.supervision = {'label_id': a.class_id, 'origin': 'annotation', 'protected': True,
                'confirmation': 'annotation_confirmed', 'annotation_ids': set(), 'source_object_ids': set(),
                'canonical_box_relative_to_root': tuple(v-root[i % 2] for i, v in enumerate(observation.canonical_box)),
                'member_foreground_priors': {}, 'preprocessing_version': self.config.preprocessing_version}
            for role, r in zip(sorted(entity.component_edges), selected):
                entity.component_edges[role]['family_id'] = families[r.region_id]
                entity.supervision['member_foreground_priors'][role] = diagnostics['quality'][r.region_id]
            status = 'ANNOTATED_SEED_CREATED'
        entity.status = 'stable'
        metadata = entity.supervision
        metadata['annotation_ids'].add(a.annotation_id)
        metadata['source_object_ids'].add(episode)
        metadata['independent_support'] = entity.evidence.support
        metadata['visually_confirmed'] = (entity.evidence.support >= self.cfg.graph_stable_support and
            max((max(m['evidence'].variance()) for m in entity.component_edges.values()), default=0.) <= self.cfg.graph_stable_variance)
        entity.version += 1
        staged_ledger = dict(self.ledger)
        staged_ledger[key] = {'entity_id': entity.node_id, 'class_id': a.class_id, 'bbox': a.bbox,
                              'digest': observation.source_digest, 'annotation_id': a.annotation_id}
        self.model.gmem_i, self.model.gmem_ii, self.model.gmem_iii = wi, wii, wiii
        self.ledger = staged_ledger
        self.graph_version += 1
        return result(status, entity_id=entity.node_id, selection=diagnostics,
            independent_support=entity.evidence.support, matching_mode='appearance_assignment_translation',
            search_complete=False, novelty_proven=False,
            memory_counts=(len(wi.nodes), len(wii.semantic_nodes), len(wiii.entity_nodes)))

    def withdraw_entity(self, entity_id):
        """Explicitly withdraw the whole template; partial evidence subtraction needs rebuilding."""
        entity = self.model.gmem_iii.entity_nodes.pop(entity_id)
        for member in entity.component_edges.values():
            self.model.gmem_ii.parent_entities[member['sem_id']].discard(entity_id)
        self.ledger = {k: v for k, v in self.ledger.items() if v['entity_id'] != entity_id}
        self.graph_version += 1

    def state_dict(self):
        return {'schema_version': 1, 'graph': self.model.state_dict(), 'config': asdict(self.config),
                'classes': self.classes, 'ledger': copy.deepcopy(self.ledger), 'graph_version': self.graph_version,
                'split_manifest': copy.deepcopy(self.split_manifest), 'torch_rng': torch.get_rng_state(),
                'memory_config': {k: copy.deepcopy(getattr(self.cfg, k)) for k in dir(self.cfg)
                                  if not k.startswith('_') and not callable(getattr(self.cfg, k))}}

    def load_state_dict(self, state):
        if state.get('schema_version') != 1 or tuple(state['classes']) != self.classes or state['config'] != asdict(self.config):
            raise ValueError('Supervised checkpoint class/config/schema mismatch')
        current = {k: getattr(self.cfg, k) for k in state.get('memory_config', {}) if k != 'device'}
        if any(current[k] != state['memory_config'][k] for k in current):
            raise ValueError('Memory configuration differs from checkpoint (device may differ)')
        entities = state['graph']['gmem_iii'].entity_nodes
        if any(v['entity_id'] not in entities for v in state['ledger'].values()):
            raise ValueError('Checkpoint ledger references missing entities')
        self.model.load_state_dict(state['graph'])
        self.ledger, self.graph_version = copy.deepcopy(state['ledger']), state['graph_version']
        self.split_manifest = copy.deepcopy(state.get('split_manifest'))

    def save(self, path, trainer=None):
        path = Path(path)
        state = self.state_dict()
        state['classifier'] = trainer.state_dict() if trainer is not None else None
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path, trainer=None):
        """Only load locally trusted checkpoints: graph snapshots contain Python objects."""
        state = torch.load(path, map_location=self.cfg.device, weights_only=False)
        # Validate the optional head before replacing the live graph.
        staged_trainer = copy.deepcopy(trainer) if trainer is not None else None
        if staged_trainer is not None:
            if state.get('classifier') is None:
                raise ValueError('Checkpoint has no classifier')
            staged_trainer.load_state_dict(state['classifier'])
        self.load_state_dict(state)
        if trainer is not None:
            trainer.load_state_dict(staged_trainer.state_dict())
        return self

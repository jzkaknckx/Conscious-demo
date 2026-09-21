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


def region_family(region):
    """Evidence source, independent of whether a child copies or refines its parent."""
    parent = getattr(region, 'support_parent_region_id', None)
    if parent is None:
        parent = region.parent_region_id
    return region.region_id if parent is None else parent


class ObjectRegionSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, report, observation, observation_ii):
        roi = observation.valid_mask.detach().cpu().numpy().reshape(-1)
        fg = (observation.foreground_probability.detach().cpu().numpy().reshape(-1)
              if observation.foreground_probability is not None else np.ones(len(roi)))
        ownership = (observation.ambiguous_ownership_mask.detach().cpu().numpy().reshape(-1)
                     if observation.ambiguous_ownership_mask is not None else np.zeros(len(roi)))
        groups, rejected, quality, quality_components = defaultdict(list), {}, {}, {}
        for r in report.regions:
            if r.semantic_id is None or r.anchor is None:
                rejected[r.region_id] = 'unwritable'
                continue
            if self.cfg.skip_truncated_regions and r.budget_truncated:
                rejected[r.region_id] = 'sampling_truncated_excluded'
                continue
            inside = float(roi[r.pixels].mean()) if len(r.pixels) else 0.
            if inside < self.cfg.min_inside_fraction:
                rejected[r.region_id] = 'outside_roi'
                continue
            foreground = inside * float(fg[r.pixels].mean()) * float((1-.5*ownership[r.pixels]).mean())
            geometry = (self.cfg.incomplete_geometry_weight
                        if 'curve_order_unresolved' in r.reasons else 1.)
            sampling = 1. if r.completed else self.cfg.incomplete_geometry_weight
            quality[r.region_id] = foreground * geometry * sampling
            quality_components[r.region_id] = dict(foreground=foreground, geometry=geometry,
                                                   sampling=sampling, completed=r.completed)
            if quality[r.region_id] <= 0:
                rejected[r.region_id] = 'zero_foreground_support'
                continue
            family = region_family(r)
            groups[family].append(r)
        selected, families, leaves, used_cells = [], {}, 0, set()
        h, w = report.shape
        candidates = list(groups.items())
        # Length and area compete only after normalization within their own modality.
        extent_scale = {mid: max(math.log1p(len(r.pixels)) for rs in groups.values()
                            for r in rs if r.modality_id == mid)
                        for mid in {r.modality_id for rs in groups.values() for r in rs}}
        while candidates and len(set(families.values())) < self.cfg.family_budget:
            def rank(item):
                fid, rs = item
                root = next((r for r in rs if r.modality_id == 0), rs[0])
                y, x = divmod(root.anchor, w)
                cell = (x*self.cfg.grid_size//w, y*self.cfg.grid_size//h)
                used_modalities = {r.modality_id for r in selected}
                value = min(quality[root.region_id], max(quality[r.region_id] * math.log1p(len(r.pixels)) /
                            max(extent_scale[r.modality_id], 1e-12) *
                            (1 + self.cfg.modality_diversity_bonus * (r.modality_id not in used_modalities))
                            for r in rs))
                return (cell not in used_cells, value, -fid)
            fid, rs = max(candidates, key=rank)
            candidates.remove((fid, rs))
            # A bound aps/ori child without its selected grad parent is not a standalone group.
            if any(region_family(r) != r.region_id for r in rs) and not any(r.region_id == fid for r in rs):
                rejected.update({r.region_id: 'missing_grad_parent' for r in rs})
                continue
            root = next((r for r in rs if r.region_id == fid), rs[0])
            ordered = [root] + sorted((r for r in rs if r is not root),
                                     key=lambda r: (-quality[r.region_id], r.region_id))
            accepted = []
            for r in ordered:
                count = len(observation_ii.semantic_nodes[r.semantic_id].related_node_ids())
                if len(selected)+len(accepted)+1 > self.cfg.member_budget or leaves+count > self.cfg.leaf_budget:
                    rejected[r.region_id] = 'selection_budget'
                    if r is root:
                        rejected.update({child.region_id: 'parent_selection_budget' for child in ordered[1:]})
                        break
                    continue
                accepted.append(r)
                leaves += count
            if not accepted:
                continue
            selected.extend(accepted)
            families.update({r.region_id: fid for r in accepted})
            y, x = divmod(rs[0].anchor, w)
            used_cells.add((x*self.cfg.grid_size//w, y*self.cfg.grid_size//h))
        selected.sort(key=lambda r: (r.modality_id != 0,
            (r.anchor % w-w/2)**2+(r.anchor//w-h/2)**2, r.region_id))
        return selected, families, {'selected_leaves': leaves, 'quality': quality, 'rejected': rejected,
                                   'quality_components': quality_components,
                                   'foreground_source': 'soft_mask' if observation.foreground_probability is not None else 'bbox_only'}


class EntityMatcher:
    """Appearance correspondence followed by one-to-one roles and full geometric verification."""
    def __init__(self, cfg, selection_cfg=None):
        self.cfg, self.matcher = cfg, SpatialStructureMatcher(cfg)
        self.selection_cfg = selection_cfg or SupervisedConfig()

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
            # Preserve parent/attribute family correspondence as well as modality.
            observed_families = {r.region_id: region_family(r) for r in selected}
            family_targets = defaultdict(set)
            for role, rid in mapping.items():
                family_targets[entity.component_edges[role].get('family_id', role)].add(observed_families[rid])
            if any(len(ids) > 1 for ids in family_targets.values()):
                continue
            targets = [next(iter(ids)) for ids in family_targets.values()]
            if len(targets) != len(set(targets)):
                continue
            # Count independently aligned roots, never count an attribute fragment
            # as a separate object part or as a replacement for its parent.
            root_roles = {role for role in roles if
                provider.nodes[gii.semantic_nodes[entity.component_edges[role]['sem_id']].anchor_id].modality_id == 0
                or sum(m.get('family_id', k) == entity.component_edges[role].get('family_id', role)
                       for k, m in entity.component_edges.items()) == 1}
            root_regions = {r.region_id for r in selected if region_family(r) == r.region_id}
            aligned_roots = {role: rid for role, rid in mapping.items() if role in root_roles and rid in root_regions}
            total_weight = sum(entity.component_edges[role].get('reliability', 1.) for role in root_roles)
            stored_coverage = sum(entity.component_edges[role].get('reliability', 1.) for role in aligned_roots) / max(total_weight, 1e-12)
            observed_weights = {r.region_id: (1. if r.completed else self.selection_cfg.incomplete_geometry_weight) *
                                               (self.selection_cfg.incomplete_geometry_weight if 'curve_order_unresolved' in r.reasons else 1.) for r in selected if r.region_id in root_regions}
            observed_coverage = sum(observed_weights[rid] for rid in set(aligned_roots.values())) / max(sum(observed_weights.values()), 1e-12)
            family_coverage = min(stored_coverage, observed_coverage)
            if family_coverage < self.cfg.graph_coverage_min:
                continue
            verified = self.matcher.evaluate(view, provider, [center], budget=budget)
            if verified and verified[0].accepted and verified[0].score >= self.cfg.graph_learn_threshold:
                value = verified[0].score * family_coverage
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
        self.selector, self.matcher = ObjectRegionSelector(self.config), EntityMatcher(self.cfg, self.config)
        self.last_debug = None  # transient observation, never serialized

    @torch.no_grad()
    def learn_object(self, observation, features, search_budget=None, capture_debug=False, profile_stages=False,
                     _local_geometry_retry=False):
        self.last_debug = None
        stage_seconds, modality_counts = {}, {}
        base = None
        def clock():
            if profile_stages and torch.device(self.cfg.device).type == 'cuda':
                torch.cuda.synchronize(self.cfg.device)
            return time.perf_counter()
        started = last_stage = clock()
        active_stage = 'validation'
        def stage(name):
            nonlocal last_stage, active_stage
            now = clock()
            stage_seconds[active_stage] = stage_seconds.get(active_stage, 0.) + now-last_stage
            last_stage, active_stage = now, name
        def result(status, **kwargs):
            stage('finished')
            return {'status': status, 'graph_version': self.graph_version,
                    'seconds': time.perf_counter()-started, 'stage_seconds': dict(stage_seconds),
                    'profile_synchronized': profile_stages, 'modality_counts': modality_counts,
                    'work_counts': dict(base.work_counts) if base is not None else {}, **kwargs}
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
        if not _local_geometry_retry:
            budget.start()
        inputs = prepare_feature_subspaces(features, self.cfg)
        if not inputs or any(tuple(x.shape[-2:]) != tuple(observation.valid_mask.shape) for x in inputs.values()):
            return result('INVALID_ANNOTATION', reason='feature/ROI shape mismatch or empty input')
        if self.cfg.once_node_reuse_mode != 'sample_instance':
            raise ValueError('Supervised learning requires sample_instance geometry')
        base = GraphConsolidationOptimizer(self.cfg)
        oi, oii = GmemoryI(), GmemoryII(self.cfg)
        stage('observation_build')
        report = base.builder.build_once(inputs, oi, oii, valid_mask=observation.valid_mask)
        stage('selection')
        selected, families, diagnostics = self.selector.select(report, observation, oii)
        for mid, support in report.supports.items():
            regions = [r for r in report.regions if r.modality_id == mid]
            modality_counts[mid] = {
                'writable': int(support.writable.sum()), 'support': int(support.support.sum()),
                'regions': len(regions), 'selected': sum(r.modality_id == mid for r in selected),
                'truncated_regions': sum(r.budget_truncated for r in regions),
                'grad_eligible': int(support.gate_context.get('grad_eligible', np.zeros(report.shape)).sum()),
                'own_gate_before_grad': int(support.gate_context.get('own_gate_before_grad', support.writable).sum())}
        if capture_debug:
            self.last_debug = {'report': report, 'selected_ids': [r.region_id for r in selected],
                               'families': dict(families), 'selection': diagnostics}
        if len(selected) < self.config.min_members:
            return result('INSUFFICIENT_STRUCTURE', selection=diagnostics)
        if any(r.budget_truncated for r in selected):
            return result('UNRESOLVED', reason='selected region sampling truncated', selection=diagnostics)
        stage('region_matching')
        gi, gii, giii = self.model.gmem_i, self.model.gmem_ii, self.model.gmem_iii
        provider = FeatureResponseCache(inputs, gi, self.cfg, observation.valid_mask)
        coarse = BinaryCoarseIndex(provider, gii) if self.cfg.graph_binary_coarse else None
        positions = {r.region_id: (r.anchor % report.shape[1], r.anchor // report.shape[1]) for r in selected}
        decisions, choices, ambiguous = {}, [], {}
        unresolved_entities = []
        reuse_decisions = {}
        try:
            for r in selected:
                region_view(oii.semantic_nodes[r.semantic_id], self.cfg)
                candidates = []
                match, uncertain = base._region_match(r, report, gi, gii, provider, HierarchyQuery(), budget, coarse,
                                                       candidate_sink=candidates)
                if uncertain:
                    if len(candidates) < 2:
                        return result('UNRESOLVED', reason='invalid region template', selection=diagnostics)
                    ambiguous[r.region_id] = candidates
                decisions[r.region_id] = match
            stage('entity_matching')
            for eid, entity in sorted(giii.entity_nodes.items()):
                metadata = getattr(entity, 'supervision', None)
                if not metadata or metadata['label_id'] != a.class_id:
                    continue
                view = entity_view(entity, gii, self.cfg)
                if any(provider.evaluable(s['node_id']) is None for s in view.slots):
                    unresolved_entities.append(eid)
                    continue
                alignment = self.matcher.match_object(entity, selected, positions, provider, gii, budget)
                if alignment:
                    choices.append((alignment[0], eid, alignment))
            choices.sort(key=lambda item: (-item[0], item[1]))
            if len(choices) > 1 and choices[0][0]-choices[1][0] < self.cfg.graph_match_margin:
                unresolved_entities.extend(item[1] for item in choices)
                choices = []  # preserve a protected observation, never merge competing entities
            if unresolved_entities:
                unresolved_entities.extend(item[1] for item in choices)
                choices = []  # unevaluable competitors cannot justify a unique update
            for r in selected:
                rid = r.region_id
                candidates = ambiguous.get(rid, [])
                compatible = []
                if candidates and choices:
                    _, eid, alignment = choices[0]
                    entity = giii.entity_nodes[eid]
                    witnesses = [entity.component_edges[role] for role, mapped_rid in alignment[2].items()
                                 if mapped_rid == rid]
                    compatible = [m for m in candidates if any(
                        m.template_id == member['sem_id'] and math.dist(m.point,
                        (alignment[1][0]+member['dx'], alignment[1][1]+member['dy'])) <= self.cfg.graph_geometry_update_limit
                        for member in witnesses)]
                    if len(compatible) == 1:
                        decisions[rid] = compatible[0]
                reuse_decisions[rid] = {
                    'state': ('ambiguous_reuse' if candidates and len(compatible) != 1 else
                              'unique_reuse' if decisions[rid] is not None else 'local_observation'),
                    'candidates': [{'semantic_id': m.template_id, 'score': m.score,
                                    'point': tuple(m.point), 'version': m.version} for m in candidates]}
            # If joint reuse remains undecided, retain a new observation instead of
            # mutating a supposedly unique entity with unresolved role identities.
            if choices and any(v['state'] == 'ambiguous_reuse' for v in reuse_decisions.values()):
                unresolved_entities.extend(item[1] for item in choices)
                choices = []
            budget.check()
        except SearchBudgetExceeded as exc:
            return result('UNRESOLVED', reason=str(exc), retry_required=True)
        except ValueError as exc:
            return result('UNRESOLVED', reason=f'invalid structure: {exc}')
        if _local_geometry_retry:
            unresolved_entities.extend(item[1] for item in choices)
            choices = []
            for rid, match in decisions.items():
                if match is not None:
                    reuse_decisions[rid]['candidates'].append({'semantic_id': match.template_id,
                        'score': match.score, 'point': tuple(match.point), 'version': match.version})
                reuse_decisions[rid]['state'] = 'local_geometry_repair'
            decisions = {rid: None for rid in decisions}
        # A single commit owns graph mutations, provenance and the replay ledger.
        stage('pool_copy')
        wi, wii, wiii = copy.deepcopy((gi, gii, giii))
        stage('update_and_commit')
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
            if match is None:
                sem.observation_quality = dict(diagnostics['quality_components'][r.region_id])
                sem.reuse_provenance = {'source': episode, **reuse_decisions[r.region_id]}
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
                    q = diagnostics['quality_components'][r.region_id]
                    member['reliability'] = q['geometry'] * q['sampling']
                    member['last_quality'] = dict(q)
                    member['quality_evidence'] = {'count': 1, 'geometry_sum': q['geometry'], 'sampling_sum': q['sampling']}
                    member['is_family_root'] = families[r.region_id] == r.region_id
                    entity.supervision['member_foreground_priors'][role] = q['foreground']
            modalities_by_family = defaultdict(set)
            for member in entity.component_edges.values():
                mid = wi.nodes[wii.semantic_nodes[member['sem_id']].anchor_id].modality_id
                modalities_by_family[member['family_id']].add(mid)
            bound_modalities = {2, 3, 4} if self.cfg.once_advanced_partition_mode == 'grad_refined' else {3, 4}
            if (self.cfg.once_bind_edge_attributes or self.cfg.once_advanced_partition_mode == 'grad_refined') and any(
                    mids & bound_modalities and 0 not in mids for mids in modalities_by_family.values()):
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
                entity.supervision['member_foreground_priors'][role] = diagnostics['quality_components'][r.region_id]['foreground']
            status = 'ANNOTATED_SEED_CREATED'
        # Persist quality on the entity occurrence, not just the shared semantic node.
        role_observations = alignment[2] if choices else dict(zip(sorted(entity.component_edges),
                                                                [r.region_id for r in selected]))
        for role, rid in role_observations.items():
            if role not in entity.component_edges:
                continue
            member = entity.component_edges[role]
            q = diagnostics['quality_components'][rid]
            history = member.setdefault('quality_evidence', {'count': 0, 'geometry_sum': 0., 'sampling_sum': 0.})
            history['count'] += 1
            history['geometry_sum'] += q['geometry']
            history['sampling_sum'] += q['sampling']
            member['reliability'] = (history['geometry_sum']/history['count']) * (history['sampling_sum']/history['count'])
            member['last_quality'] = dict(q)
            member['is_family_root'] = families[rid] == rid
        entity.supervision.setdefault('pending_reuse', {})[episode] = {
            'regions': {rid: value for rid, value in reuse_decisions.items() if value['state'] in ('ambiguous_reuse', 'local_geometry_repair')},
            'entity_candidates': sorted(set(unresolved_entities))}
        entity.supervision['reuse_unresolved'] = any(
            item['regions'] or item['entity_candidates'] for item in entity.supervision['pending_reuse'].values())
        entity.status = 'stable'
        metadata = entity.supervision
        metadata['annotation_ids'].add(a.annotation_id)
        metadata['source_object_ids'].add(episode)
        metadata['independent_support'] = entity.evidence.support
        metadata['visually_confirmed'] = (not metadata['reuse_unresolved'] and entity.evidence.support >= self.cfg.graph_stable_support and
            max((max(m['evidence'].variance()) for m in entity.component_edges.values()), default=0.) <= self.cfg.graph_stable_variance)
        entity.version += 1
        write_center = tuple(alignment[1]) if choices else tuple(root)
        # A proposed template must explain its own write geometry before committing.
        # Region reuse may otherwise collapse distinct roles onto the same evidence.
        stage('write_validation')
        write_provider = FeatureResponseCache(inputs, wi, self.cfg, observation.valid_mask)
        verified_write = SpatialStructureMatcher(self.cfg).evaluate(entity_view(entity, wii, self.cfg),
                                                                    write_provider, [write_center])
        if not verified_write or not verified_write[0].accepted:
            if not _local_geometry_retry:
                repaired = self.learn_object(observation, features, search_budget=budget,
                    capture_debug=capture_debug, profile_stages=profile_stages, _local_geometry_retry=True)
                repaired['geometry_repaired'] = repaired['status'] in ('ANNOTATED_SEED_CREATED', 'MATCHED_UPDATED')
                repaired['initial_write_rejection'] = [m.summary() for m in verified_write]
                repaired['geometry_retry_prior_stage_seconds'] = dict(stage_seconds)
                repaired['seconds'] = time.perf_counter()-started
                return repaired
            return result('UNRESOLVED', reason='local observation fails write geometry validation',
                          write_validation=[m.summary() for m in verified_write])
        try:
            budget.check()
        except SearchBudgetExceeded as exc:
            return result('UNRESOLVED', reason=str(exc), retry_required=True)
        if capture_debug:
            self.last_debug['write_witness'] = {'entity_id': entity.node_id,
                'center': tuple(alignment[1]) if choices else tuple(root),
                'purpose': 'known write pose diagnostic only, never retrieval evidence'}
        staged_ledger = dict(self.ledger)
        staged_ledger[key] = {'entity_id': entity.node_id, 'class_id': a.class_id, 'bbox': a.bbox,
                              'digest': observation.source_digest, 'annotation_id': a.annotation_id}
        self.model.gmem_i, self.model.gmem_ii, self.model.gmem_iii = wi, wii, wiii
        self.ledger = staged_ledger
        self.graph_version += 1
        return result(status, entity_id=entity.node_id, selection=diagnostics,
            reuse_decisions=reuse_decisions, reuse_unresolved=metadata['reuse_unresolved'],
            independent_support=entity.evidence.support, matching_mode='appearance_assignment_translation',
            search_complete=False, novelty_proven=False,
            memory_counts=(len(wi.nodes), len(wii.semantic_nodes), len(wiii.entity_nodes)))

    @torch.no_grad()
    def diagnose_write_recall(self, observation, features, witness):
        """Compare normal retrieval with a known write pose; never mutate graph/evidence."""
        model = self.model
        eid, center = witness['entity_id'], tuple(witness['center'])
        entity = model.gmem_iii.entity_nodes[eid]
        provider = FeatureResponseCache(prepare_feature_subspaces(features, self.cfg), model.gmem_i,
                                        self.cfg, observation.valid_mask)
        matcher = SpatialStructureMatcher(self.cfg)
        entity_matches = matcher.evaluate(entity_view(entity, model.gmem_ii, self.cfg), provider, [center])
        regions = []
        for role, member in entity.component_edges.items():
            matches = matcher.evaluate(region_view(model.gmem_ii.semantic_nodes[member['sem_id']], self.cfg),
                provider, [(center[0]+member['dx'], center[1]+member['dy'])])
            regions.extend({'role': role, **m.summary()} for m in matches)
        query = model.query_hierarchy(features, valid_mask=observation.valid_mask, trace=True)
        return {'entity_id': eid, 'diagnostic_only': True,
                'known_pose_entity': [m.summary() for m in entity_matches], 'known_pose_regions': regions,
                'ordinary_target_hit': any(m.template_id == eid for m in query.entities),
                'ordinary_entities': [m.summary() for m in query.entities], 'query': query.diagnostics}

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

"""Conservative association lifecycle. Confirmation is an affinity, never an automatic graph merge."""
import copy
from collections import Counter
from .graph_memorypool_onceoptimizer import SpatialStructureMatcher, region_view

ACTIVE = {'pending', 'stale'}


def normalize_pending(entity):
    metadata = entity.supervision
    before = copy.deepcopy(metadata)
    for episode, record in metadata.get('pending_reuse', {}).items():
        for rid, item in record.get('regions', {}).items():
            if not item.get('candidates'):
                item['status'] = 'resolved_local'
                item['resolution_reason'] = 'repair_history_without_competing_association'
            else:
                item.setdefault('status', 'pending')
                for candidate in item['candidates']:
                    candidate.setdefault('status', 'pending')
                    candidate.setdefault('positive_episodes', [])
                    candidate.setdefault('negative_episodes', [])
        record.setdefault('entity_associations', [dict(entity_id=eid, status='pending')
                                                  for eid in record.get('entity_candidates', [])])
    refresh_status(entity)
    return before != metadata


def refresh_status(entity):
    entity.supervision['reuse_unresolved'] = any(
        any(item.get('status', 'pending') in ACTIVE for item in record.get('regions', {}).values()) or
        any(item.get('status', 'pending') in ACTIVE for item in record.get('entity_associations', []))
        for record in entity.supervision.get('pending_reuse', {}).values())


def register_outcome(candidate, episode, outcome, required_support):
    """Only independently sourced, comparable observations may change a hypothesis."""
    key = str(episode)
    positive = candidate.setdefault('positive_episodes', [])
    negative = candidate.setdefault('negative_episodes', [])
    if key in positive or key in negative or outcome is None: return
    (positive if outcome else negative).append(key)
    if len(positive) >= required_support and not negative:
        candidate['status'] = 'confirmed'
    elif len(negative) >= required_support and not positive:
        candidate['status'] = 'rejected'
    else:
        candidate['status'] = 'pending'


def review_regions(entity, gii, provider, center, episode, cfg):
    normalize_pending(entity)
    matcher = SpatialStructureMatcher(cfg)
    for source, record in entity.supervision.get('pending_reuse', {}).items():
        if str(source) == str(episode): continue  # re-observation of the source is not independent evidence
        for item in record.get('regions', {}).values():
            # A previously settled affinity does not survive a changed candidate template.
            for candidate in item.get('candidates', []):
                target = gii.semantic_nodes.get(candidate['semantic_id'])
                if target is None or candidate.get('version') != target.version:
                    candidate.setdefault('history', []).append({k: copy.deepcopy(candidate.get(k))
                        for k in ('version', 'status', 'positive_episodes', 'negative_episodes')})
                    candidate.update(version=target.version if target is not None else None,
                                     status='stale', positive_episodes=[], negative_episodes=[])
                    item['status'] = 'stale'
                    item.pop('confirmed_semantic_id', None)
            if item.get('status') not in ACTIVE: continue
            sid = item.get('local_semantic_id')
            roles = [m for m in entity.component_edges.values() if m['sem_id'] == sid]
            if sid not in gii.semantic_nodes or len(roles) != 1:
                item['status'] = 'stale'; item['resolution_reason'] = 'local_role_needs_reassociation'; continue
            point = (center[0]+roles[0]['dx'], center[1]+roles[0]['dy'])
            local = matcher.evaluate(region_view(gii.semantic_nodes[sid], cfg), provider, [point])[0]
            if not local.accepted: continue  # unknown, not negative
            for candidate in item['candidates']:
                target = gii.semantic_nodes.get(candidate['semantic_id'])
                if target is None:
                    candidate['status'] = 'stale'; continue
                match = matcher.evaluate(region_view(target, cfg), provider, [point])[0]
                outcome = True if match.accepted and match.score >= cfg.graph_learn_threshold else (
                    False if match.evaluable_coverage >= cfg.graph_evaluable_min and
                    match.score < cfg.graph_recall_threshold else None)
                register_outcome(candidate, episode, outcome, cfg.graph_stable_support)
            confirmed = [c for c in item['candidates'] if c['status'] == 'confirmed']
            competitors = [c for c in item['candidates'] if c['status'] not in ('rejected', 'confirmed')]
            if len(confirmed) == 1 and not competitors:
                item['status'] = 'confirmed'; item['confirmed_semantic_id'] = confirmed[0]['semantic_id']
            elif all(c['status'] == 'rejected' for c in item['candidates']):
                item['status'] = 'rejected'; item['resolution_reason'] = 'retain_local_observation'
    refresh_status(entity)


def pending_summary(entities):
    statuses = Counter()
    for entity in entities.values():
        for record in (getattr(entity, 'supervision', None) or {}).get('pending_reuse', {}).values():
            statuses.update(item.get('status', 'pending') for item in record.get('regions', {}).values())
    return {'region_states': dict(statuses), 'unresolved_entities': sum(
        (getattr(e, 'supervision', None) or {}).get('reuse_unresolved', False) for e in entities.values())}


def review_entities(entity, entities, verified_alignments, episode, cfg):
    """Positive joint affinity only; a sparse miss never rejects another entity."""
    verified = {eid: value for value, eid, _ in verified_alignments if value >= cfg.graph_learn_threshold}
    for source, record in entity.supervision.get('pending_reuse', {}).items():
        if str(source) == str(episode): continue
        for candidate in record.get('entity_associations', []):
            target = entities.get(candidate['entity_id'])
            if target is None:
                candidate['status'] = 'stale'; continue
            if candidate.get('version', target.version) != target.version:
                candidate.update(status='stale', positive_episodes=[], negative_episodes=[])
            candidate['version'] = target.version
            if candidate['entity_id'] in verified:
                register_outcome(candidate, episode, True, cfg.graph_stable_support)
    refresh_status(entity)

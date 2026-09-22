"""Small, JSON-safe evidence records; diagnostic candidates never become accepted matches."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray): return json_value(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return json_value(value.detach().cpu().tolist())
    if isinstance(value, Path): return str(value)
    return value


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def query_audit(query, prediction=None):
    d = query.diagnostics
    keys = ('mode', 'search_complete', 'assignment_search_complete', 'elapsed',
            'feature_events', 'event_budget_dropped', 'candidate_budget_dropped',
            'region_candidates', 'weak_regions', 'accepted_regions', 'entity_candidates',
            'geometry_evaluations', 'structural_fallback', 'fallback_policy', 'gate_counts')
    result = {key: d.get(key) for key in keys}
    trace = d.get('trace', [])
    result['trace'] = trace
    result['rejection_counts'] = dict(Counter(reason for row in trace
        for reason in (row.get('best') or {}).get('rejection_reasons', [])))
    result['accepted_entities'] = [m.summary() for m in query.entities]
    if prediction is not None: result['prediction'] = prediction
    if 'initial_sparse' in d:
        initial = d['initial_sparse']
        result['initial_sparse'] = {key: initial.get(key) for key in keys}
        result['initial_sparse']['trace'] = initial.get('trace', [])
    return json_value(result)


def weak_class_features(query, coordinator, classes):
    """Diagnostic class evidence, including rejected L3 hypotheses. Never authorizes graph writes."""
    rows = np.zeros((len(classes), 6), np.float32)
    rows[:, 4:] = 1.
    traces = query.diagnostics.get('trace', []) + query.diagnostics.get('initial_sparse', {}).get('trace', [])
    ranks = {}
    for row in traces:
        if row['level'] != 3 or not row.get('best'): continue
        entity = coordinator.gmem_iii.entity_nodes.get(row['template_id'])
        label = (getattr(entity, 'supervision', None) or {}).get('label_id')
        if label not in classes: continue
        c = classes.index(label); best = row['best']
        rank = best['score']*best['coverage']
        if c in ranks and rank <= ranks[c]: continue
        ranks[c] = rank
        rows[c] = (best['score'], best['evaluable'], best['coverage'],
                   best['geometry_error']/coordinator.cfg.graph_geometry_sigma,
                   float(not best['accepted']), float(not query.diagnostics.get('search_complete', False)))
    return torch.from_numpy(rows.flatten())

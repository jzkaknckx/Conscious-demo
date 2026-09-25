"""Bounded joint reassignment of colliding supported slots; never relax exclusivity."""
import math
from collections import defaultdict
import numpy as np


def collisions(slots, choices, nodes):
    occupied = defaultdict(list)
    pairs = []
    for s in slots:
        key = s['key']
        if key not in choices: continue
        score, point = choices[key]
        evidence = (nodes[s['node_id']].modality_id, tuple(round(float(x), 4) for x in point))
        for previous in occupied[evidence]:
            if math.dist(previous['xy'], s['xy']) > .5:
                pairs.append((previous['key'], key))
        occupied[evidence].append(s)
    return pairs


def repair(matcher, view, provider, center, data, index, weights, shifts, penalty, budget=None):
    cfg = matcher.cfg
    choices = {s['key']: (float(data[s['key']][0][index]), tuple(data[s['key']][2][index]))
               for s in view.slots if data[s['key']][1][index] and
               data[s['key']][0][index] >= cfg.graph_recall_threshold}
    pairs = collisions(view.slots, choices, provider.nodes)
    if not pairs: return {}, {}
    involved = {k for pair in pairs for k in pair}
    slots = {s['key']: s for s in view.slots}
    audit = {'initial_conflicts': len(pairs), 'conflict_weight': sum(weights[k] for k in involved),
             'status': 'disabled', 'search_exhaustive': False,
             'pairs': [{'slots': [str(a), str(b)], 'modality': provider.nodes[slots[a]['node_id']].modality_id,
                        'expected': [list(center+slots[a]['xy']), list(center+slots[b]['xy'])],
                        'selected': [list(choices[a][1]), list(choices[b][1])],
                        'scores': [choices[a][0], choices[b][0]]} for a,b in pairs],
             'alternatives': {}}
    if not cfg.graph_joint_assignment: return {}, audit
    neighbors = defaultdict(set)
    for a,b in pairs: neighbors[a].add(b); neighbors[b].add(a)
    remaining=set(involved);components=[]
    while remaining:
        pending=[min(remaining,key=str)];component=set()
        while pending:
            key=pending.pop()
            if key in component:continue
            component.add(key);pending.extend(neighbors[key]-component)
        remaining-=component;components.append(component)
    audit['component_sizes']=[len(c) for c in components]
    if any(len(c) > cfg.graph_assignment_max_slots for c in components):
        audit['status'] = 'slot_budget_exceeded'; return {}, audit
    options = {}
    for key in involved:
        if budget is not None: budget.check()
        s = slots[key]; points = center + s['xy'] + shifts
        response = provider.get(s['node_id'])
        values = matcher._sample(response, points).detach().cpu().numpy()*penalty
        # Preserve the independent-maximum upper bound used by progressive pruning,
        # including CPU float64 / tensor float32 rounding at the boundary.
        values = np.minimum(values, choices[key][0])
        order = np.argsort(-values, kind='stable')
        ids = [i for i in order if values[i] >= cfg.graph_recall_threshold][:cfg.graph_assignment_topk]
        options[key] = [(float(values[i]), tuple(points[i])) for i in ids]
        audit['alternatives'][str(key)] = [{'score': score, 'point': list(point)} for score,point in options[key]]
    solution = dict(choices)
    limit = 2*cfg.graph_geometry_radius+cfg.graph_cycle_tolerance
    for component in components:
        fixed = {k:v for k,v in solution.items() if k not in component}
        beam = [(0., fixed)]
        for key in sorted(component, key=lambda k:(len(options[k]), str(k))):
            if budget is not None: budget.check()
            expanded = []
            for cost, selected in beam:
                for score,point in options[key]:
                    proposed = dict(selected); proposed[key] = (score, point)
                    # Check only the newly assigned role against occupied evidence.
                    bad = any(provider.nodes[slots[j]['node_id']].modality_id == provider.nodes[slots[key]['node_id']].modality_id
                              and tuple(round(float(x),4) for x in p) == tuple(round(float(x),4) for x in point)
                              and math.dist(slots[j]['xy'], slots[key]['xy']) > .5 for j,(_,p) in selected.items())
                    if bad: continue
                    for edge in view.constraints:
                        if key not in (edge['source'],edge['target']): continue
                        a,b = proposed.get(edge['source']),proposed.get(edge['target'])
                        if a and b and np.linalg.norm(np.asarray(b[1])-a[1]-edge['delta']) > limit:
                            bad = True; break
                    if not bad: expanded.append((cost+weights[key]*math.log(max(score,1e-8)),proposed))
            expanded.sort(key=lambda x:-x[0])
            beam = expanded[:cfg.graph_assignment_beam]
            if not beam:
                audit['status'] = 'no_bounded_solution'; return {}, audit
        solution = beam[0][1]
    audit['status'] = 'reassigned'
    # Ordinary verifier recomputes coverage, score, geometry and exclusivity afterwards.
    return {k:solution[k] for k in involved}, audit

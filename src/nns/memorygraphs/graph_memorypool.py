import math
import time
import pickle
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Unified Config
# =============================
class MemoryConfig:
    H = 256
    W = 256

    # Modality bit allocations
    bitset_mod_bits = {
        0: 8,  # edge_proto
        1: 6,  # hue_proto
        2: 4,  # curvature_bank
        3: 4,  # aspect_bank
        4: 4   # orient_bank
    }
    
    # Graph I
    scales = [8, 4, 2, 1]
    base_bucket_size = 8
    graphs_per_layer = {'graphI': 5}
    bitset_bits = 64
    topk_per_scale = {8: 128, 4: 64, 2: 32, 1: 16}
    topk_per_modality = 20000
    proto_form_thresh = 3
    proto_merge_thresh = 0.7
    strengthen_threshold = 0.5
    proto_match_thresh = 0.08
    max_nodes_per_collection = 50000
    max_edges_per_node = 12
    max_candidates_per_step = 512
    max_activated_per_collection = 128
    max_writes_per_modality = 32

    # Saccade & Episode
    max_fixations_per_episode = 20
    edge_search_radius = 40
    texture_window_radius = 32
    w_grad = 0.5
    w_hue = 0.3
    w_cont = 0.2

    # Graph II
    P_global = 128
    P_local = 64
    M_centroid = 6
    K_fast = 200
    K_final = 10
    
    # Matching & Scoring
    w_jaccard = 0.55
    w_pos = 0.30
    w_pair = 0.10
    w_strength = 0.05
    graph2_match_threshold = 0.5
    promote_high = 0.7
    promote_low = 0.4

    # Node/Edge Update Rates
    graph2_strength_beta = 0.05
    graph2_pos_beta = 0.06
    graph2_edge_init_weight = 1.0
    graph2_edge_inc = 1.0
    graph2_vec_beta = 0.06
    graph2_edge_decay = 0.999
    graph2_edge_min = 0.1
    graph2_decay = 0.999
    graph2_min_strength = 0.05
    graph2_min_count = 1
    graph2_merge_thresh = 0.78

    device = torch.device('cpu')


# =============================
# Episode Manager
# =============================
class EpisodeManager:
    def __init__(self):
        self.episodes = {}
        self.next_episode_id = 1

    def create_episode(self, layer_level: int) -> int:
        eid = self.next_episode_id
        self.next_episode_id += 1
        self.episodes[eid] = {
            'layer_level': layer_level,
            'fixations': [],
            'active_protos': set(),
            'start_time': None
        }
        return eid

    def append_observation(self, episode_id: int, fixation_point: Tuple[int, int], 
                           features: List[Dict], texture_resp: float, edge_resp: List, 
                           time_now: int, active_protos: List[Tuple[int, int]]):
        ep = self.episodes.get(episode_id)
        if not ep: return
        
        if ep['start_time'] is None:
            ep['start_time'] = time_now
            
        ep['fixations'].append({
            'point': fixation_point,
            'time': time_now
        })
        ep['active_protos'].update(active_protos)

    def finalize_episode(self, episode_id: int) -> List[Dict[str, Any]]:
        ep = self.episodes.pop(episode_id, None)
        if not ep or not ep['active_protos']: 
            return []
            
        completed_hypothesis = {
            'members': list(ep['active_protos']),
            'layer_level': ep['layer_level'],
            'fixation_count': len(ep['fixations'])
        }
        return [completed_hypothesis]


# =============================
# Attention Policy (Merged Texture/Edge/Selector)
# =============================
class AttentionPolicy:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg

    def _compute_texture_resp(self, point: Tuple[int, int], image: Any) -> float:
        return 0.5 

    def _next_texture_point(self, current_point: Tuple[int, int], saccade_matrix: Dict[Tuple[int,int], bool]) -> Optional[Tuple[int, int]]:
        cx, cy = current_point
        for dx, dy in [(0, 16), (16, 0), (0, -16), (-16, 0), (16, 16), (-16, -16)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in saccade_matrix:
                return (nx, ny)
        return None

    def _query_edges(self, point: Tuple[int, int], radius: int) -> List[Any]:
        return []

    def _next_edge_point(self, current_point: Tuple[int, int], saccade_matrix: Dict[Tuple[int,int], bool]) -> Optional[Tuple[int, int]]:
        return None

    def next_point(self, current_point: Tuple[int, int], saccade_matrix: Dict[Tuple[int,int], bool], image: Any) -> Optional[Tuple[int, int]]:
        local_edges = self._query_edges(current_point, self.cfg.edge_search_radius)
        if not local_edges:
            return self._next_texture_point(current_point, saccade_matrix)
        else:
            return self._next_edge_point(current_point, saccade_matrix)

    def global_initial_point(self, image: Any) -> Tuple[int, int]:
        if hasattr(image, 'shape') and len(image.shape) >= 2:
            return (image.shape[-1] // 2, image.shape[-2] // 2)
        return (128, 128)

    def extract_responses(self, point: Tuple[int, int], image: Any) -> Tuple[float, List[Any]]:
        return self._compute_texture_resp(point, image), self._query_edges(point, 0)


# =============================
# Saccade
# =============================
class Saccade:
    """
    Encapsulates episode-based saccade logic.
    """
    def __init__(self, cfg: MemoryConfig, device=None):
        self.cfg = cfg
        self.device = device or cfg.device
        self.visited: Optional[torch.Tensor] = None   
        self.episode_manager = EpisodeManager()
        self.policy = AttentionPolicy(cfg)
        
        self.state = 'idle'
        self.current_fixation: Optional[Tuple[int, int]] = None
        self.fixation_history = deque(maxlen=cfg.max_fixations_per_episode)
        self.episode_id: Optional[int] = None
        self.saccade_matrix: Dict[Tuple[int, int], bool] = {}
        self.layer_level = 0

    def reset_visited(self, H: int, W: int):
        self.visited = torch.zeros((H, W), device=self.device, dtype=torch.uint8)
        self.saccade_matrix.clear()

    def mark_visited(self, center: Tuple[int,int], r: int):
        if self.visited is None: return
        cx, cy = int(center[0]), int(center[1])
        H, W = self.visited.shape
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)
        self.visited[y0:y1+1, x0:x1+1] = 1
        self.saccade_matrix[(cx, cy)] = True

    def start_episode(self, initial_point: Tuple[int, int], layer_level: int = 0) -> int:
        self.layer_level = layer_level
        self.episode_id = self.episode_manager.create_episode(layer_level)
        self.current_fixation = initial_point
        self.fixation_history.clear()
        self.saccade_matrix.clear()
        if self.visited is not None:
            self.visited.zero_()
        self.state = 'learn'
        return self.episode_id

    def perform_fixation(self, image_dict: Any, graphI: 'GraphI', time_now: int):
        if self.state != 'learn' or self.current_fixation is None:
            return
            
        if self.visited is None:
            # Initialize visited tensor based on first image dict if needed
            h_shape = image_dict.get('hue').shape if image_dict.get('hue') is not None else (self.cfg.H, self.cfg.W)
            self.reset_visited(h_shape[-2], h_shape[-1])

        features = graphI.extract_features(
            grad=image_dict.get('grad'), hue=image_dict.get('hue'), 
            curvature_bank=image_dict.get('curvature'), aspect_bank=image_dict.get('aspect'), orient_bank=image_dict.get('orient'),
            center=self.current_fixation, r=self.cfg.texture_window_radius, ts=time_now, visited_mask_tensor=self.visited
        )

        texture_resp, edge_resp = self.policy.extract_responses(self.current_fixation, image_dict)
        active_protos = graphI.fast_update(features, self.current_fixation, time_now)
        
        self.episode_manager.append_observation(
            self.episode_id, self.current_fixation, features, 
            texture_resp, edge_resp, time_now, active_protos
        )
        
        self.fixation_history.append((self.current_fixation, time_now))
        self.mark_visited(self.current_fixation, self.cfg.texture_window_radius)

    def decide_next_fixation(self, image_dict: Any) -> str:
        if len(self.fixation_history) >= self.cfg.max_fixations_per_episode:
            self.state = 'end'
            return 'end_episode'
            
        next_pt = self.policy.next_point(self.current_fixation, self.saccade_matrix, image_dict)
        if next_pt is None:
            self.state = 'end'
            return 'end_episode'
            
        self.current_fixation = next_pt
        return 'continue'

    def end_episode(self, graphII: 'GraphII', collections: List[Any], time_now: int):
        completed_hypotheses = self.episode_manager.finalize_episode(self.episode_id)
        for hyp in completed_hypotheses:
            graphII.batch_update_from_hypothesis(hyp, collections, time_now)
        self.state = 'idle'

    def global_search_for_initial_fixation(self, image_dict: Any) -> Tuple[int, int]:
        return self.policy.global_initial_point(image_dict)


# =============================
# Graph I related
# =============================
class Graph0Node:
    __slots__ = ('id', 'signature', 'modality', 'pos', 'value', 'count', 'last_seen')
    def __init__(self, nid:int, signature:int, modality:int, pos:Tuple[int,int], value:float, ts:int=0):
        self.id = nid
        self.signature = int(signature)
        self.modality = int(modality)
        self.pos = (int(pos[0]), int(pos[1]))
        self.value = float(value)
        self.count = 1
        self.last_seen = ts

class Graph1Proto:
    __slots__ = ('id', 'members', 'bitset', 'count', 'strength', 'last_updated')
    def __init__(self, pid:int, member_ids:List[int], graph0_nodes:Dict[int, Graph0Node], bits:int):
        self.id = pid
        self.members = list(member_ids)
        self.bitset = 0
        for nid in self.members:
            if nid in graph0_nodes:
                self.bitset |= (1 << (hash(graph0_nodes[nid].signature) & (bits - 1)))
        self.count = 1
        self.strength = 1.0
        self.last_updated = 0

class GraphCollection:
    def __init__(self, bucket_size:int, cfg:MemoryConfig):
        self.bucket_size = bucket_size
        self.cfg = cfg
        self.nodes: Dict[int, Graph0Node] = {}
        self.buckets: Dict[Tuple[int,int], List[int]] = {}
        self.edges: Dict[int, Dict[int,int]] = {}
        self.protos: Dict[int, Graph1Proto] = {}
        self._next_node_id = 1
        self._next_proto_id = 1

    def _bucket_coord(self, x:int, y:int) -> Tuple[int,int]:
        return (int(x // self.bucket_size), int(y // self.bucket_size))

    def reset_buckets(self):
        self.buckets.clear()

    def add_or_merge_node(self, signature:int, modality:int, pos:Tuple[int,int], value:float, ts:int=0) -> int:
        b = self._bucket_coord(pos[0], pos[1])
        for nid in self.buckets.get(b, []):
            n = self.nodes[nid]
            if n.signature == signature:
                n.count += 1
                n.value = 0.9 * n.value + 0.1 * value
                n.last_seen = ts
                return nid
        nid = self._next_node_id; self._next_node_id += 1
        node = Graph0Node(nid, signature, modality, pos, value, ts)
        self.nodes[nid] = node
        self.buckets.setdefault(b, []).append(nid)
        self.edges.setdefault(nid, {})
        return nid

    def add_node_no_merge(self, signature:int, modality:int, pos:Tuple[int,int], value:float, ts:int=0) -> int:
        nid = self._next_node_id; self._next_node_id += 1
        node = Graph0Node(nid, signature, modality, pos, value, ts)
        self.nodes[nid] = node
        b = self._bucket_coord(pos[0], pos[1])
        self.buckets.setdefault(b, []).append(nid)
        self.edges.setdefault(nid, {})
        return nid

    def add_proto(self, member_ids:List[int]) -> int:
        pid = self._next_proto_id; self._next_proto_id += 1
        proto = Graph1Proto(pid, member_ids, self.nodes, self.cfg.bitset_bits)
        self.protos[pid] = proto
        return pid

    def search_protos_by_postings_bitset(self, postings:List[int], topk:int=20) -> List[Tuple[int,float]]:
        qbit = 0
        for nid in postings:
            if nid in self.nodes:
                qbit |= (1 << (hash(self.nodes[nid].signature) & (self.cfg.bitset_bits - 1)))
        scored = []
        for pid, p in self.protos.items():
            common = (qbit & p.bitset)
            score = common.bit_count()
            if score > 0:
                scored.append((pid, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def ensure_edge(self, u:int, v:int, inc:int=1):
        if u == v: return
        ed = self.edges.setdefault(u, {})
        ed[v] = ed.get(v, 0) + inc
        if len(ed) > self.cfg.max_edges_per_node:
            sorted_edges = sorted(ed.items(), key=lambda kv: kv[1], reverse=True)
            keep = dict(sorted_edges[:self.cfg.max_edges_per_node])
            self.edges[u] = keep

    def prune_inactive_nodes(self):
        if len(self.nodes) <= self.cfg.max_nodes_per_collection:
            return
        items = list(self.nodes.items())
        items.sort(key=lambda kv: (kv[1].count, kv[1].last_seen))
        while len(self.nodes) > self.cfg.max_nodes_per_collection:
            nid, node = items.pop(0)
            b = self._bucket_coord(node.pos[0], node.pos[1])
            if b in self.buckets and nid in self.buckets[b]:
                self.buckets[b].remove(nid)
            if nid in self.edges: del self.edges[nid]
            for u in list(self.edges.keys()):
                if nid in self.edges[u]:
                    del self.edges[u][nid]
            if nid in self.nodes: del self.nodes[nid]
            for pid, p in list(self.protos.items()):
                if nid in p.members:
                    p.members.remove(nid)
                    bset = 0
                    for m in p.members:
                        if m in self.nodes:
                            bset |= (1 << (hash(self.nodes[m].signature) & (self.cfg.bitset_bits-1)))
                    p.bitset = bset
                    if len(p.members) == 0:
                        del self.protos[pid]


class GraphI:
    def __init__(self, cfg:MemoryConfig, device=None):
        self.cfg = cfg
        self.device = device or cfg.device
        n_col = cfg.graphs_per_layer.get('graphI', 5)
        self.graphI: List[GraphCollection] = [GraphCollection(cfg.base_bucket_size, cfg) for _ in range(n_col)]
        self.cached_feats: List[Dict[str,Any]] = []

    def reset_all_buckets(self):
        for coll in self.graphI:
            coll.reset_buckets()

    def extract_features(self,
                         grad: torch.Tensor,
                         hue: torch.Tensor,
                         curvature_bank: Optional[torch.Tensor] = None,
                         aspect_bank: Optional[torch.Tensor] = None,
                         orient_bank: Optional[torch.Tensor] = None,
                         center: Optional[Tuple[int,int]] = None,
                         r: Optional[int] = None,
                         ts: int = 0,
                         visited_mask_tensor: Optional[torch.Tensor] = None) -> List[Dict[str,Any]]:
        device = self.device
        H = int(hue.shape[-2]); W = int(hue.shape[-1])

        if center is None or r is None:
            return self._extract_global_features(grad, hue, curvature_bank, aspect_bank, orient_bank, ts)

        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)

        prev_feats = []
        for f in self.cached_feats:
            fx, fy = int(f['pos'][0]), int(f['pos'][1])
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                prev_feats.append(f)

        if visited_mask_tensor is None:
            visited_mask = torch.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=torch.bool, device=device)
        else:
            visited_patch = visited_mask_tensor[y0:y1+1, x0:x1+1]
            visited_mask = (visited_patch > 0)
        
        new_feats: List[Dict[str,Any]] = []

        # Grad
        g = grad.squeeze(0)
        if g.ndim == 4 and g.shape[-1] == 2:
            g = g.squeeze(0)
        if g.ndim == 3 and g.shape[-1] == 2:
            gx = g[..., 0].to(device); gy = g[..., 1].to(device)
        else:
            gx = torch.zeros((H, W), device=device); gy = torch.zeros_like(gx)
        mag = torch.sqrt(gx*gx + gy*gy)
        mag_patch = mag[y0:y1+1, x0:x1+1].clone()
        mag_patch[visited_mask] = float('-inf')
        flat = mag_patch.flatten()
        topk = min(self.cfg.topk_per_modality, int((mag_patch != float('-inf')).sum().item()))
        if topk > 0:
            vals, idxs = torch.topk(flat, topk)
            Wpatch = x1 - x0 + 1
            for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                if not np.isfinite(v): continue
                ry = idx // Wpatch; rx = idx % Wpatch
                absx = x0 + rx; absy = y0 + ry
                angle = math.atan2(float(gy[absy, absx].item()), float(gx[absy, absx].item()))
                binidx = int(((angle + math.pi) / (2*math.pi)) * 16) & 0xff
                sig = int((0 << 24) | binidx)
                new_feats.append({'modality': 0, 'signature': sig, 'pos': (absx, absy), 'value': float(v)})

        # Hue
        hmap = hue.squeeze(0).squeeze(0).to(device)
        hue_patch = hmap[y0:y1+1, x0:x1+1]
        k = 7; pad = k // 2
        kernel = torch.ones((1,1,k,k), device=device) / (k*k)
        hue_mean = F.conv2d(hmap[None,None], kernel, padding=pad)[0,0]
        hue_patch_mean = hue_mean[y0:y1+1, x0:x1+1]
        sal_patch = torch.abs(hue_patch - hue_patch_mean)
        sal_patch[visited_mask] = float('-inf')
        flat = sal_patch.flatten()
        topk = min(self.cfg.topk_per_modality, int((sal_patch != float('-inf')).sum().item()))
        if topk > 0:
            vals, idxs = torch.topk(flat, topk)
            Wpatch = x1 - x0 + 1
            for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                if not np.isfinite(v): continue
                ry = idx // Wpatch; rx = idx % Wpatch
                absx = x0 + rx; absy = y0 + ry
                binidx = int((hmap[absy, absx].item()) * 32) & 0xff
                sig = int((1 << 24) | binidx)
                new_feats.append({'modality': 1, 'signature': sig, 'pos': (absx, absy), 'value': float(v)})

        # Banks
        banks = [curvature_bank, aspect_bank, orient_bank]
        for mi, bank in enumerate(banks, start=2):
            if bank is None: continue
            b = bank.squeeze(0).to(device)  
            patch = b[:, y0:y1+1, x0:x1+1]  
            sal = torch.norm(patch, dim=0)
            sal[visited_mask] = float('-inf')
            flat = sal.flatten()
            topk = min(self.cfg.topk_per_modality, int((sal != float('-inf')).sum().item()))
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                Wpatch = x1 - x0 + 1
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    if not np.isfinite(v): continue
                    ry = idx // Wpatch; rx = idx % Wpatch
                    absx = x0 + rx; absy = y0 + ry
                    ch = int(torch.argmax(patch[:, ry, rx]).item())
                    sig = int((mi << 24) | (ch & 0xffff))
                    new_feats.append({'modality': mi, 'signature': sig, 'pos': (absx, absy), 'value': float(v)})

        self.cached_feats.extend(new_feats)
        return prev_feats + new_feats

    def _extract_global_features(self, grad, hue, curvature_bank, aspect_bank, orient_bank, ts:int=0) -> List[Dict[str,Any]]:
        feats = []
        device = self.device
        if grad is not None:
            g = grad.squeeze(0)
            g = g.squeeze(0)
            if g.ndim == 3 and g.shape[-1] == 2:
                gx = g[...,0].to(device); gy = g[...,1].to(device)
            else:
                gx = torch.zeros((self.cfg.H, self.cfg.W), device=device); gy = torch.zeros_like(gx)
            mag = torch.sqrt(gx*gx + gy*gy)
            flat = mag.flatten()
            topk = min(self.cfg.topk_per_modality, flat.numel())
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                H, W = mag.shape
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    ry = idx // W; rx = idx % W
                    angle = math.atan2(float(gy[ry,rx].item()), float(gx[ry,rx].item()))
                    binidx = int(((angle + math.pi) / (2*math.pi)) * 16)
                    sig = int((0 << 24) | (binidx & 0xffff))
                    feats.append({'modality': 0, 'signature': sig, 'pos': (int(rx), int(ry)), 'value': float(v)})

        if hue is not None:
            hmap = hue.squeeze(0).squeeze(0).to(device)
            H, W = hmap.shape
            sal = torch.abs(hmap - 0.5)
            flat = sal.flatten()
            topk = min(self.cfg.topk_per_modality, flat.numel())
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    ry = idx // W; rx = idx % W
                    binidx = int((hmap[ry,rx].item()) * 32)
                    sig = int((1 << 24) | (binidx & 0xffff))
                    feats.append({'modality': 1, 'signature': sig, 'pos':(int(rx),int(ry)), 'value':float(v)})

        banks = [curvature_bank, aspect_bank, orient_bank]
        for mi, bank in enumerate(banks, start=2):
            if bank is None: continue
            b = bank.squeeze(0).to(device)
            sal = torch.norm(b, dim=0)
            flat = sal.flatten()
            topk = min(self.cfg.topk_per_modality, flat.numel())
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    ry = idx // W; rx = idx % W
                    ch = int(torch.argmax(b[:,ry,rx]).item())
                    sig = int((mi << 24) | (ch & 0xffff))
                    feats.append({'modality':mi, 'signature':sig, 'pos':(int(rx),int(ry)), 'value':float(v)})
        return feats

    def retrieve_candidates(self, feats:List[Dict[str,Any]], ts:int=0):
        if len(feats) == 0:
            return set(), [], 0.0

        candidate_nodes_by_collection = [set() for _ in range(len(self.graphI))]
        for f in feats:
            mod = int(f['modality'])
            idx = mod % len(self.graphI)
            coll = self.graphI[idx]
            sig = int(f['signature'])
            x, y = int(f['pos'][0]), int(f['pos'][1])

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx = x + dx * coll.bucket_size
                    ny = y + dy * coll.bucket_size
                    b = coll._bucket_coord(nx, ny)
                    for nid in coll.buckets.get(b, []):
                        n = coll.nodes[nid]
                        if n.signature == sig:
                            candidate_nodes_by_collection[idx].add(nid)

        for idx, cset in enumerate(candidate_nodes_by_collection):
            if len(cset) > self.cfg.max_candidates_per_step:
                ranked = sorted(list(cset), key=lambda nid: self.graphI[idx].nodes[nid].count if nid in self.graphI[idx].nodes else 0, reverse=True)
                candidate_nodes_by_collection[idx] = set(ranked[:self.cfg.max_candidates_per_step])

        activated_nodes = set()
        proto_matches = []
        proto_support_total = 0.0; proto_support_count = 0
        for idx, coll in enumerate(self.graphI):
            postings = [nid for nid in candidate_nodes_by_collection[idx] if nid in coll.nodes]
            if not postings: continue
            proto_cands = coll.search_protos_by_postings_bitset(postings, topk=20)
            for pid, score in proto_cands:
                p = coll.protos.get(pid)
                if p is None: continue
                overlap = len(set(p.members).intersection(postings)) / max(1, len(p.members))
                if overlap >= self.cfg.proto_match_thresh:
                    proto_matches.append((idx, pid, float(overlap)))
                    proto_support_total += overlap
                    proto_support_count += 1
                    for nid in p.members:
                        activated_nodes.add(nid)
                        
        for idx, cset in enumerate(candidate_nodes_by_collection):
            added = 0
            for nid in cset:
                if added >= self.cfg.max_activated_per_collection: break
                activated_nodes.add(nid)
                added += 1

        proto_support_score = (proto_support_total / proto_support_count) if proto_support_count > 0 else 0.0
        node_support_score = min(1.0, len(activated_nodes) / max(1, self.cfg.max_activated_per_collection * len(self.graphI)))
        w_proto, w_node = 0.7, 0.3
        S = w_proto * proto_support_score + w_node * node_support_score
        return activated_nodes, proto_matches, float(S)

    def apply_strengthen(self, activated_nodes:set, proto_matches:List[Tuple[int,int,float]], ts:int=0):
        for idx, coll in enumerate(self.graphI):
            coll_activated = [nid for nid in activated_nodes if nid in coll.nodes]
            if not coll_activated: continue
            if len(coll_activated) > self.cfg.max_activated_per_collection:
                coll_activated = coll_activated[:self.cfg.max_activated_per_collection]
            for i in range(len(coll_activated)):
                u = coll_activated[i]
                for j in range(i+1, len(coll_activated)):
                    v = coll_activated[j]
                    nu = coll.nodes[u]; nv = coll.nodes[v]
                    dx = nu.pos[0] - nv.pos[0]; dy = nu.pos[1] - nv.pos[1]
                    if dx*dx + dy*dy <= ((coll.bucket_size * 4)**2):
                        coll.ensure_edge(u, v, inc=1)
                        coll.ensure_edge(v, u, inc=1)
            for (cidx, pid, overlap) in proto_matches:
                if cidx != idx: continue
                p = coll.protos.get(pid)
                if p is None: continue
                p.count += 1
                p.strength += 0.5 * overlap
                p.last_updated = ts
        for nid in activated_nodes:
            for coll in self.graphI:
                if nid in coll.nodes:
                    node = coll.nodes[nid]
                    node.count += 1
                    node.last_seen = ts
        
        activated_protos = []
        for (cidx, pid, overlap) in proto_matches:
            activated_protos.append((cidx, pid))
        return activated_protos

    def learn(self, feats:List[Dict[str,Any]], activated_nodes:set, proto_matches:List[Tuple[int,int,float]], ts:int=0):
        ncols = len(self.graphI)
        new_nodes_by_collection = [ [] for _ in range(ncols) ]
        created = []
        
        for f in feats:
            mod = int(f['modality']); idx = mod % ncols
            coll = self.graphI[idx]
            sig = int(f['signature'])
            x,y = int(f['pos'][0]), int(f['pos'][1])
            
            if mod in (0, 1):
                b = coll._bucket_coord(x,y)
                found = False
                for nid in coll.buckets.get(b, []):
                    n = coll.nodes[nid]
                    if n.signature == sig:
                        found = True; break
                if not found:
                    nid = coll.add_or_merge_node(sig, mod, (x,y), f['value'], ts)
                    new_nodes_by_collection[idx].append(nid)
            else:
                nid = coll.add_node_no_merge(sig, mod, (x,y), f['value'], ts)
                pid = coll.add_proto([nid])
                p = coll.protos[pid]
                p.count = 1; p.strength = 1.0; p.last_updated = ts
                created.append((idx, pid))
                
        for idx, coll in enumerate(self.graphI):
            groups: Dict[Tuple[int,int], List[int]] = {}
            for nid in new_nodes_by_collection[idx]:
                n = coll.nodes.get(nid); 
                if n is None: continue
                b = coll._bucket_coord(n.pos[0], n.pos[1])
                groups.setdefault(b, []).append(nid)
                
            for bucket_key, members in groups.items():
                if len(members) >= self.cfg.proto_form_thresh:
                    pid = coll.add_proto(members)
                    created.append((idx, pid))
                    p = coll.protos.get(pid)
                    p.count = 1; p.strength = 1.0; p.last_updated = ts
                    for other_pid, other_p in list(coll.protos.items()):
                        if other_pid == pid: continue
                        inter = (p.bitset & other_p.bitset).bit_count()
                        union = (p.bitset | other_p.bitset).bit_count()
                        ratio = inter / max(1, union)
                        if ratio >= self.cfg.proto_merge_thresh:
                            new_bset = 0
                            for m in other_p.members:
                                if m in coll.nodes:
                                    sig_m = coll.nodes[m].signature
                                    bitpos = (hash(sig_m) & (coll.cfg.bitset_bits - 1))
                                    new_bset |= (1 << bitpos)
                            other_p.bitset = new_bset
                            if pid in coll.protos: 
                                del coll.protos[pid]
                            break
        return created

    def _limit_feats_per_modality(self, feats:List[Dict[str,Any]], per_mod_limit:int) -> List[Dict[str,Any]]:
        grouped: Dict[int, List[Dict[str,Any]]] = {}
        for f in feats:
            grouped.setdefault(int(f['modality']), []).append(f)
        out = []
        for mod, fl in grouped.items():
            fl_sorted = sorted(fl, key=lambda x: x['value'], reverse=True)
            out.extend(fl_sorted[:per_mod_limit])
        return out

    def fast_update(self, features: List[Dict[str, Any]], location: Tuple[int, int], ts: int = 0) -> List[Tuple[int, int]]:
        activated_nodes, proto_matches, S = self.retrieve_candidates(features, ts)

        if S >= self.cfg.strengthen_threshold and len(activated_nodes) > 0:
            activated_protos = self.apply_strengthen(activated_nodes, proto_matches, ts)
            new_protos = []
        else:
            feats_for_learning = self._limit_feats_per_modality(features, self.cfg.max_writes_per_modality)
            new_protos = self.learn(feats_for_learning, set(), [], ts)
            activated_protos = proto_matches and [(cidx, pid) for (cidx, pid, _) in proto_matches] or []
        
        active_protos = list(dict.fromkeys(activated_protos + new_protos))
        for coll in self.graphI:
            coll.prune_inactive_nodes()
            
        return active_protos

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> 'GraphI':
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Graph II
# =============================================================================
class Graph2Node:
    def __init__(self,
                 node_id: int,
                 members: List[Tuple[int, int]],
                 bitset: int,
                 centroids: List[Tuple[float, float]],
                 ts: int,
                 cfg: MemoryConfig):
        self.id = int(node_id)
        self.members = list(members)            
        self.modalities = {c for c, _ in members}
        self.bitset = int(bitset)
        self.centroids = list(centroids)

        self.count = 1
        self.strength = 1.0
        self.last_seen = ts
        self.examples = deque(maxlen=8) 

    def update(self, event_bitset: int, event_centroids: List[Tuple[float, float]], ts: int, cfg: MemoryConfig):
        self.count += 1
        self.last_seen = ts
        self.strength = (1 - cfg.graph2_strength_beta) * self.strength + cfg.graph2_strength_beta * 1.0
        self.bitset |= int(event_bitset)

        all_c = self.centroids + event_centroids
        if len(all_c) > cfg.M_centroid:
            self.centroids = all_c[:cfg.M_centroid]
        else:
            self.centroids = all_c

    def merge_into(self, other:'Graph2Node', cfg: MemoryConfig):
        existing_set = set(self.members)
        for m in other.members:
            if m not in existing_set:
                self.members.append(m)
                existing_set.add(m)
                self.modalities.add(m[0])
                
        self.bitset |= other.bitset
        
        all_c = self.centroids + other.centroids
        self.centroids = all_c[:cfg.M_centroid]

        self.count += other.count
        self.strength = max(self.strength, other.strength) + cfg.graph2_strength_beta
        self.last_seen = max(self.last_seen, other.last_seen)
        for ex in other.examples:
            self.examples.append(ex)


class Graph2Edge:
    def __init__(self, src: int, dst: int, vec: Tuple[float, float], ts: int, cfg: MemoryConfig):
        self.src = int(src)
        self.dst = int(dst)
        self.count = 1
        self.weight = float(cfg.graph2_edge_init_weight)
        self.vec_mean = list(vec)
        self.last_seen = ts

    def update(self, vec: Tuple[float, float], ts: int, cfg: MemoryConfig):
        self.count += 1
        self.last_seen = ts
        self.weight += cfg.graph2_edge_inc
        delta_x = vec[0] - self.vec_mean[0]
        delta_y = vec[1] - self.vec_mean[1]
        self.vec_mean[0] += cfg.graph2_vec_beta * delta_x
        self.vec_mean[1] += cfg.graph2_vec_beta * delta_y


class GraphII:
    def __init__(self, cfg: Optional[MemoryConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg if cfg is not None else MemoryConfig()
        if device is not None:
            self.cfg.device = device
        self.device = self.cfg.device

        self.graph2_nodes: Dict[int, Graph2Node] = {}
        self.graph2_edges: Dict[Tuple[int, int], Graph2Edge] = {}
        self.next_graph2_id = 0

        self.inverted_index_proto2graph2: Dict[Tuple[int, int], set] = defaultdict(set)

    def _compute_spatial_bitset_and_centroids(self, activated_protos, collections):
        bitset = 0
        proto_pos_list = []
        
        for (cidx, pid) in activated_protos:
            if cidx < 0 or cidx >= len(collections): continue
            coll = collections[cidx]
            p = coll.protos.get(pid)
            if p is None: continue
            
            p_type = hash(p.bitset) % self.cfg.P_global
            
            xs = []
            ys = []
            for nid in getattr(p, 'members', []):
                n = coll.nodes.get(nid)
                if n is not None:
                    xs.append(n.pos[0])
                    ys.append(n.pos[1])
            if not xs:
                continue
            
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            proto_pos_list.append((cx, cy))
            
            nx = cx / max(1, self.cfg.W)  
            ny = cy / max(1, self.cfg.H)
            nx = min(max(nx, 0.0), 0.9999)
            ny = min(max(ny, 0.0), 0.9999)
            
            bitset |= (1 << (p_type * 21 + 0))
            
            r2 = int(ny * 2)
            c2 = int(nx * 2)
            bitset |= (1 << (p_type * 21 + 1 + r2 * 2 + c2))
            
            r4 = int(ny * 4)
            c4 = int(nx * 4)
            bitset |= (1 << (p_type * 21 + 5 + r4 * 4 + c4))
            
        centroids = proto_pos_list[:self.cfg.M_centroid]
        return bitset, centroids

    def _match_score(self, g2: Graph2Node, event_bitset: int, event_centroids: List[Tuple[float, float]]) -> float:
        inter = (g2.bitset & event_bitset).bit_count()
        union = (g2.bitset | event_bitset).bit_count()
        bit_score = inter / float(max(1, union))
        
        pos_score = 0.0
        if event_centroids and g2.centroids:
            total_rbf = 0.0
            sigma2 = 0.1
            for qx, qy in event_centroids:
                best_rbf = 0.0
                for nx, ny in g2.centroids:
                    dx = qx - nx
                    dy = qy - ny
                    rbf = math.exp(-(dx*dx + dy*dy) / sigma2)
                    if rbf > best_rbf:
                        best_rbf = rbf
                total_rbf += best_rbf
            pos_score = total_rbf / len(event_centroids)
            
        score = self.cfg.w_jaccard * bit_score + self.cfg.w_pos * pos_score + self.cfg.w_strength * g2.strength
        return float(score)

    def _update_edge(self, src: int, dst: int, vec: Tuple[float, float], ts: int):
        key = (int(src), int(dst))
        if key not in self.graph2_edges:
            self.graph2_edges[key] = Graph2Edge(src, dst, vec, ts, self.cfg)
        else:
            self.graph2_edges[key].update(vec, ts, self.cfg)

    def batch_update_from_hypothesis(self, hyp_summary: Dict[str, Any], collections: List[Any], ts: int):
        members = hyp_summary.get('members', [])
        if not members:
            return None
            
        event_bitset, event_centroids = self._compute_spatial_bitset_and_centroids(members, collections)
        if event_bitset == 0 and not event_centroids:
            return None
            
        candidate_summary = {
            'members': members,
            'bitset': event_bitset,
            'centroids': event_centroids
        }
        
        return self.soft_merge(candidate_summary, ts)

    def soft_merge(self, candidate: Dict[str, Any], ts: int) -> int:
        candidate_ids = set()
        for (c, pid) in candidate['members']:
            candidate_ids.update(self.inverted_index_proto2graph2.get((c, pid), set()))
            
        best_id = None
        best_score = 0.0
        
        if candidate_ids:
            for gid in candidate_ids:
                g2 = self.graph2_nodes.get(gid)
                if not g2: continue
                score = self._match_score(g2, candidate['bitset'], candidate['centroids'])
                if score > best_score:
                    best_score = score
                    best_id = gid

        if best_score >= self.cfg.promote_high and best_id is not None:
            node = self.graph2_nodes[best_id]
            node.update(candidate['bitset'], candidate['centroids'], ts, self.cfg)
            return node.id
            
        elif best_score >= self.cfg.promote_low and best_id is not None:
            new_id = self._create_node_from_candidate(candidate, ts)
            self._update_edge(best_id, new_id, (0.0, 0.0), ts)
            return new_id
            
        else:
            return self._create_node_from_candidate(candidate, ts)

    def _create_node_from_candidate(self, candidate: Dict[str, Any], ts: int) -> int:
        gid = self.next_graph2_id
        self.next_graph2_id += 1
        node = Graph2Node(gid, candidate['members'], candidate['bitset'], candidate['centroids'], ts, self.cfg)
        self.graph2_nodes[gid] = node
        
        for (c, pid) in candidate['members']:
            self.inverted_index_proto2graph2[(c, pid)].add(gid)
            
        return gid

    def consolidate(self):
        merge_thresh = self.cfg.graph2_merge_thresh
        visited = set()
        for gid, g in list(self.graph2_nodes.items()):
            if gid in visited:
                continue
            candidates = set()
            for (c, pid) in g.members:
                candidates.update(self.inverted_index_proto2graph2.get((c, pid), set()))
            candidates.discard(gid)
            
            for other_id in list(candidates):
                if other_id == gid or other_id not in self.graph2_nodes:
                    continue
                other = self.graph2_nodes[other_id]
                
                inter = (g.bitset & other.bitset).bit_count()
                union = (g.bitset | other.bitset).bit_count()
                bit_j = inter / float(max(1, union))
                
                pos_score = 0.0
                if g.centroids and other.centroids:
                    total_rbf = 0.0
                    sigma2 = 0.1
                    for qx, qy in g.centroids:
                        best_rbf = 0.0
                        for nx, ny in other.centroids:
                            dx = qx - nx
                            dy = qy - ny
                            rbf = math.exp(-(dx*dx + dy*dy) / sigma2)
                            if rbf > best_rbf: best_rbf = rbf
                        total_rbf += best_rbf
                    pos_score = total_rbf / len(g.centroids)
                
                sim = 0.5 * bit_j + 0.5 * pos_score
                
                if sim >= merge_thresh:
                    if g.count >= other.count:
                        g.merge_into(other, self.cfg)
                        self._remove_node(other_id)
                    else:
                        other.merge_into(g, self.cfg)
                        self._remove_node(gid)
                        visited.add(other_id)
                        break
            visited.add(gid)

    def _remove_node(self, gid: int):
        node = self.graph2_nodes.pop(gid, None)
        if node is None:
            return
        for (c, pid) in node.members:
            s = self.inverted_index_proto2graph2.get((c, pid), None)
            if s:
                s.discard(gid)
                if not s:
                    del self.inverted_index_proto2graph2[(c, pid)]
        for key in list(self.graph2_edges.keys()):
            if key[0] == gid or key[1] == gid:
                del self.graph2_edges[key]

    def decay(self):
        for gid in list(self.graph2_nodes.keys()):
            g = self.graph2_nodes[gid]
            g.strength *= self.cfg.graph2_decay
            if g.strength < self.cfg.graph2_min_strength and g.count < self.cfg.graph2_min_count:
                self._remove_node(gid)
        for key in list(self.graph2_edges.keys()):
            e = self.graph2_edges[key]
            e.weight *= self.cfg.graph2_edge_decay
            if e.weight < self.cfg.graph2_edge_min:
                del self.graph2_edges[key]


# =============================
# Multilevel Coordinator
# =============================
class MultilevelCoordinator:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.graphI = GraphI(cfg, device=self.device)
        self.graphII = GraphII(cfg, device=self.device)
        self.saccade = Saccade(cfg, device=self.device)
        
        # self.levels = [2, 1, 0] # Example: 2=object, 1=part, 0=patch
        self.levels = [2]
        self.timestep = 0

        self.current_view = None

    def handle_new_view(self, image_dict: Dict[str, Any]):
        self.timestep += 1
        
        initial_point = self.saccade.global_search_for_initial_fixation(image_dict)

        for level in self.levels:
            self.saccade.start_episode(initial_point, layer_level=level)
            
            while self.saccade.state == 'learn':
                self.saccade.perform_fixation(image_dict, self.graphI, self.timestep)
                
                action = self.saccade.decide_next_fixation(image_dict)
                self.current_view = self.saccade.current_fixation
                if action == 'end_episode':
                    self.saccade.end_episode(self.graphII, self.graphI.graphI, self.timestep)
                    break

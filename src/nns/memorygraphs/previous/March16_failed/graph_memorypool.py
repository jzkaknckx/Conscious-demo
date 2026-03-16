
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
# Config
# =============================
class Config:
    H = 256
    W = 256

    # 模态与位段分配（对外留出接口）
    bitset_mod_bits = {
        0: 8,  # edge_proto
        1: 6,  # hue_proto
        2: 4,  # curvature_bank
        3: 4,  # aspect_bank
        4: 4   # orient_bank
    }
    
    scales = [8, 4, 2, 1]
    base_bucket_size = 8

    graphs_per_layer = {'graphI': 5}
    bitset_bits = 64

    topk_per_scale = {8: 128, 4: 64, 2: 32, 1: 16}
    topk_per_modality = 20000
    proto_form_thresh = 3
    proto_merge_thresh = 0.7
    strengthen_threshold = 0.5 # 无 embedding 后分数阈值调整
    proto_match_thresh = 0.08

    max_nodes_per_collection = 50000
    max_edges_per_node = 12

    max_candidates_per_step = 512
    max_activated_per_collection = 128

    # Saccade params
    saccade_radii = [32, 64] # to enhance
    learn_saccades_N = 0
    max_writes_per_modality = 32
    interest_end_thresh = 0.02
    path_sigma = 3.0

    # interest weights
    w_grad = 0.5
    w_hue = 0.3
    w_cont = 0.2

    device = torch.device('cpu')
    
class GraphIIConfig:
    H = 256
    W = 256
    
    # 模态与位段分配（同 GraphI，用于统一维护结构）
    bitset_mod_bits = {
        0: 8,
        1: 6,
        2: 4,
        3: 4,
        4: 4
    }
    
    P_global = 128            # 全局 proto 词表大小
    P_local = 64              # 每 node 激活 proto 上限
    scales = [1, 2, 4]        # multi-scale grids (S_total = 1+4+16 = 21)
    M_centroid = 6            # 精排使用 proto 数量上限
    K_fast = 200              # 粗筛候选数
    K_final = 10              # 返回 top
    
    # formation & consolidation
    graph2_form_count = 3           
    T_coalesce = 200                
    event_merge_thresh = 0.55       
    bucket_key_n = 3                

    # matching & scoring weights (基于检索规格文档)
    w_jaccard = 0.55
    w_pos = 0.30
    w_pair = 0.10
    w_strength = 0.05
    
    graph2_match_threshold = 0.5

    # node update rates
    graph2_strength_beta = 0.05
    graph2_pos_beta = 0.06

    # edge params
    graph2_edge_init_weight = 1.0
    graph2_edge_inc = 1.0
    graph2_vec_beta = 0.06
    graph2_edge_decay = 0.999
    graph2_edge_min = 0.1

    # node decay/prune
    graph2_decay = 0.999
    graph2_min_strength = 0.05
    graph2_min_count = 1

    # consolidation merging nodes
    graph2_merge_thresh = 0.78

    device = torch.device('cpu')

# =============================
# Saccade
# =============================
class Saccade:
    """
    Encapsulates all saccade / fixation / path / interest logic.
    """
    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or cfg.device
        self.interest: Optional[torch.Tensor] = None 
        self.path: Optional[torch.Tensor] = None      
        self.visited: Optional[torch.Tensor] = None   
        self.mode: str = 'wander'
        self.current_center: Optional[Tuple[int,int]] = None
        self.learn_saccades_left: int = 0

    def reset(self, grad: torch.Tensor, hue: torch.Tensor):
        H = int(hue.shape[-2]); W = int(hue.shape[-1])
        self.path = torch.zeros((H, W), device=self.device, dtype=torch.float32)
        self.visited = torch.zeros((H, W), device=self.device, dtype=torch.uint8)
        self.interest = self._compute_interest(grad, hue)
        self.interest = torch.clamp(self.interest, 0.0, 1.0)
        self.mode = 'wander'
        self.current_center = None
        self.learn_saccades_left = 0

    def _compute_interest(self, grad: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
        g = grad.squeeze(0)
        if g.ndim == 4 and g.shape[-1] == 2:
            g = g.squeeze(0)
        if g.ndim == 3 and g.shape[-1] == 2:
            gx = g[..., 0].to(self.device); gy = g[..., 1].to(self.device)
        else:
            gx = torch.zeros((self.cfg.H, self.cfg.W), device=self.device)
            gy = torch.zeros_like(gx)
        grad_mag = torch.sqrt(gx*gx + gy*gy)

        h = hue.squeeze(0).squeeze(0).to(self.device)
        k = 7; pad = k // 2
        kernel = torch.ones((1,1,k,k), device=self.device) / (k*k)
        hue_mean = F.conv2d(h[None,None], kernel, padding=pad)[0,0]
        hue_contrast = torch.abs(h - hue_mean)

        def norm_map(t):
            tmin = float(t.min().item()); tmax = float(t.max().item())
            if tmax - tmin < 1e-9:
                return torch.zeros_like(t)
            return (t - tmin) / (tmax - tmin + 1e-12)

        gm = norm_map(grad_mag)
        hc = norm_map(hue_contrast)
        interest = self.cfg.w_grad * gm + self.cfg.w_hue * hc
        return interest.clamp(0.0, 1.0)

    def add_path_bump(self, center: Tuple[int,int], window_radius: int):
        if self.path is None:
            return
        cx, cy = int(center[0]), int(center[1])
        H, W = self.path.shape
        sigma = max(1.0, float(window_radius) * 0.5)
        radius = int(math.ceil(3.0 * sigma))
        x0 = max(0, cx - radius); x1 = min(W - 1, cx + radius)
        y0 = max(0, cy - radius); y1 = min(H - 1, cy + radius)
        if x1 < x0 or y1 < y0: return
        xs = torch.arange(x0, x1+1, device=self.device, dtype=torch.float32)
        ys = torch.arange(y0, y1+1, device=self.device, dtype=torch.float32)
        dx2 = (xs - float(cx)) ** 2
        dy2 = (ys - float(cy)) ** 2
        denom = 2.0 * (sigma * sigma + 1e-12)
        g = torch.exp(-(dy2.unsqueeze(1) + dx2.unsqueeze(0)) / denom)
        self.path[y0:y1+1, x0:x1+1] = torch.clamp(self.path[y0:y1+1, x0:x1+1] + g, 0.0, 1.0)

    def mark_visited(self, center: Tuple[int,int], r: int):
        if self.visited is None: return
        cx, cy = int(center[0]), int(center[1])
        H, W = self.visited.shape
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)
        self.visited[y0:y1+1, x0:x1+1] = 1

    def prob_map(self) -> torch.Tensor:
        return F.relu(self.interest - self.path)

    def choose_fixation(self) -> Optional[Tuple[int,int]]:
        pm = self.prob_map()
        if float(pm.max().item()) <= 0.0:
            return None
        idx = int(torch.argmax(pm.flatten()).item())
        W = pm.shape[1]
        return (idx % W, idx // W)

    def choose_radius(self, center: Tuple[int,int]) -> Optional[int]:
        cx, cy = int(center[0]), int(center[1])
        best_r = None; best_score = -1.0
        H, W = self.interest.shape
        for r in self.cfg.saccade_radii:
            x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
            y0 = max(0, cy - r); y1 = min(H - 1, cy + r)
            region = self.interest[y0:y1+1, x0:x1+1]
            if region.numel() == 0: continue
            score = float(region.sum().item()) / float(region.numel())
            if score > best_score:
                best_score = score; best_r = r
        if best_score <= 0: return None
        return best_r

    def step(self) -> Tuple[str, Optional[Tuple[int,int]], Optional[int]]:
        if self.mode == 'wander':
            center = self.choose_fixation()
            if center is None:
                self.mode = 'end'
                return 'end', None, None
            r = self.choose_radius(center)
            if r is None:
                return 'wander', center, None
            self.current_center = center
            self.learn_saccades_left = self.cfg.learn_saccades_N
            self.add_path_bump(center, r)
            self.mark_visited(center, r)
            return 'wander', center, r

        elif self.mode == 'learn':
            if self.learn_saccades_left <= 0:
                self.mode = 'wander'
                self.current_center = None
                return 'wander', None, None
            center = self.current_center
            if center is None:
                center = self.choose_fixation()
                if center is None:
                    self.mode = 'end'
                    return 'end', None, None
                self.current_center = center
            r = self.choose_radius(center)
            if r is None:
                self.mode = 'wander'
                self.current_center = None
                return 'wander', None, None
            self.add_path_bump(center, r)
            self.mark_visited(center, r)
            self.learn_saccades_left -= 1
            return 'learn', center, r
        
        else:
            return 'end', None, None


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
    def __init__(self, bucket_size:int, cfg:Config):
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
        """ 清空当前已有的所有 bucket """
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
        """ 用于 Bank 模态直接作为特征插入，而不进行 buckets 中的相似聚合 """
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
    def __init__(self, cfg:Config, device=None):
        self.cfg = cfg
        self.device = device or cfg.device

        n_col = cfg.graphs_per_layer.get('graphI', 5)
        self.graphI: List[GraphCollection] = [GraphCollection(cfg.base_bucket_size, cfg) for _ in range(n_col)]

        self.saccade = Saccade(cfg = self.cfg, device=self.device)
        self.cached_feats: List[Dict[str,Any]] = []

    def reset_image(self, grad, hue, curvature_bank=None, aspect_bank=None, orient_bank=None):
        self.cached_feats = []
        self.saccade.reset(grad, hue)

    def reset_all_buckets(self):
        """ 清空所有 GraphI 中 GraphCollection 的 buckets 聚合缓存 """
        for coll in self.graphI:
            coll.reset_buckets()

    def propose_window(self) -> Tuple[str, Optional[Tuple[int,int]], Optional[int]]:
        return self.saccade.step()

    def extract_features(self,
                         grad: torch.Tensor,
                         hue: torch.Tensor,
                         curvature_bank: Optional[torch.Tensor] = None,
                         aspect_bank: Optional[torch.Tensor] = None,
                         orient_bank: Optional[torch.Tensor] = None,
                         center: Optional[Tuple[int,int]] = None,
                         r: Optional[int] = None,
                         ts: int = 0) -> List[Dict[str,Any]]:
        device = self.device
        H = int(hue.shape[-2]); W = int(hue.shape[-1])

        if center is None or r is None:
            feats = self._extract_global_features(grad, hue, curvature_bank, aspect_bank, orient_bank, ts)
            self.cached_feats = feats
            if self.saccade.visited is not None:
                self.saccade.visited[:, :] = 1
            return feats

        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)

        prev_feats = []
        for f in self.cached_feats:
            fx, fy = int(f['pos'][0]), int(f['pos'][1])
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                prev_feats.append(f)

        if self.saccade.visited is None:
            unvisited_patch = torch.ones((y1 - y0 + 1, x1 - x0 + 1), dtype=torch.bool, device=device)
        else:
            visited_patch = self.saccade.visited[y0:y1+1, x0:x1+1]
            unvisited_patch = (visited_patch == 0)
        
        # visited_mask = ~unvisited_patch
        visited_mask = unvisited_patch
        new_feats: List[Dict[str,Any]] = []

        g = grad.squeeze(0)
        if g.ndim == 4 and g.shape[-1] == 2:
            g = g.squeeze(0)
        if g.ndim == 3 and g.shape[-1] == 2:
            gx = g[..., 0].to(device); gy = g[..., 1].to(device)
        else:
            gx = torch.zeros((H, W), device=device); gy = torch.zeros_like(gx)
        mag = torch.sqrt(gx*gx + gy*gy)
        mag_patch = mag[y0:y1+1, x0:x1+1].clone()
        mag_patch_masked = mag_patch.clone()
        mag_patch_masked[visited_mask] = float('-inf')
        flat = mag_patch_masked.flatten()
        topk = min(self.cfg.topk_per_modality, int((mag_patch_masked != float('-inf')).sum().item()))
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
                fdict = {'modality': 0, 'signature': sig, 'pos': (absx, absy), 'value': float(v)}
                new_feats.append(fdict)

        hmap = hue.squeeze(0).squeeze(0).to(device)
        hue_patch = hmap[y0:y1+1, x0:x1+1]
        k = 7; pad = k // 2
        kernel = torch.ones((1,1,k,k), device=device) / (k*k)
        hue_mean = F.conv2d(hmap[None,None], kernel, padding=pad)[0,0]
        hue_patch_mean = hue_mean[y0:y1+1, x0:x1+1]
        sal_patch = torch.abs(hue_patch - hue_patch_mean)
        sal_patch_masked = sal_patch.clone()
        sal_patch_masked[visited_mask] = float('-inf')
        flat = sal_patch_masked.flatten()
        topk = min(self.cfg.topk_per_modality, int((sal_patch_masked != float('-inf')).sum().item()))
        if topk > 0:
            vals, idxs = torch.topk(flat, topk)
            Wpatch = x1 - x0 + 1
            for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                if not np.isfinite(v): continue
                ry = idx // Wpatch; rx = idx % Wpatch
                absx = x0 + rx; absy = y0 + ry
                binidx = int((hmap[absy, absx].item()) * 32) & 0xff
                sig = int((1 << 24) | binidx)
                fdict = {'modality': 1, 'signature': sig, 'pos': (absx, absy), 'value': float(v)}
                new_feats.append(fdict)

        banks = [curvature_bank, aspect_bank, orient_bank]
        for mi, bank in enumerate(banks, start=2):
            if bank is None: continue
            b = bank.squeeze(0).to(device)  
            patch = b[:, y0:y1+1, x0:x1+1]  
            sal = torch.norm(patch, dim=0)
            sal_masked = sal.clone()
            sal_masked[visited_mask] = float('-inf')
            flat = sal_masked.flatten()
            topk = min(self.cfg.topk_per_modality, int((sal_masked != float('-inf')).sum().item()))
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                Wpatch = x1 - x0 + 1
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    if not np.isfinite(v): continue
                    ry = idx // Wpatch; rx = idx % Wpatch
                    absx = x0 + rx; absy = y0 + ry
                    ch = int(torch.argmax(patch[:, ry, rx]).item())
                    sig = int((mi << 24) | (ch & 0xffff))
                    fdict = {'modality': mi, 'signature': sig, 'pos': (absx, absy), 'value': float(v)}
                    new_feats.append(fdict)

        self.cached_feats.extend(new_feats)
        self.saccade.mark_visited(center, r)
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

        # 取消了 embedding，通过空间网格查询相同 signature 的节点做粗筛
        candidate_nodes_by_collection = [set() for _ in range(len(self.graphI))]
        for f in feats:
            mod = int(f['modality'])
            idx = mod % len(self.graphI)
            coll = self.graphI[idx]
            sig = int(f['signature'])
            x, y = int(f['pos'][0]), int(f['pos'][1])

            # 基于位置的快速搜索
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
        
        # 1. Edge & Hue 走聚合，Banks 绕过直接创建 proto
        for f in feats:
            mod = int(f['modality']); idx = mod % ncols
            coll = self.graphI[idx]
            sig = int(f['signature'])
            x,y = int(f['pos'][0]), int(f['pos'][1])
            
            if mod in (0, 1):
                # 聚合逻辑
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
                # Bank 数据不需要聚合直接映射为 graph1proto
                nid = coll.add_node_no_merge(sig, mod, (x,y), f['value'], ts)
                pid = coll.add_proto([nid])
                p = coll.protos[pid]
                p.count = 1; p.strength = 1.0; p.last_updated = ts
                created.append((idx, pid))
                
        # 2. 只有模态为 0/1 时生成的 nodes 在这里才会走 bucket aggregation 形成 proto
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
                    # merge if similar
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

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> 'GraphI':
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Graph II (基于文档：Bitset 粗筛 + 位置精排进行检索匹配)
# =============================================================================
class Graph2Node:
    def __init__(self,
                 node_id: int,
                 members: List[Tuple[int, int]],
                 bitset: int,
                 centroids: List[Tuple[float, float]],
                 ts: int,
                 cfg: GraphIIConfig):
        self.id = int(node_id)
        self.members = list(members)            
        self.modalities = {c for c, _ in members}
        self.bitset = int(bitset)
        self.centroids = list(centroids)

        self.count = 1
        self.strength = 1.0
        self.last_seen = ts
        self.examples = deque(maxlen=8) 

    def update(self, event_bitset: int, event_centroids: List[Tuple[float, float]], ts: int, cfg: GraphIIConfig):
        self.count += 1
        self.last_seen = ts
        self.strength = (1 - cfg.graph2_strength_beta) * self.strength + cfg.graph2_strength_beta * 1.0
        self.bitset |= int(event_bitset)

        all_c = self.centroids + event_centroids
        if len(all_c) > cfg.M_centroid:
            self.centroids = all_c[:cfg.M_centroid]
        else:
            self.centroids = all_c

    def merge_into(self, other:'Graph2Node', cfg: GraphIIConfig):
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
    def __init__(self, src: int, dst: int, vec: Tuple[float, float], ts: int, cfg: GraphIIConfig):
        self.src = int(src)
        self.dst = int(dst)
        self.count = 1
        self.weight = float(cfg.graph2_edge_init_weight)
        self.vec_mean = list(vec)
        self.last_seen = ts

    def update(self, vec: Tuple[float, float], ts: int, cfg: GraphIIConfig):
        self.count += 1
        self.last_seen = ts
        self.weight += cfg.graph2_edge_inc
        delta_x = vec[0] - self.vec_mean[0]
        delta_y = vec[1] - self.vec_mean[1]
        self.vec_mean[0] += cfg.graph2_vec_beta * delta_x
        self.vec_mean[1] += cfg.graph2_vec_beta * delta_y


class GraphII:
    """
    GraphII implements candidate-cache + consolidation for cross-modal graph2.
    """
    def __init__(self, cfg: Optional[GraphIIConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg if cfg is not None else GraphIIConfig()
        if device is not None:
            self.cfg.device = device
        self.device = self.cfg.device

        self.graph2_nodes: Dict[int, Graph2Node] = {}
        self.graph2_edges: Dict[Tuple[int, int], Graph2Edge] = {}
        self.next_graph2_id = 0

        self.inverted_index_proto2graph2: Dict[Tuple[int, int], set] = defaultdict(set)
        self.cooccur_cache: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

        self.prev_active: Optional[int] = None
        self.prev_center: Optional[Tuple[float, float]] = None

    def _make_bucket_key(self, members: List[Tuple[int, int]]) -> Tuple:
        tokens = [f"{c}:{pid}" for (c, pid) in members]
        tokens.sort()
        prefix = tuple(tokens[:self.cfg.bucket_key_n])
        return prefix

    def _compute_spatial_bitset_and_centroids(self, activated_protos, collections):
        """ 遵循检索文档规范：按多尺度空间计算位特征及生成 centroid 核信息用于后续二阶精排 """
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
            
            # S_total = 21 (1x1: 1, 2x2: 4, 4x4: 16)
            bitset |= (1 << (p_type * 21 + 0))
            
            r2 = int(ny * 2)
            c2 = int(nx * 2)
            bitset |= (1 << (p_type * 21 + 1 + r2 * 2 + c2))
            
            r4 = int(ny * 4)
            c4 = int(nx * 4)
            bitset |= (1 << (p_type * 21 + 5 + r4 * 4 + c4))
            
        centroids = proto_pos_list[:self.cfg.M_centroid]
        return bitset, centroids

    def _record_similarity(self, recA: Dict, recB: Dict) -> float:
        a = int(recA['bitset']); b = int(recB['bitset'])
        inter = (a & b).bit_count()
        union = (a | b).bit_count()
        bit_j = inter / float(max(1, union))

        cA = recA.get('centroids', [])
        cB = recB.get('centroids', [])
        pos_score = 0.0
        if cA and cB:
            total_rbf = 0.0
            sigma2 = 0.1
            for qx, qy in cA:
                best_rbf = 0.0
                for nx, ny in cB:
                    dx = qx - nx
                    dy = qy - ny
                    rbf = math.exp(-(dx*dx + dy*dy) / sigma2)
                    if rbf > best_rbf: best_rbf = rbf
                total_rbf += best_rbf
            pos_score = total_rbf / len(cA)

        modsA = {c for (c, _) in recA['members']}
        modsB = {c for (c, _) in recB['members']}
        mod_overlap = len(modsA & modsB) / float(max(1, len(modsA | modsB)))

        score = (self.cfg.w_jaccard * bit_j) + (self.cfg.w_pos * pos_score) + (self.cfg.w_pair * mod_overlap)
        return float(score)

    def _insert_event_into_bucket(self, bucket_key: Any, event: Dict, ts: int) -> Dict:
        bucket = self.cooccur_cache[bucket_key]
        for rec in bucket:
            if ts - rec['first_ts'] > self.cfg.T_coalesce:
                continue
            sim = self._record_similarity(rec, event)
            if sim >= self.cfg.event_merge_thresh:
                rec['count'] += 1
                rec['last_ts'] = ts
                rec['bitset'] |= int(event['bitset'])
                
                all_c = rec.get('centroids', []) + event.get('centroids', [])
                rec['centroids'] = all_c[:self.cfg.M_centroid]
                
                existing = set(rec['members'])
                for m in event['members']:
                    if m not in existing:
                        rec['members'].append(m)
                        existing.add(m)
                return rec

        newrec = {
            'members': list(event['members']),
            'bitset': int(event['bitset']),
            'centroids': list(event.get('centroids', [])),
            'first_ts': ts,
            'last_ts': ts,
            'count': 1
        }
        bucket.append(newrec)
        return newrec

    def _try_promote_bucket_record(self, bucket_key: Any, rec: Dict, ts: int):
        if rec['count'] < self.cfg.graph2_form_count:
            return None
        age = ts - rec['first_ts']
        if age > self.cfg.T_coalesce:
            return None
            
        gid = self.next_graph2_id
        self.next_graph2_id += 1
        node = Graph2Node(gid, rec['members'], rec['bitset'], rec['centroids'], ts, self.cfg)
        self.graph2_nodes[gid] = node
        
        for (c, pid) in rec['members']:
            self.inverted_index_proto2graph2[(c, pid)].add(gid)
            
        bucket = self.cooccur_cache[bucket_key]
        for i, r in enumerate(bucket):
            if r is rec:
                del bucket[i]
                break
        return node

    def observe(self, activated_protos: List[Tuple[int, int]], collections: List[Any], ts: int) -> Optional[int]:
        if not activated_protos:
            self.prev_active = None
            self.prev_center = None
            return None

        event_bitset, event_centroids = self._compute_spatial_bitset_and_centroids(activated_protos, collections)
        if event_bitset == 0 and not event_centroids:
            return None

        members = []
        for (cidx, pid) in activated_protos:
            if 0 <= cidx < len(collections):
                members.append((cidx, pid))

        if not members:
            return None

        # 1) attempt to match existing graph2 nodes via inverted index -> small candidate set
        candidate_ids = set()
        for (c, pid) in members:
            candidate_ids.update(self.inverted_index_proto2graph2.get((c, pid), set()))
            
        best_id = None; best_score = 0.0
        if candidate_ids:
            for gid in candidate_ids:
                g2 = self.graph2_nodes.get(gid, None)
                if g2 is None: continue
                score = self._match_score(g2, event_bitset, event_centroids)
                if score > best_score:
                    best_score, best_id = score, gid

        # decide match vs candidate creation
        active_node_id = None
        if best_score >= self.cfg.graph2_match_threshold and best_id is not None:
            node = self.graph2_nodes[best_id]
            node.update(event_bitset, event_centroids, ts, self.cfg)
            active_node_id = best_id
        else:
            # no strong existing match -> insert event into cooccur cache bucket and possibly promote
            bucket_key = self._make_bucket_key(members)
            event = {'members': list(members), 'bitset': event_bitset, 'centroids': event_centroids}
            merged_rec = self._insert_event_into_bucket(bucket_key, event, ts)
            promoted = self._try_promote_bucket_record(bucket_key, merged_rec, ts)
            if promoted is not None:
                active_node_id = promoted.id

        # 2) update directed transition edges using prev_active -> active_node_id
        if (self.prev_active is not None) and (active_node_id is not None) and (self.prev_active != active_node_id):
            if event_centroids:
                cx = sum(c[0] for c in event_centroids) / len(event_centroids)
                cy = sum(c[1] for c in event_centroids) / len(event_centroids)
                cur_centroid = (cx, cy)
            else:
                cur_centroid = (0.0, 0.0)
                
            prev_centroid = self.prev_center if self.prev_center is not None else cur_centroid
            vec = (cur_centroid[0] - prev_centroid[0], cur_centroid[1] - prev_centroid[1])
            self._update_edge(self.prev_active, active_node_id, vec, ts)

        # update prev_active / prev_center if we have active node
        if active_node_id is not None:
            if event_centroids:
                cx = sum(c[0] for c in event_centroids) / len(event_centroids)
                cy = sum(c[1] for c in event_centroids) / len(event_centroids)
                self.prev_center = (cx, cy)
            else:
                self.prev_center = (0.0, 0.0)
            self.prev_active = active_node_id
            
        return active_node_id

    def _match_score(self, g2: Graph2Node, event_bitset: int, event_centroids: List[Tuple[float, float]]) -> float:
        # 1. 粗筛：按位计算 Jaccard similarity
        inter = (g2.bitset & event_bitset).bit_count()
        union = (g2.bitset | event_bitset).bit_count()
        bit_score = inter / float(max(1, union))
        
        # 2. 精排：基于 Centroid Mahalanobis 核 (RBF 近似) 的分布一致性打分
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
# FoveatedMemory (orchestrator)
# =============================
class FoveatedMemory:
    def __init__(self, cfg:Config = Config(), g2cfg:Config = GraphIIConfig()):
        self.cfg = cfg
        self.device = cfg.device
        self.graphI = GraphI(cfg, device=self.device)
        self.graphII = GraphII(g2cfg, device=self.device)
        self.timestep = 0

    def run_one_step(self,
                     grad: torch.Tensor,
                     hue: torch.Tensor,
                     curvature_bank: Optional[torch.Tensor] = None,
                     aspect_bank: Optional[torch.Tensor] = None,
                     orient_bank: Optional[torch.Tensor] = None,
                     reset_image: bool = False,
                     ts: Optional[int] = None):
        if ts is None:
            ts = self.timestep
        self.timestep += 1

        if reset_image:
            self.graphI.reset_image(grad, hue, curvature_bank, aspect_bank, orient_bank)

        mode_sacc, center, r = self.graphI.propose_window()
        if mode_sacc == 'end':
            return 'end',None, None, None

        if center is None or r is None:
            return 'wander', None, None, None

        window_feats = self.graphI.extract_features(grad, hue, curvature_bank, aspect_bank, orient_bank, center=center, r=r, ts=ts)
        activated_nodes, proto_matches, S = self.graphI.retrieve_candidates(window_feats, ts)

        if S >= self.cfg.strengthen_threshold and len(activated_nodes) > 0:
            self.graphI.saccade.mode = 'wander'
            activated_protos = self.graphI.apply_strengthen(activated_nodes, proto_matches, ts)
            new_protos = []
        else:
            self.graphI.saccade.mode = 'learn'
            feats_for_learning = self.graphI._limit_feats_per_modality(window_feats, self.cfg.max_writes_per_modality)
            new_protos = self.graphI.learn(feats_for_learning, set(), [], ts)
            activated_protos = proto_matches and [(cidx,pid) for (cidx,pid,_) in proto_matches] or []
        
        active_protos = []
        if activated_protos:
            active_protos.extend(activated_protos)
        if new_protos:
            active_protos.extend(new_protos)
            
        active_protos = list(dict.fromkeys(active_protos))
        self.graphII.observe(active_protos, self.graphI.graphI, ts)
        
        for coll in self.graphI.graphI:
            coll.prune_inactive_nodes()

        if self.graphI.saccade.mode == 'strengthen':
            return self.graphI.saccade.mode, len(activated_nodes), len(active_protos), center
        elif self.graphI.saccade.mode == 'learn':
            return self.graphI.saccade.mode, len(feats_for_learning), len(new_protos), center
        else:
            return self.graphI.saccade.mode, None, None, center
    
    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> 'FoveatedMemory':
        with open(path, 'rb') as f:
            return pickle.load(f)
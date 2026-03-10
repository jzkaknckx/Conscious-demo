# foveated_hierarchical_memory_refactored.py

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

    descriptor_C = 32
    scales = [8, 4, 2, 1]
    base_bucket_size = 8

    graphs_per_layer = {'graphI': 5}
    bitset_bits = 64

    topk_per_scale = {8: 128, 4: 64, 2: 32, 1: 16}
    topk_per_modality = 20000
    proto_form_thresh = 3
    proto_merge_thresh = 0.7
    strengthen_threshold = 8 # to enhance
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

    # GraphII params (defaults used by GraphII)
    # formation & consolidation
    graph2_form_count = 3           # promote candidate->node when count >= this
    T_coalesce = 200                # time window (in ts units) for promotion
    event_merge_thresh = 0.55       # merge threshold between events within same bucket
    bucket_key_n = 3                # number of proto tokens used in bucket key generation

    # matching & scoring weights
    w_g2_bit = 0.80
    w_g2_emb = 0.00
    w_g2_mod = 0.20
    graph2_match_threshold = 0.5

    # node update rates
    graph2_strength_beta = 0.05
    graph2_embed_beta = 0.06
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

    # embedding dimensionality (pseudo-embedding if needed)
    descriptor_C = 32

    device = torch.device('cpu')
    
class GraphIIConfig:
    # formation & consolidation
    graph2_form_count = 3           # promote candidate->node when count >= this
    T_coalesce = 200                # time window (in ts units) for promotion
    event_merge_thresh = 0.55       # merge threshold between events within same bucket
    bucket_key_n = 3                # number of proto tokens used in bucket key generation

    # matching & scoring weights
    w_g2_bit = 0.45
    w_g2_emb = 0.35
    w_g2_mod = 0.20
    graph2_match_threshold = 0.5

    # node update rates
    graph2_strength_beta = 0.05
    graph2_embed_beta = 0.06
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

    # embedding dimensionality (pseudo-embedding if needed)
    descriptor_C = 32

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

        # per-image state
        self.interest: Optional[torch.Tensor] = None  # H x W
        self.path: Optional[torch.Tensor] = None      # H x W
        self.visited: Optional[torch.Tensor] = None   # H x W binary: 1 = already observed
        # mode & counters
        self.mode: str = 'wander'
        self.current_center: Optional[Tuple[int,int]] = None
        self.learn_saccades_left: int = 0

    # --------------------------
    # reset on new image
    # --------------------------
    def reset(self, grad: torch.Tensor, hue: torch.Tensor):
        """Initialize interest/path/visited for a new image. Use grad/hue to compute interest."""
        H = int(hue.shape[-2]); W = int(hue.shape[-1])
        self.path = torch.zeros((H, W), device=self.device, dtype=torch.float32)
        self.visited = torch.zeros((H, W), device=self.device, dtype=torch.uint8)
        self.interest = self._compute_interest(grad, hue)
        # clip to [0,1]
        self.interest = torch.clamp(self.interest, 0.0, 1.0)
        self.mode = 'wander'
        self.current_center = None
        self.learn_saccades_left = 0

    # --------------------------
    # compute interest field
    # --------------------------
    def _compute_interest(self, grad: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
        """
        Compute interest from grad and hue. Returns (H,W) float tensor on device.
        """
        # grad shape handling
        g = grad.squeeze(0)
        if g.ndim == 4 and g.shape[-1] == 2:
            g = g.squeeze(0)
        if g.ndim == 3 and g.shape[-1] == 2:
            gx = g[..., 0].to(self.device); gy = g[..., 1].to(self.device)
        else:
            gx = torch.zeros((self.cfg.H, self.cfg.W), device=self.device)
            gy = torch.zeros_like(gx)
        grad_mag = torch.sqrt(gx*gx + gy*gy)

        # hue contrast
        h = hue.squeeze(0).squeeze(0).to(self.device)
        k = 7; pad = k // 2
        kernel = torch.ones((1,1,k,k), device=self.device) / (k*k)
        hue_mean = F.conv2d(h[None,None], kernel, padding=pad)[0,0]
        hue_contrast = torch.abs(h - hue_mean)

        # normalize each map to [0,1]
        def norm_map(t):
            tmin = float(t.min().item()); tmax = float(t.max().item())
            if tmax - tmin < 1e-9:
                return torch.zeros_like(t)
            return (t - tmin) / (tmax - tmin + 1e-12)

        gm = norm_map(grad_mag)
        hc = norm_map(hue_contrast)
        # simple interest = weighted sum
        interest = self.cfg.w_grad * gm + self.cfg.w_hue * hc
        return interest.clamp(0.0, 1.0)

    # --------------------------
    # add path bump (radius proportional to window_radius)
    # --------------------------
    def add_path_bump(self, center: Tuple[int,int], window_radius: int):
        """Add a Gaussian bump to path. Gaussian sigma proportional to window_radius."""
        if self.path is None:
            return
        cx, cy = int(center[0]), int(center[1])
        H, W = self.path.shape
        # sigma proportional to window_radius
        sigma = max(1.0, float(window_radius) * 0.5)
        radius = int(math.ceil(3.0 * sigma))
        # clip bounding box
        x0 = max(0, cx - radius); x1 = min(W - 1, cx + radius)
        y0 = max(0, cy - radius); y1 = min(H - 1, cy + radius)
        if x1 < x0 or y1 < y0:
            return
        xs = torch.arange(x0, x1+1, device=self.device, dtype=torch.float32)
        ys = torch.arange(y0, y1+1, device=self.device, dtype=torch.float32)
        dx2 = (xs - float(cx)) ** 2
        dy2 = (ys - float(cy)) ** 2
        denom = 2.0 * (sigma * sigma + 1e-12)
        g = torch.exp(-(dy2.unsqueeze(1) + dx2.unsqueeze(0)) / denom)
        # add and clamp
        self.path[y0:y1+1, x0:x1+1] = torch.clamp(self.path[y0:y1+1, x0:x1+1] + g, 0.0, 1.0)

    # --------------------------
    # mark visited window (set visited=1 for the whole square)
    # --------------------------
    def mark_visited(self, center: Tuple[int,int], r: int):
        """Mark the entire 2r x 2r (square) window centered at center as visited (1)."""
        if self.visited is None:
            return
        cx, cy = int(center[0]), int(center[1])
        H, W = self.visited.shape
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)
        self.visited[y0:y1+1, x0:x1+1] = 1

    # --------------------------
    # get unvisited mask in window (boolean)
    # --------------------------
    def unvisited_mask_in_window(self, center: Tuple[int,int], r: int) -> torch.Tensor:
        """Return boolean mask for unvisited pixels inside the window (Hwin x Wwin) on device."""
        cx, cy = int(center[0]), int(center[1])
        H, W = self.visited.shape
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)
        patch = self.visited[y0:y1+1, x0:x1+1]
        # return True where not visited
        return (patch == 0)

    # --------------------------
    # compute prob map and choose fixation
    # --------------------------
    def prob_map(self) -> torch.Tensor:
        return F.relu(self.interest - self.path)

    def choose_fixation(self) -> Optional[Tuple[int,int]]:
        pm = self.prob_map()
        if float(pm.max().item()) <= 0.0:
            return None
        idx = int(torch.argmax(pm.flatten()).item())
        W = pm.shape[1]
        return (idx % W, idx // W)

    # --------------------------
    # choose radius by interest density
    # --------------------------
    def choose_radius(self, center: Tuple[int,int]) -> Optional[int]:
        cx, cy = int(center[0]), int(center[1])
        best_r = None; best_score = -1.0
        H, W = self.interest.shape
        for r in self.cfg.saccade_radii:
            x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
            y0 = max(0, cy - r); y1 = min(H - 1, cy + r)
            region = self.interest[y0:y1+1, x0:x1+1]
            if region.numel() == 0:
                continue
            score = float(region.sum().item()) / float(region.numel())
            if score > best_score:
                best_score = score; best_r = r
        if best_score <= 0:
            return None
        return best_r

    # --------------------------
    # main step to pick or continue learning
    # --------------------------
    def step(self) -> Tuple[str, Optional[Tuple[int,int]], Optional[int]]:
        """
        Return (mode, center, r):
        - If returns ('end', None, None) -> finished image
        - If returns ('wander', center, None) -> no useful window chosen this call
        - If returns ('learn', center, r) -> do learning at this window
        - If returns ('strengthen', center, r) -> (not used here) leave strengthen decision to FoveatedMemory
        """
        if self.mode == 'wander':
            center = self.choose_fixation()
            if center is None:
                self.mode = 'end'
                return 'end', None, None
            r = self.choose_radius(center)
            if r is None:
                # nothing to do right now, keep wandering
                return 'wander', center, None
            # enter learn stage
            self.current_center = center
            self.learn_saccades_left = self.cfg.learn_saccades_N
            # mark path bump proportional to r
            self.add_path_bump(center, r)
            # mark visited region now (we mark whole window visited to avoid recomputing)
            self.mark_visited(center, r)
            return 'wander', center, r

        elif self.mode == 'learn':
            if self.learn_saccades_left <= 0:
                self.mode = 'wander'
                self.current_center = None
                return 'wander', None, None
            center = self.current_center
            if center is None:
                # fallback: pick new fixation
                center = self.choose_fixation()
                if center is None:
                    self.mode = 'end'
                    return 'end', None, None
                self.current_center = center
            r = self.choose_radius(center)
            if r is None:
                # nothing to learn at this center -> finish this learn sequence
                self.mode = 'wander'
                self.current_center = None
                return 'wander', None, None
            # each learn saccade: bump path & mark visited
            self.add_path_bump(center, r)
            self.mark_visited(center, r)
            self.learn_saccades_left -= 1
            return 'learn', center, r
        
        else:
            return 'end', None, None

    # -------------------------
    # Mode helpers
    # -------------------------
    def start_learning(self, center: Tuple[int,int], n_saccades: int):
        self.mode = 'learn'
        self.current_center = center
        self.saccades_left = n_saccades
        # immediately mark path bump for the learned fixation
        self.add_path_bump(center)

    def end_learning(self):
        self.mode = 'wander'
        self.current_center = None
        self.saccades_left = 0


# =============================
# Graph I related (Graph0, Graph1, Embedder, Collection, manager)
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

class SignatureEmbedder(nn.Module):
    def __init__(self, C:int, sig_vocab:int=4096, n_modalities:int=16):
        super().__init__()
        self.C = C
        self.sig_emb = nn.Embedding(sig_vocab, C)
        self.mod_emb = nn.Embedding(n_modalities, C)
        self.mlp = nn.Sequential(nn.Linear(C*2, C), nn.ReLU(), nn.Linear(C, C))

    def forward(self, modality:int, signature_hash:int) -> torch.Tensor:
        sidx = int(signature_hash % self.sig_emb.num_embeddings)
        midx = int(modality % self.mod_emb.num_embeddings)
        s = self.sig_emb(torch.tensor(sidx, dtype=torch.long, device=self.mlp[0].weight.device))
        m = self.mod_emb(torch.tensor(midx, dtype=torch.long, device=self.mlp[0].weight.device))
        out = self.mlp(torch.cat([s, m], dim=-1))
        return out.detach()

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
            # score = bin(qbit & p.bitset).count('1')
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
                    # recompute bitset
                    bset = 0
                    for m in p.members:
                        if m in self.nodes:
                            bset |= (1 << (hash(self.nodes[m].signature) & (self.cfg.bitset_bits-1)))
                    p.bitset = bset
                    if len(p.members) == 0:
                        del self.protos[pid]


class GraphI:
    """
    Manager for Graph I functionality: owns descriptor maps, GraphCollections (graph0, graph1),
    extraction, retrieval, strengthen and learning procedures.
    """
    def __init__(self, cfg:Config, device=None):
        self.cfg = cfg
        self.device = device or cfg.device

        self.embedder = SignatureEmbedder(cfg.descriptor_C, sig_vocab=4096, n_modalities=max(16, cfg.graphs_per_layer['graphI']))

        # descriptor maps and counts per scale
        self.desc_maps: Dict[int, torch.Tensor] = {}
        self.desc_counts: Dict[int, torch.Tensor] = {}
        for s in cfg.scales:
            Hc = math.ceil(cfg.H / s)
            Wc = math.ceil(cfg.W / s)
            self.desc_maps[s] = torch.zeros((cfg.descriptor_C, Hc, Wc), device=self.device, dtype=torch.float32)
            self.desc_counts[s] = torch.zeros((1, Hc, Wc), device=self.device, dtype=torch.float32)

        # Graph I collections
        n_col = cfg.graphs_per_layer.get('graphI', 5)
        self.graphI: List[GraphCollection] = [GraphCollection(cfg.base_bucket_size, cfg) for _ in range(n_col)]

        # saccade-related state
        self.saccade = Saccade(cfg = self.cfg, device=self.device)
        self.cached_feats: List[Dict[str,Any]] = []

    # --------------------------
    # reset image (clear caches & reset saccade)
    # --------------------------
    def reset_image(self, grad, hue, curvature_bank=None, aspect_bank=None, orient_bank=None):
        # reset descriptor maps & counts
        for s in self.cfg.scales:
            self.desc_maps[s].zero_(); self.desc_counts[s].zero_()
        # clear cached feats
        self.cached_feats = []
        # reset saccade controller (interest/path/visited)
        self.saccade.reset(grad, hue)

    # --------------------------
    # Public small helper to run saccade decision
    # --------------------------
    def propose_window(self) -> Tuple[str, Optional[Tuple[int,int]], Optional[int]]:
        return self.saccade.step()

    # --------------------------
    # extract_features: incremental within a window
    # --------------------------
    def extract_features(self,
                         grad: torch.Tensor,
                         hue: torch.Tensor,
                         curvature_bank: Optional[torch.Tensor] = None,
                         aspect_bank: Optional[torch.Tensor] = None,
                         orient_bank: Optional[torch.Tensor] = None,
                         center: Optional[Tuple[int,int]] = None,
                         r: Optional[int] = None,
                         ts: int = 0) -> List[Dict[str,Any]]:
        """
        Incremental extraction copied from previous monolithic implementation but now owned by GraphI.
        """
        device = self.device
        H = int(hue.shape[-2]); W = int(hue.shape[-1])

        # If full-image extraction requested (center is None), fallback to previous global behavior.
        if center is None or r is None:
            feats = self._extract_global_features(grad, hue, curvature_bank, aspect_bank, orient_bank, ts)
            # cache & mark visited whole image
            self.cached_feats = feats
            if self.saccade.visited is not None:
                self.saccade.visited[:, :] = 1
            return feats

        # compute window bounding box
        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - r); x1 = min(W - 1, cx + r)
        y0 = max(0, cy - r); y1 = min(H - 1, cy + r)

        # collect previously cached feats in this window
        prev_feats = []
        for f in self.cached_feats:
            fx, fy = int(f['pos'][0]), int(f['pos'][1])
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                prev_feats.append(f)

        # Determine unvisited mask patch
        if self.saccade.visited is None:
            unvisited_patch = torch.ones((y1 - y0 + 1, x1 - x0 + 1), dtype=torch.bool, device=device)
        else:
            visited_patch = self.saccade.visited[y0:y1+1, x0:x1+1]
            unvisited_patch = (visited_patch == 0)

        new_feats: List[Dict[str,Any]] = []

        # --- modality: edges (grad) ---
        g = grad.squeeze(0)
        if g.ndim == 4 and g.shape[-1] == 2:
            g = g.squeeze(0)
        if g.ndim == 3 and g.shape[-1] == 2:
            gx = g[..., 0].to(device); gy = g[..., 1].to(device)
        else:
            gx = torch.zeros((H, W), device=device); gy = torch.zeros_like(gx)
        mag = torch.sqrt(gx*gx + gy*gy)
        # take window patch
        mag_patch = mag[y0:y1+1, x0:x1+1].clone()
        # mask visited pixels (ensure they are not selected again)
        mag_patch_masked = mag_patch.clone()
        mag_patch_masked[unvisited_patch] = float('-inf')
        flat = mag_patch_masked.flatten()
        topk = min(self.cfg.topk_per_modality, int((mag_patch_masked != float('-inf')).sum().item()))
        if topk > 0:
            vals, idxs = torch.topk(flat, topk)
            Wpatch = x1 - x0 + 1
            for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                if not np.isfinite(v):
                    continue
                ry = idx // Wpatch; rx = idx % Wpatch
                absx = x0 + rx; absy = y0 + ry
                angle = math.atan2(float(gy[absy, absx].item()), float(gx[absy, absx].item()))
                binidx = int(((angle + math.pi) / (2*math.pi)) * 16) & 0xff
                sig = int((0 << 24) | binidx)
                emb = self._embed_stub(0, sig)
                fdict = {'modality': 0, 'signature': sig, 'pos': (absx, absy), 'value': float(v), 'embedding': emb}
                new_feats.append(fdict)
                # update descriptor maps
                self._accumulate_descriptor(emb, absx, absy)

        # --- modality: hue ---
        hmap = hue.squeeze(0).squeeze(0).to(device)
        hue_patch = hmap[y0:y1+1, x0:x1+1]
        # local contrast (global method but we compute per pixel patch using same formula)
        k = 7; pad = k // 2
        kernel = torch.ones((1,1,k,k), device=device) / (k*k)
        hue_mean = F.conv2d(hmap[None,None], kernel, padding=pad)[0,0]
        hue_patch_mean = hue_mean[y0:y1+1, x0:x1+1]
        sal_patch = torch.abs(hue_patch - hue_patch_mean)
        sal_patch_masked = sal_patch.clone(); sal_patch_masked[unvisited_patch] = float('-inf')
        flat = sal_patch_masked.flatten()
        topk = min(self.cfg.topk_per_modality, int((sal_patch_masked != float('-inf')).sum().item()))
        if topk > 0:
            vals, idxs = torch.topk(flat, topk)
            Wpatch = x1 - x0 + 1
            for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                if not np.isfinite(v):
                    continue
                ry = idx // Wpatch; rx = idx % Wpatch
                absx = x0 + rx; absy = y0 + ry
                binidx = int((hmap[absy, absx].item()) * 32) & 0xff
                sig = int((1 << 24) | binidx)
                emb = self._embed_stub(1, sig)
                fdict = {'modality': 1, 'signature': sig, 'pos': (absx, absy), 'value': float(v), 'embedding': emb}
                new_feats.append(fdict)
                self._accumulate_descriptor(emb, absx, absy)

        # --- banks: curvature/aspect/orient (modalities 2,3,4) ---
        banks = [curvature_bank, aspect_bank, orient_bank]
        for mi, bank in enumerate(banks, start=2):
            if bank is None:
                continue
            b = bank.squeeze(0).to(device)  # [C, H, W]
            Cb, Hfull, Wfull = b.shape
            patch = b[:, y0:y1+1, x0:x1+1]  # [C, Hpatch, Wpatch]
            sal = torch.norm(patch, dim=0)
            sal_masked = sal.clone(); sal_masked[unvisited_patch] = float('-inf')
            flat = sal_masked.flatten()
            topk = min(self.cfg.topk_per_modality, int((sal_masked != float('-inf')).sum().item()))
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                Wpatch = x1 - x0 + 1
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    if not np.isfinite(v):
                        continue
                    ry = idx // Wpatch; rx = idx % Wpatch
                    absx = x0 + rx; absy = y0 + ry
                    # channel signature
                    ch = int(torch.argmax(patch[:, ry, rx]).item())
                    sig = int((mi << 24) | (ch & 0xffff))
                    emb = self._embed_stub(mi, sig)
                    fdict = {'modality': mi, 'signature': sig, 'pos': (absx, absy), 'value': float(v), 'embedding': emb}
                    new_feats.append(fdict)
                    self._accumulate_descriptor(emb, absx, absy)

        # Combine: add new_feats into cache; mark visited entire window (per requirement)
        # Avoid duplicating existing cached feat entries at same pos/signature
        self.cached_feats.extend(new_feats)
        # Mark visited for the whole window region (so future calls won't recompute)
        self.saccade.mark_visited(center, r)

        # Return combined feats within the window (previously cached + newly added)
        combined_feats = prev_feats + new_feats
        return combined_feats

    # --------------------------
    # Feature extraction (full image)
    # --------------------------
    def _extract_global_features(self, grad, hue, curvature_bank, aspect_bank, orient_bank, ts:int=0) -> List[Dict[str,Any]]:
        feats = []
        device = self.device
        # EDGE modality (0)
        if grad is not None:
            g = grad.squeeze(0)
            g = g.squeeze(0)  # H,W,2
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
                    # signature: orientation bin
                    angle = math.atan2(float(gy[ry,rx].item()), float(gx[ry,rx].item()))
                    binidx = int(((angle + math.pi) / (2*math.pi)) * 16)
                    sig = int((0 << 24) | (binidx & 0xffff))
                    emb = self.embedder(0, sig).cpu()
                    feats.append({'modality': 0, 'signature': sig, 'pos': (int(rx), int(ry)), 'value': float(v), 'embedding': emb})
                    self._accumulate_descriptor(emb, rx, ry)

        # HUE modality (1)
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
                    emb = self.embedder(1, sig).cpu()
                    feats.append({'modality': 1, 'signature': sig, 'pos':(int(rx),int(ry)), 'value':float(v), 'embedding':emb})
                    self._accumulate_descriptor(emb, rx, ry)

        # curvature/aspect/orient banks: modalities 2,3,4
        banks = [curvature_bank, aspect_bank, orient_bank]
        for mi, bank in enumerate(banks, start=2):
            if bank is None: continue
            b = bank.squeeze(0).to(device)  # [C,H,W]
            Cb, H, W = b.shape
            sal = torch.norm(b, dim=0)
            flat = sal.flatten()
            topk = min(self.cfg.topk_per_modality, flat.numel())
            if topk > 0:
                vals, idxs = torch.topk(flat, topk)
                for v, idx in zip(vals.cpu().numpy().tolist(), idxs.cpu().numpy().tolist()):
                    ry = idx // W; rx = idx % W
                    ch = int(torch.argmax(b[:,ry,rx]).item())
                    sig = int((mi << 24) | (ch & 0xffff))
                    emb = self.embedder(mi, sig).cpu()
                    feats.append({'modality':mi, 'signature':sig, 'pos':(int(rx),int(ry)), 'value':float(v), 'embedding':emb})
                    self._accumulate_descriptor(emb, rx, ry)
        return feats

    def _accumulate_descriptor(self, emb:torch.Tensor, x:int, y:int):
        emb = emb.to(self.device)
        for s in self.cfg.scales:
            Hc = self.desc_maps[s].shape[1]; Wc = self.desc_maps[s].shape[2]
            cx = min(Wc-1, max(0, int(x // s)))
            cy = min(Hc-1, max(0, int(y // s)))
            self.desc_maps[s][:, cy, cx] += emb
            self.desc_counts[s][0, cy, cx] += 1.0

    # --------------------------
    # helper: embedding stub & descriptor update
    # --------------------------
    def _embed_stub(self, modality:int, signature:int):
        C = self.cfg.descriptor_C
        h = hash((modality, signature)) & 0xffffffff
        rnd = np.random.RandomState(h)
        vec = torch.tensor(rnd.randn(C).astype(np.float32), device=self.device)
        return vec

    # --------------------------
    # Retrieval functions (coarse->fine)
    # --------------------------
    def retrieve_candidates(self, feats:List[Dict[str,Any]], ts:int=0):
        if len(feats) == 0:
            return set(), [], 0.0
        xs = [f['pos'][0] for f in feats]; ys = [f['pos'][1] for f in feats]
        cx0 = int(sum(xs) / len(xs)); cy0 = int(sum(ys) / len(ys))

        conv_scores_per_scale = {}
        matched_cells_per_scale = {}
        for s in self.cfg.scales:
            dm = self.desc_maps[s]
            counts = self.desc_counts[s]
            valid = counts.clone(); valid[valid==0] = 1.0
            norm_map = dm / valid
            norm_map = F.normalize(norm_map.unsqueeze(0), p=2, dim=1)
            K = 5
            tpl = torch.zeros((self.cfg.descriptor_C, K, K), device=self.device)
            cnt = torch.zeros((K, K), device=self.device)
            for f in feats:
                rx = int(f['pos'][0] // s) - int(cx0 // s) + K//2
                ry = int(f['pos'][1] // s) - int(cy0 // s) + K//2
                if 0 <= rx < K and 0 <= ry < K:
                    tpl[:, ry, rx] += f['embedding'].to(self.device)
                    cnt[ry, rx] += 1.0
            cnt[cnt==0]=1.0
            tpl = (tpl / cnt.unsqueeze(0)).unsqueeze(0)
            tpl = F.normalize(tpl, p=2, dim=1)
            scores = F.conv2d(norm_map, tpl, padding=(K//2, K//2))[0,0]
            conv_scores_per_scale[s] = scores
            flat = scores.flatten(); k = min(self.cfg.topk_per_scale.get(s,64), flat.numel())
            if k <= 0:
                matched_cells_per_scale[s] = []
            else:
                vals, idxs = torch.topk(flat, k)
                Hc, Wc = scores.shape
                cells = [(int(i % Wc), int(i // Wc)) for i in idxs.cpu().numpy().tolist()]
                matched_cells_per_scale[s] = cells

        candidate_nodes_by_collection = [set() for _ in range(len(self.graphI))]
        for s, cells in matched_cells_per_scale.items():
            for (cx, cy) in cells:
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        ncx = cx + dx; ncy = cy + dy
                        px = int((ncx + 0.5) * s); py = int((ncy + 0.5) * s)
                        for idx, coll in enumerate(self.graphI):
                            b = coll._bucket_coord(px, py)
                            for nid in coll.buckets.get(b, []):
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

        conv_score_mean = 0.0; s_count = 0
        for s, scores in conv_scores_per_scale.items():
            if scores is None: continue
            flat = scores.flatten()
            topk = min(8, flat.numel())
            if topk == 0: continue
            v = float(torch.topk(flat, topk).values.mean().item())
            conv_score_mean += v; s_count += 1
        conv_score_mean = conv_score_mean / max(1, s_count)
        proto_support_score = (proto_support_total / proto_support_count) if proto_support_count > 0 else 0.0
        node_support_score = min(1.0, len(activated_nodes) / max(1, self.cfg.max_activated_per_collection * len(self.graphI)))
        w_conv, w_proto, w_node = 0.4, 0.4, 0.2
        S = w_conv * conv_score_mean + w_proto * proto_support_score + w_node * node_support_score
        return activated_nodes, proto_matches, float(S)

    # --------------------------
    # Strengthen branch (unchanged semantics)
    # --------------------------
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
        # for graph2
        activated_protos = []
        for (cidx, pid, overlap) in proto_matches:
            activated_protos.append((cidx, pid))
        return activated_protos

    # --------------------------
    # Learning (slightly adapted)
    # --------------------------
    def learn(self, feats:List[Dict[str,Any]], activated_nodes:set, proto_matches:List[Tuple[int,int,float]], ts:int=0):
        ncols = len(self.graphI)
        new_nodes_by_collection = [ [] for _ in range(ncols) ]
        created = []
        # add new nodes
        for f in feats:
            mod = int(f['modality']); idx = mod % ncols
            coll = self.graphI[idx]
            sig = int(f['signature'])
            x,y = int(f['pos'][0]), int(f['pos'][1])
            b = coll._bucket_coord(x,y)
            found = False
            for nid in coll.buckets.get(b, []):
                n = coll.nodes[nid]
                if n.signature == sig:
                    found = True; break
            if not found:
                nid = coll.add_or_merge_node(sig, mod, (x,y), f['value'], ts)
                new_nodes_by_collection[idx].append(nid)
        for idx, coll in enumerate(self.graphI):
            # group by bucket
            groups: Dict[Tuple[int,int], List[int]] = {}
            for nid in new_nodes_by_collection[idx]:
                n = coll.nodes.get(nid); 
                if n is None: continue
                b = coll._bucket_coord(n.pos[0], n.pos[1])
                groups.setdefault(b, []).append(nid)
            # proto formation
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
                                    sig = coll.nodes[m].signature
                                    bitpos = (hash(sig) & (coll.cfg.bitset_bits - 1))
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

    # For persistence
    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> 'GraphI':
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Graph II (unchanged, copied over)
# =============================================================================
class Graph2Node:
    def __init__(self,
                 node_id: int,
                 members: List[Tuple[int, int]],
                 embedding: torch.Tensor,
                 bitset: int,
                 positions: Optional[torch.Tensor],
                 ts: int,
                 cfg: GraphIIConfig):
        self.id = int(node_id)
        self.members = list(members)            # list of (collection_idx, proto_id)
        self.modalities = {c for c, _ in members}
        self.embedding = embedding.detach().clone().float()
        self.bitset = int(bitset)

        if (positions is not None) and (positions.numel() > 0):
            self.pos_mean = positions.mean(dim=0).clone().float()
            diffs = positions - self.pos_mean.unsqueeze(0)
            self.pos_cov = (diffs.t() @ diffs) / max(1.0, float(positions.shape[0]))
        else:
            # default small covariance
            self.pos_mean = torch.zeros(2, device=self.embedding.device, dtype=torch.float32)
            self.pos_cov = torch.eye(2, device=self.embedding.device, dtype=torch.float32) * 1.0

        self.count = 1
        self.strength = 1.0
        self.last_seen = ts
        self.examples = deque(maxlen=8)  # store small set of example events (for debugging)

    def update(self, event_embedding: torch.Tensor, event_bitset: int,
               event_positions: Optional[torch.Tensor], ts: int, cfg: GraphIIConfig):
        # increment count / time / strength
        self.count += 1
        self.last_seen = ts
        self.strength = (1 - cfg.graph2_strength_beta) * self.strength + cfg.graph2_strength_beta * 1.0

        # embedding EMA (normalized)
        self.embedding = F.normalize((1 - cfg.graph2_embed_beta) * self.embedding + cfg.graph2_embed_beta * event_embedding, dim=0)

        # bitset union
        self.bitset |= int(event_bitset)

        # positions EMA mean/cov
        if (event_positions is not None) and (event_positions.numel() > 0):
            ev_mean = event_positions.mean(dim=0)
            self.pos_mean = (1 - cfg.graph2_pos_beta) * self.pos_mean + cfg.graph2_pos_beta * ev_mean
            diffs = event_positions - ev_mean.unsqueeze(0)
            ev_cov = (diffs.t() @ diffs) / max(1.0, float(event_positions.shape[0]))
            self.pos_cov = (1 - cfg.graph2_pos_beta) * self.pos_cov + cfg.graph2_pos_beta * ev_cov

    def merge_into(self, other:'Graph2Node', cfg: GraphIIConfig):
        """
        Merge 'other' node into self (self absorbs other).
        Update members, embedding (weighted), bitset, counts, pos stats.
        """
        # combine members (keep unique)
        existing_set = set(self.members)
        for m in other.members:
            if m not in existing_set:
                self.members.append(m)
                existing_set.add(m)
                self.modalities.add(m[0])
        # weighted embedding mean by counts
        total_count = float(self.count + other.count)
        self.embedding = F.normalize((self.embedding * self.count + other.embedding * other.count) / total_count, dim=0)
        # union bitset
        self.bitset |= other.bitset
        # combine pos stats approximately by weighted mean
        self.pos_mean = (self.pos_mean * self.count + other.pos_mean * other.count) / total_count
        # simple cov combine (approx)
        self.pos_cov = (self.pos_cov * self.count + other.pos_cov * other.count) / total_count
        # counts and strength
        self.count += other.count
        self.strength = max(self.strength, other.strength) + cfg.graph2_strength_beta
        # last seen is latest
        self.last_seen = max(self.last_seen, other.last_seen)
        # examples: extend
        for ex in other.examples:
            self.examples.append(ex)


class Graph2Edge:
    def __init__(self, src: int, dst: int, vec: torch.Tensor, ts: int, cfg: GraphIIConfig):
        self.src = int(src)
        self.dst = int(dst)
        self.count = 1
        self.weight = float(cfg.graph2_edge_init_weight)
        self.vec_mean = vec.detach().clone().float()
        self.vec_cov = torch.zeros((2, 2), device=vec.device, dtype=torch.float32)
        self.last_seen = ts

    def update(self, vec: torch.Tensor, ts: int, cfg: GraphIIConfig):
        vec = vec.detach().float()
        self.count += 1
        self.last_seen = ts
        self.weight += cfg.graph2_edge_inc
        delta = vec - self.vec_mean
        self.vec_mean = self.vec_mean + cfg.graph2_vec_beta * delta
        self.vec_cov = (1 - cfg.graph2_vec_beta) * self.vec_cov + cfg.graph2_vec_beta * torch.ger(delta, delta)


# ---------- GraphII manager ----------
class GraphII:
    """
    GraphII implements candidate-cache + consolidation for cross-modal graph2.
    """
    def __init__(self, cfg: Optional[GraphIIConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg if cfg is not None else GraphIIConfig()
        if device is not None:
            self.cfg.device = device
        self.device = self.cfg.device

        # core containers
        self.graph2_nodes: Dict[int, Graph2Node] = {}
        self.graph2_edges: Dict[Tuple[int, int], Graph2Edge] = {}
        self.next_graph2_id = 0

        # inverted index: (collection_idx, proto_id) -> set(graph2_id)
        self.inverted_index_proto2graph2: Dict[Tuple[int, int], set] = defaultdict(set)

        # candidate cache: bucket_key -> list of records
        # record: dict with keys: members(tuple), bitset(int), emb(tensor), positions(tensor or None), first_ts, last_ts, count
        self.cooccur_cache: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

        # previous active node for transition updates
        self.prev_active: Optional[int] = None
        self.prev_center: Optional[torch.Tensor] = None

    # ---------- utility: create bucket key from members ----------
    def _make_bucket_key(self, members: List[Tuple[int, int]]) -> Tuple:
        """
        Simple stable bucket key:
          - compute tokens like 'c:p' for each member
          - sort tokens and take prefix of length bucket_key_n
        This groups events that share several same top tokens.
        """
        tokens = [f"{c}:{pid}" for (c, pid) in members]
        tokens.sort()
        prefix = tuple(tokens[:self.cfg.bucket_key_n])
        return prefix

    # ---------- utility: synthesize embedding from bitset if needed ----------
    def _bitset_to_embedding(self, bitset: int, dim: int) -> torch.Tensor:
        vec = torch.zeros(dim, device=self.device, dtype=torch.float32)
        b = int(bitset)
        i = 0
        # iterate up to at most 64 bits for speed
        while b and i < 64:
            lowest = (b & -b).bit_length() - 1
            idx = (lowest * 2654435761) % dim
            vec[idx] += 1.0
            b = b & (b - 1)
            i += 1
        if vec.sum() == 0:
            vec += 1e-6
        return F.normalize(vec, dim=0)

    # ---------- similarity between two candidate records ----------
    def _record_similarity(self, recA: Dict, recB: Dict) -> float:
        # bitset Jaccard
        a = int(recA['bitset']); b = int(recB['bitset'])
        inter = (a & b).bit_count()
        union = (a | b).bit_count()
        bit_j = inter / float(max(1, union))

        # embedding cosine (guard)
        embA = recA['emb']; embB = recB['emb']
        if (embA is None) or (embB is None):
            emb_cos = 0.0
        else:
            emb_cos = float(F.cosine_similarity(embA, embB, dim=0).item())

        # modality overlap ratio
        modsA = {c for (c, _) in recA['members']}
        modsB = {c for (c, _) in recB['members']}
        mod_overlap = len(modsA & modsB) / float(max(1, len(modsA | modsB)))

        # weighted sum
        score = (self.cfg.w_g2_bit * bit_j) + (self.cfg.w_g2_emb * emb_cos) + (self.cfg.w_g2_mod * mod_overlap)
        return float(score)

    # ---------- insert/merge event into bucket ----------
    def _insert_event_into_bucket(self, bucket_key: Any, event: Dict, ts: int) -> Dict:
        """
        Try to merge event into an existing record in bucket (if similarity >= threshold),
        otherwise append as new record.
        Returns the merged/created record.
        """
        bucket = self.cooccur_cache[bucket_key]
        for rec in bucket:
            # If rec is old beyond T_coalesce, skip merge (it will eventually be evicted)
            if ts - rec['first_ts'] > self.cfg.T_coalesce:
                continue
            sim = self._record_similarity(rec, event)
            if sim >= self.cfg.event_merge_thresh:
                # merge event into rec (EMA-like)
                rec['count'] += 1
                rec['last_ts'] = ts
                # emb EMA
                if rec['emb'] is None:
                    rec['emb'] = event['emb']
                elif event['emb'] is None:
                    pass
                else:
                    rec['emb'] = F.normalize((1 - 0.1) * rec['emb'] + 0.1 * event['emb'], dim=0)
                # bitset union
                rec['bitset'] |= int(event['bitset'])
                # append positions if present
                if (rec['positions'] is None) and (event['positions'] is not None):
                    rec['positions'] = event['positions'].clone()
                elif (rec['positions'] is not None) and (event['positions'] is not None):
                    rec['positions'] = torch.cat([rec['positions'], event['positions']], dim=0)
                # extend members list by union (keep canonical order)
                existing = set(rec['members'])
                for m in event['members']:
                    if m not in existing:
                        rec['members'].append(m)
                        existing.add(m)
                return rec
        # no merge -> append new record
        newrec = {
            'members': list(event['members']),     # list of tuples
            'bitset': int(event['bitset']),
            'emb': None if event['emb'] is None else event['emb'].detach().clone(),
            'positions': None if event['positions'] is None else (event['positions'].detach().clone()),
            'first_ts': ts,
            'last_ts': ts,
            'count': 1
        }
        bucket.append(newrec)
        return newrec

    # ---------- promotion: try promote records in bucket_key to node ----------
    def _try_promote_bucket_record(self, bucket_key: Any, rec: Dict, ts: int):
        """
        Promote rec -> Graph2Node if rec['count'] >= form_count and age within T_coalesce.
        """
        if rec['count'] < self.cfg.graph2_form_count:
            return None
        age = ts - rec['first_ts']
        if age > self.cfg.T_coalesce:
            # outdated; do not promote
            return None
        # Create node
        gid = self.next_graph2_id
        self.next_graph2_id += 1
        dim = self.cfg.descriptor_C
        emb = rec['emb'] if rec['emb'] is not None else self._bitset_to_embedding(rec['bitset'], dim)
        positions = rec['positions'] if rec['positions'] is not None else None
        node = Graph2Node(gid, rec['members'], emb, rec['bitset'], positions, ts, self.cfg)
        self.graph2_nodes[gid] = node
        # update inverted index
        for (c, pid) in rec['members']:
            self.inverted_index_proto2graph2[(c, pid)].add(gid)
        # clear record (remove from bucket)
        bucket = self.cooccur_cache[bucket_key]
        for i, r in enumerate(bucket):
            if r is rec:
                del bucket[i]
                break
        return node

    # ---------- public observe API ----------
    def observe(self, activated_protos: List[Tuple[int, int]], collections: List[Any], ts: int) -> Optional[int]:
        """
        Main call per fixation:
          - activated_protos: list[(collection_idx, proto_id)]
          - collections: list of GraphCollection to read proto bitset/embedding/positions
          - ts: integer timestamp (monotonic)
        Returns: active graph2 node id (if matched or created), else None
        """
        if not activated_protos:
            # no protos -> reset prev context
            self.prev_active = None
            self.prev_center = None
            return None

        # prepare event: gather proto embeddings, bitsets, and proto member positions
        emb_list = []
        bitset_union = 0
        pos_list = []
        members = []
        for (cidx, pid) in activated_protos:
            if cidx < 0 or cidx >= len(collections):
                continue
            coll = collections[cidx]
            p = coll.protos.get(pid, None)
            if p is None:
                continue
            # get embedding or synthesize
            if getattr(p, 'embedding', None) is not None:
                emb = p.embedding.detach().to(self.device).float()
            else:
                # use bitset if available
                b = int(getattr(p, 'bitset', 0))
                emb = self._bitset_to_embedding(b, self.cfg.descriptor_C)
            emb_list.append(emb)
            bitset_union |= int(getattr(p, 'bitset', 0))
            # gather member node positions from coll.nodes
            pos_acc = []
            for nid in getattr(p, 'members', []):
                n = coll.nodes.get(nid, None)
                if n is not None:
                    pos_acc.append([float(n.pos[0]), float(n.pos[1])])
            if pos_acc:
                pos_list.append(torch.tensor(pos_acc, device=self.device, dtype=torch.float32))
            members.append((cidx, pid))

        if not members:
            # nothing meaningful
            return None

        # event embedding
        proto_embeddings = torch.stack(emb_list, dim=0) if emb_list else None
        event_emb = F.normalize(proto_embeddings.mean(dim=0), dim=0) if proto_embeddings is not None else None
        event_positions = torch.cat(pos_list, dim=0) if pos_list else None

        # 1) attempt to match existing graph2 nodes via inverted index -> small candidate set
        candidate_ids = set()
        for (c, pid) in members:
            candidate_ids.update(self.inverted_index_proto2graph2.get((c, pid), set()))
        # if empty candidate set, fall back to scanning small subset (or none)
        best_id = None; best_score = 0.0
        if candidate_ids:
            for gid in candidate_ids:
                g2 = self.graph2_nodes.get(gid, None)
                if g2 is None:
                    continue
                score = self._match_score(g2, members, event_emb, bitset_union)
                if score > best_score:
                    best_score, best_id = score, gid

        # decide match vs candidate creation
        active_node_id = None
        if best_score >= self.cfg.graph2_match_threshold and best_id is not None:
            node = self.graph2_nodes[best_id]
            node.update(event_emb if event_emb is not None else self._bitset_to_embedding(bitset_union, self.cfg.descriptor_C),
                        bitset_union, event_positions, ts, self.cfg)
            active_node_id = best_id
        else:
            # no strong existing match -> insert event into cooccur cache bucket and possibly promote
            bucket_key = self._make_bucket_key(members)
            event = {'members': list(members), 'bitset': int(bitset_union), 'emb': event_emb, 'positions': event_positions}
            merged_rec = self._insert_event_into_bucket(bucket_key, event, ts)
            promoted = self._try_promote_bucket_record(bucket_key, merged_rec, ts)
            if promoted is not None:
                active_node_id = promoted.id

        # 2) update directed transition edges using prev_active -> active_node_id
        if (self.prev_active is not None) and (active_node_id is not None) and (self.prev_active != active_node_id):
            # compute displacement vector: current centroid - prev_center
            if event_positions is not None:
                cur_centroid = event_positions.mean(dim=0)
            else:
                cur_centroid = torch.zeros(2, device=self.device)
            prev_centroid = self.prev_center if self.prev_center is not None else cur_centroid
            vec = cur_centroid - prev_centroid
            self._update_edge(self.prev_active, active_node_id, vec, ts)

        # update prev_active / prev_center if we have active node
        if active_node_id is not None:
            if event_positions is not None:
                self.prev_center = event_positions.mean(dim=0).detach().clone()
            else:
                self.prev_center = torch.zeros(2, device=self.device)
            self.prev_active = active_node_id
        # return currently active graph2 id (or None)
        return active_node_id

    # ---------- match scoring ----------
    def _match_score(self, g2: Graph2Node, members: List[Tuple[int, int]], event_emb: Optional[torch.Tensor], event_bitset: int) -> float:
        # modality overlap
        event_mods = {c for (c, _) in members}
        mod_overlap = len(event_mods & g2.modalities) / float(max(1, len(g2.modalities)))
        # bitset jaccard
        inter = (g2.bitset & int(event_bitset)).bit_count()
        union = (g2.bitset | int(event_bitset)).bit_count()
        bit_score = inter / float(max(1, union))
        '''
        emb_score is invalid for match, it is a discontinuous function
        # embedding cosine
        if event_emb is None:
            emb_score = 0.0
        else:
            emb_score = float(F.cosine_similarity(g2.embedding, event_emb, dim=0).item())
        score = (self.cfg.w_g2_mod * mod_overlap + self.cfg.w_g2_bit * bit_score + self.cfg.w_g2_emb * emb_score)
        '''
        score = (self.cfg.w_g2_mod * mod_overlap + self.cfg.w_g2_bit * bit_score)
        return float(score)

    # ---------- update/create edge ----------
    def _update_edge(self, src: int, dst: int, vec: torch.Tensor, ts: int):
        key = (int(src), int(dst))
        if key not in self.graph2_edges:
            self.graph2_edges[key] = Graph2Edge(src, dst, vec, ts, self.cfg)
        else:
            self.graph2_edges[key].update(vec, ts, self.cfg)

    # ---------- periodic consolidation: merge similar nodes ----------
    def consolidate(self):
        """
        Merge very similar graph2 nodes to reduce fragmentation.
        Strategy:
          - For each node, find neighbors via inverted_index union of their member protos
          - Compute pairwise similarity and merge pairs with similarity >= merge_thresh
        This is a conservative single-pass merge (merging smaller into larger).
        """
        merge_thresh = self.cfg.graph2_merge_thresh
        visited = set()
        # build candidate neighbors via inverted index
        for gid, g in list(self.graph2_nodes.items()):
            if gid in visited:
                continue
            # gather neighbor candidates via member protos
            candidates = set()
            for (c, pid) in g.members:
                candidates.update(self.inverted_index_proto2graph2.get((c, pid), set()))
            candidates.discard(gid)
            # compute similarities
            for other_id in list(candidates):
                if other_id == gid or other_id not in self.graph2_nodes:
                    continue
                other = self.graph2_nodes[other_id]
                # compute simple similarity via bitset Jaccard + embedding cosine
                inter = (g.bitset & other.bitset).bit_count()
                union = (g.bitset | other.bitset).bit_count()
                bit_j = inter / float(max(1, union))
                emb_cos = float(F.cosine_similarity(g.embedding, other.embedding, dim=0).item())
                sim = 0.5 * bit_j + 0.5 * emb_cos
                if sim >= merge_thresh:
                    # merge smaller into larger
                    if g.count >= other.count:
                        g.merge_into(other, self.cfg)
                        # remove other from storage & inverted index
                        self._remove_node(other_id)
                    else:
                        other.merge_into(g, self.cfg)
                        self._remove_node(gid)
                        visited.add(other_id)
                        break
            visited.add(gid)

    # ---------- remove node helper ----------
    def _remove_node(self, gid: int):
        node = self.graph2_nodes.pop(gid, None)
        if node is None:
            return
        # remove from inverted index
        for (c, pid) in node.members:
            s = self.inverted_index_proto2graph2.get((c, pid), None)
            if s:
                s.discard(gid)
                if not s:
                    del self.inverted_index_proto2graph2[(c, pid)]
        # remove edges touching gid
        for key in list(self.graph2_edges.keys()):
            if key[0] == gid or key[1] == gid:
                del self.graph2_edges[key]

    # ---------- decay / pruning ----------
    def decay(self):
        # decay nodes
        for gid in list(self.graph2_nodes.keys()):
            g = self.graph2_nodes[gid]
            g.strength *= self.cfg.graph2_decay
            if g.strength < self.cfg.graph2_min_strength and g.count < self.cfg.graph2_min_count:
                self._remove_node(gid)
        # decay edges
        for key in list(self.graph2_edges.keys()):
            e = self.graph2_edges[key]
            e.weight *= self.cfg.graph2_edge_decay
            if e.weight < self.cfg.graph2_edge_min:
                del self.graph2_edges[key]

    # ---------- debug / inspection helpers ----------
    def get_node_summary(self, gid: int):
        g = self.graph2_nodes.get(gid, None)
        if g is None:
            return None
        return {
            'id': g.id,
            'count': g.count,
            'strength': g.strength,
            'members': list(g.members),
            'modalities': list(g.modalities),
            'pos_mean': g.pos_mean.cpu().tolist() if isinstance(g.pos_mean, torch.Tensor) else None
        }

    def list_nodes(self):
        return list(self.graph2_nodes.keys())

    def list_edges(self):
        return [(k, {'weight': v.weight, 'count': v.count, 'vec_mean': v.vec_mean.cpu().tolist()}) for k, v in self.graph2_edges.items()]

# =============================
# FoveatedMemory (orchestrator): only initialization + run_one_step + small helpers
# =============================
class FoveatedMemory:
    def __init__(self, cfg:Config = Config()):
        self.cfg = cfg
        self.device = cfg.device

        # Graph I manager
        self.graphI = GraphI(cfg, device=self.device)

        # Graph II manager (unchanged)
        self.graphII = GraphII(cfg, device=self.device)

        # runtime bookkeeping
        self.timestep = 0

    def run_one_step(self,
                     grad: torch.Tensor,
                     hue: torch.Tensor,
                     curvature_bank: Optional[torch.Tensor] = None,
                     aspect_bank: Optional[torch.Tensor] = None,
                     orient_bank: Optional[torch.Tensor] = None,
                     reset_image: bool = False,
                     ts: Optional[int] = None):
        """
        Orchestrator. Delegates GraphI-related work to GraphI instance. Does NOT implement GraphII training here (per requirements).
        """
        if ts is None:
            ts = self.timestep
        self.timestep += 1

        if reset_image:
            self.graphI.reset_image(grad, hue, curvature_bank, aspect_bank, orient_bank)

        # Get saccade decision from GraphI's saccade controller
        mode_sacc, center, r = self.graphI.propose_window()
        if mode_sacc == 'end':
            return 'end',None, None, None

        if center is None or r is None:
            # nothing to do now, remain in wander
            return 'wander', None, None, None

        # Incremental extraction limited to window (delegated)
        window_feats = self.graphI.extract_features(grad, hue, curvature_bank, aspect_bank, orient_bank, center=center, r=r, ts=ts)

        # Retrieve & score (delegated)
        activated_nodes, proto_matches, S = self.graphI.retrieve_candidates(window_feats, ts)

        # Decide branch: only strengthen OR learn (not both)
        if S >= self.cfg.strengthen_threshold and len(activated_nodes) > 0:
            # perform strengthen
            self.graphI.saccade.mode = 'wander'
            activated_protos = self.graphI.apply_strengthen(activated_nodes, proto_matches, ts)
            new_protos = []
        else:
            # learning branch
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
    

    # persistence delegations
    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> 'FoveatedMemory':
        with open(path, 'rb') as f:
            return pickle.load(f)



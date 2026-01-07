# foveated_memory_with_saccades.py
import math
import pickle
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

    # retrieval / matching
    topk_per_scale = {8: 128, 4: 64, 2: 32, 1: 16}
    topk_per_modality = 200000
    proto_form_thresh = 3
    proto_merge_thresh = 0.7
    strengthen_threshold = 0.45
    proto_match_thresh = 0.08

    max_nodes_per_collection = 50000
    max_edges_per_node = 12

    max_candidates_per_step = 512
    max_activated_per_collection = 128

    # Saccade / window / learning params
    saccade_radii = [32, 64, 96]  # radii in pixels
    learn_saccades_N = 4
    max_writes_per_modality = 5  # per saccade, per modality
    interest_end_thresh = 0.45
    path_sigma = 3.0  # gaussian sigma for marking path
    device = torch.device('cpu')

    # interest weights
    w_grad = 0.5
    w_hue = 0.3
    w_cont = 0.2

class Saccade:
    """
    Encapsulates all saccade / fixation / path / interest logic.
    """

    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or cfg.device
        self.alpha = 1.2

        self.interest: Optional[torch.Tensor] = None  # H x W
        self.path: Optional[torch.Tensor] = None      # H x W

        self.mode: str = 'wander'   # wander | learn | end
        self.current_center: Optional[Tuple[int, int]] = None
        self.learn_saccades_left: int = 0

    # -------------------------------------------------
    # Image-level reset
    # -------------------------------------------------
    def reset(self,
              grad: torch.Tensor,
              hue: torch.Tensor):
        H, W = hue.shape[-2], hue.shape[-1]
        self.path = torch.zeros((H, W), device=self.device)
        self.interest = self._compute_interest(grad, hue)
        self.mode = 'wander'
        self.current_center = None
        self.learn_saccades_left = 0

    # -------------------------------------------------
    # Interest map
    # -------------------------------------------------
    def _compute_interest(self, grad, hue):
        g = grad.squeeze(0)
        g = g.squeeze(0) if g.ndim == 4 else g
        gx, gy = g[..., 0], g[..., 1]
        grad_mag = torch.sqrt(gx * gx + gy * gy)

        # hue contrast
        h = hue.squeeze(0).squeeze(0)
        kernel = torch.ones((1, 1, 7, 7), device=self.device) / 49.0
        h_mean = F.conv2d(h[None, None], kernel, padding=3)[0, 0]
        hue_contrast = torch.abs(h - h_mean)

        # normalize
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-6)
        
        interest = (
            self.cfg.w_grad * norm(grad_mag) +
            self.cfg.w_hue * norm(hue_contrast)
        )
        interest = norm(interest)
        
        return torch.exp((interest - 1) / self.alpha).clamp(0.0, 1.0)

    # -------------------------------------------------
    # Path bump (radius-aware)
    # -------------------------------------------------
    def add_path_bump(self, center: Tuple[int, int], window_radius: int):
        """
        Gaussian bump radius proportional to window_radius.
        """
        cx, cy = center
        H, W = self.path.shape

        sigma = max(1.0, window_radius * 0.5)
        radius = int(math.ceil(3 * sigma))

        x0 = max(0, cx - radius)
        x1 = min(W - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(H - 1, cy + radius)

        xs = torch.arange(x0, x1 + 1, device=self.device)
        ys = torch.arange(y0, y1 + 1, device=self.device)

        dx2 = (xs - cx).float() ** 2
        dy2 = (ys - cy).float() ** 2
        g = torch.exp(-(dy2[:, None] + dx2[None, :]) / (2 * sigma * sigma))

        self.path[y0:y1 + 1, x0:x1 + 1] = torch.clamp(
            self.path[y0:y1 + 1, x0:x1 + 1] + g,
            0.0, 1.0
        )

    # -------------------------------------------------
    # Fixation selection
    # -------------------------------------------------
    def _prob_map(self):
        return F.relu(self.interest - self.path)

    def choose_fixation(self):
        prob = self._prob_map()
        vmax = float(prob.max())
        if vmax < self.cfg.interest_end_thresh:
            return None
        idx = int(torch.argmax(prob))
        W = prob.shape[1]
        return (idx % W, idx // W)

    # -------------------------------------------------
    # Radius selection
    # -------------------------------------------------
    def choose_radius(self, center):
        cx, cy = center
        best_r, best_score = None, -1.0
        for r in self.cfg.saccade_radii:
            x0 = max(0, cx - r)
            x1 = min(self.interest.shape[1] - 1, cx + r)
            y0 = max(0, cy - r)
            y1 = min(self.interest.shape[0] - 1, cy + r)
            region = self.interest[y0:y1 + 1, x0:x1 + 1]
            score = float(region.mean())
            if score > best_score:
                best_score, best_r = score, r
        return best_r if best_score > 0 else None

    # -------------------------------------------------
    # Main step
    # -------------------------------------------------
    def step(self):
        if self.mode == 'wander':
            center = self.choose_fixation()
            if center is None:
                self.mode = 'end'
                return 'end', None, None

            r = self.choose_radius(center)
            if r is None:
                return 'wander', center, None

            self.mode = 'learn'
            self.current_center = center
            self.learn_saccades_left = self.cfg.learn_saccades_N
            self.add_path_bump(center, r)
            return 'learn', center, r

        elif self.mode == 'learn':
            if self.learn_saccades_left <= 0:
                self.mode = 'wander'
                self.current_center = None
                return 'wander', None, None

            r = self.choose_radius(self.current_center)
            self.add_path_bump(self.current_center, r)
            self.learn_saccades_left -= 1
            return 'learn', self.current_center, r

        return 'end', None, None

    # -------------------------
    # Utility: feats in window (filter by pos) - uses feats list maintained externally
    # -------------------------
    @staticmethod
    def feats_in_window(center: Tuple[int,int], r: int, feats: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        cx, cy = int(center[0]), int(center[1])
        res = []
        for f in feats:
            fx, fy = int(f['pos'][0]), int(f['pos'][1])
            if abs(fx - cx) <= r and abs(fy - cy) <= r:
                res.append(f)
        return res

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
# Small helper classes (Graph0, Graph1 etc. - similar to earlier)
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

# =============================
# Embedding
# =============================
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

# =============================
# Main class with saccades
# =============================
class FoveatedMemory:
    def __init__(self, cfg:Config = Config()):
        self.cfg = cfg
        self.device = cfg.device

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
        self.saccade = Saccade(self.cfg, device=self.device)
        
        self.path = None            # H x W tensor, accumulated gaussian bumps
        self.interest = None        # H x W tensor, computed per image
        self.last_feats: List[Dict[str,Any]] = []  # last full-image feature list
        self.saccade.mode = 'wander'        # current mode: 'wander'/'learn'/'strengthen'/'end'
        self.saccades_left = 0
        self.current_center = None  # current fixation (x,y)
        self.timestep = 0

    # --------------------------
    # Public API (run_one_step)
    # --------------------------
    def run_one_step(self, grad, hue, curvature_bank, aspect_bank, orient_bank,
                    reset_image=True, ts=None):

        if ts is None:
            ts = self.timestep
        self.timestep += 1

        if reset_image:
            self.saccade.reset(grad, hue)

        feats = self.extract_features(
            grad, hue, curvature_bank, aspect_bank, orient_bank, ts
        )
        self.last_feats = feats

        mode, center, r = self.saccade.step()

        if mode == 'end':
            return 'end', [], None

        if center is None or r is None:
            return mode, [], None

        window_feats = [
            f for f in feats
            if abs(f['pos'][0] - center[0]) <= r and
            abs(f['pos'][1] - center[1]) <= r
        ]
        print(len(window_feats))

        activated_nodes, proto_matches, S = self.retrieve_candidates(window_feats, ts)

        if S >= self.cfg.strengthen_threshold:
            self.apply_strengthen(activated_nodes, proto_matches, ts)
            return 'strengthen', list(activated_nodes), center
        else:
            feats_for_learning = self._limit_feats_per_modality(
                window_feats, self.cfg.max_writes_per_modality
            )
            print(len(feats_for_learning))
            self.learn(feats_for_learning, set(), [], ts)
            return 'learn', [], center

    # --------------------------
    # Feature extraction (full image)
    # --------------------------
    def extract_features(self, grad, hue, curvature_bank, aspect_bank, orient_bank, ts:int=0) -> List[Dict[str,Any]]:
        """
        Extract candidate features across modalities (edge, hue, curvature, aspect, orient).
        Append embeddings and update descriptor maps & counts.
        """
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
    # Retrieval functions (coarse->fine)
    # --------------------------
    def retrieve_candidates(self, feats:List[Dict[str,Any]], ts:int=0):
        # reuse same retrieve implementation as earlier; returns (activated_nodes, proto_matches, S)
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
    # Strengthen branch (unchanged)
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

    # --------------------------
    # Learning (slightly adapted)
    # --------------------------
    def learn(self, feats:List[Dict[str,Any]], activated_nodes:set, proto_matches:List[Tuple[int,int,float]], ts:int=0):
        ncols = len(self.graphI)
        new_nodes_by_collection = [ [] for _ in range(ncols) ]
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
            groups: Dict[Tuple[int,int], List[int]] = {}
            for nid in new_nodes_by_collection[idx]:
                n = coll.nodes.get(nid); 
                if n is None: continue
                b = coll._bucket_coord(n.pos[0], n.pos[1])
                groups.setdefault(b, []).append(nid)
            for bucket_key, members in groups.items():
                if len(members) >= self.cfg.proto_form_thresh:
                    pid = coll.add_proto(members)
                    p = coll.protos.get(pid)
                    p.count = 1; p.strength = 1.0; p.last_updated = ts
                    for other_pid, other_p in list(coll.protos.items()):
                        if other_pid == pid: continue
                        inter = (p.bitset & other_p.bitset).bit_count()
                        union = (p.bitset | other_p.bitset).bit_count()
                        ratio = inter / max(1, union)
                        if ratio >= self.cfg.proto_merge_thresh:
                            other_p.members = list(set(other_p.members) | set(p.members))
                            other_p.update_bitset = None
                            # merge: simplistic - just delete new proto (we don't recompute bitset for speed)
                            if pid in coll.protos: del coll.protos[pid]
                            break

    # --------------------------
    # Helpers: select radius by interest density & feats in window
    # --------------------------
    def _choose_radius_by_interest(self, center:Tuple[int,int]) -> Optional[int]:
        """
        For each candidate r in saccade_radii compute: sum(interest in 2r x 2r window) / area.
        Choose r that maximizes this value. If all windows sum ~0 return None.
        """
        H, W = self.saccade.interest.shape
        cx, cy = int(center[0]), int(center[1])
        best_r = None; best_score = -1.0
        for r in self.cfg.saccade_radii:
            half = r
            x0 = max(0, cx - half); x1 = min(W-1, cx + half)
            y0 = max(0, cy - half); y1 = min(H-1, cy + half)
            region = self.saccade.interest[y0:y1+1, x0:x1+1]
            if region.numel() == 0: continue
            s = float(region.sum().item())
            area = region.numel()
            score = s / area
            if score > best_score:
                best_score = score; best_r = r
        if best_score <= 0.0:
            return None
        return best_r

    def _feats_in_window(self, center:Tuple[int,int], r:int) -> List[Dict[str,Any]]:
        cx, cy = int(center[0]), int(center[1])
        res = []
        for f in self.last_feats:
            fx, fy = int(f['pos'][0]), int(f['pos'][1])
            if abs(fx - cx) <= r and abs(fy - cy) <= r:
                res.append(f)
        return res

    def _limit_feats_per_modality(self, feats:List[Dict[str,Any]], per_mod_limit:int) -> List[Dict[str,Any]]:
        grouped: Dict[int, List[Dict[str,Any]]] = {}
        for f in feats:
            grouped.setdefault(int(f['modality']), []).append(f)
        out = []
        for mod, fl in grouped.items():
            fl_sorted = sorted(fl, key=lambda x: x['value'], reverse=True)
            out.extend(fl_sorted[:per_mod_limit])
        return out

    # --------------------------
    # For persistence
    # --------------------------
    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path:str) -> 'FoveatedMemory':
        with open(path, 'rb') as f:
            return pickle.load(f)
'''
# =============================
# Quick smoke test
# =============================
if __name__ == '__main__':
    cfg = Config()
    cfg.H = 128; cfg.W = 128
    fm = FoveatedMemory(cfg)

    # synthetic grad & hue & banks
    H = cfg.H; W = cfg.W
    gx = torch.randn((H, W)) * 0.5
    gy = torch.randn((H, W)) * 0.5
    grad = torch.zeros((1,1,H,W,2)); grad[0,0,:,:,0] = gx; grad[0,0,:,:,1] = gy
    hue = torch.rand((1,1,H,W))
    cur = torch.randn((1,4,H,W)); asp = torch.randn((1,4,H,W)); ori = torch.randn((1,4,H,W))

    # start image (reset_image True) - first saccade
    mode, nodes, center = fm.run_one_step(grad, hue, cur, asp, ori, reset_image=True)
    print("Step1 mode:", mode, "center:", center, "nodes sample:", nodes[:10])

    # subsequent saccades on same image
    for i in range(5):
        mode, nodes, center = fm.run_one_step(grad, hue, cur, asp, ori, reset_image=False)
        print(f"Step {i+2}: mode={mode}, center={center}, #nodes={len(nodes)}")
'''

'''
改进:
    learn时的眼跳机制需要修改 现在是根据interest-path
        应该改成不同尺度下的interest-path相互竞争 
        将匹配时的下采样结果复用
    extract_features针对全图,改为window内或缓存或增量式
'''
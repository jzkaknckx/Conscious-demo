import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


# --------------------------------------------------
# Route A: Sparse Graph Memory + Per-modality Inverted Index
# Enhanced to ingest tensor outputs from RetinaModel and
# MultiScaleFeatureBank and produce per-pixel/modal features
# for retrieval and update.
# --------------------------------------------------

'''
网络任务: 
    特征编码->检索(降复杂度)->存储(更新)    整理(遗忘)
改进方向
    节点到底存储什么: 
    第一层图按感受野尺寸/位置分化
    图层和图层之间的信息呈递
'''


class SparseGraphMemory:
    """Sparse graph memory implementing Route A: modal-separated graphs + inverted indexes.

    This enhanced implementation accepts tensor outputs from the two upstream modules:
    - RetinaModel.forward -> returns (x, grad, h, diff, flowvelocity, cropped)
      where expected shapes (batch 1):.
        grad: [1, 1, H, W, 2]  (we interpret last dim as gx, gy)
        h:    [1, 1, H, W]
        diff: [1, C, H, W]
        flowvelocity: [1, C, H, W, 4] (we use first two as vx,vy)
    - MultiScaleFeatureBank.forward -> returns (curv_bank, aspect_bank, orient_bank)
      each: [1, S, H, W]

    Main new API helpers:
      - ingest_from_tensors(retina_outputs, msfb_outputs, stride=8, thresholds=...)
          builds a list of discrete per-location features (modality, signature, position)
      - retrieve/update operate using these generated discrete features.

    NOTE: This implementation targets clarity for debugging and unit tests.
    For large-scale use, replace python dicts and loops with more optimized data structures.
    """

    class Node:
        def __init__(self, node_id: int, modality: str, signature: Any, position: Tuple[float, float], vector: Optional[List[float]] = None):
            self.id = node_id
            self.modality = modality
            self.signature = signature  # discrete key or quantized hash
            self.position = position
            self.vector = vector  # optional continuous vector representation
            self.count = 1
            self.last_seen = time.time()

        def touch(self, pos: Tuple[float, float], alpha: float = 0.2):
            """Update node position with EMA and increment count."""
            x_old, y_old = self.position
            x_new = alpha * pos[0] + (1 - alpha) * x_old
            y_new = alpha * pos[1] + (1 - alpha) * y_old
            self.position = (x_new, y_new)
            self.count += 1
            self.last_seen = time.time()

    class Edge:
        def __init__(self, src: int, dst: int):
            self.src = src
            self.dst = dst
            self.weight = 1
            self.last_seen = time.time()

        def touch(self):
            self.weight += 1
            self.last_seen = time.time()

    class ModalGraph:
        def __init__(self, modality: str):
            self.modality = modality
            self.nodes: Dict[int, SparseGraphMemory.Node] = {}
            # adjacency: node_id -> dict(neighbor_id -> Edge)
            self.adj: Dict[int, Dict[int, SparseGraphMemory.Edge]] = {}

        def add_node(self, node: 'SparseGraphMemory.Node') -> None:
            self.nodes[node.id] = node
            self.adj[node.id] = {}

        def add_edge(self, src_id: int, dst_id: int) -> None:
            if src_id not in self.adj:
                self.adj[src_id] = {}
            if dst_id in self.adj[src_id]:
                self.adj[src_id][dst_id].touch()
            else:
                e = SparseGraphMemory.Edge(src_id, dst_id)
                self.adj[src_id][dst_id] = e

        def remove_node(self, node_id: int) -> None:
            # remove edges to/from node and delete node entry
            if node_id in self.adj:
                # remove outgoing
                del self.adj[node_id]
            # remove incoming
            for nid, neigh in list(self.adj.items()):
                if node_id in neigh:
                    del neigh[node_id]
            if node_id in self.nodes:
                del self.nodes[node_id]

    def __init__(self,
                 modalities: List[str],
                 spatial_threshold: float = 8.0,
                 pos_window: float = 300.0,
                 prune_interval: float = 60.0,
                 min_node_count: int = 1):
        """Create a SparseGraphMemory.

        Args:
            modalities: list of modality names (each gets its own modal graph and inverted index)
            spatial_threshold: max distance (pixels or units) to consider a node match
            pos_window: time window (seconds) used for timestamp-based pruning decisions
            prune_interval: how often (s) to run pruning automatically (if enable_auto_prune True)
            min_node_count: minimal count to keep node during pruning
        """
        self.modalities = modalities
        self.spatial_threshold = spatial_threshold
        self.pos_window = pos_window
        self.prune_interval = prune_interval
        self.min_node_count = min_node_count

        self.graphs: Dict[str, SparseGraphMemory.ModalGraph] = {m: SparseGraphMemory.ModalGraph(m) for m in modalities}
        # inverted_index: modality -> key -> set(node_ids)
        self.inverted_index: Dict[str, Dict[Any, Set[int]]] = {m: {} for m in modalities}
        # node id allocator
        self._next_node_id = 1
        # locks per modality
        self.locks: Dict[str, bool] = {m: False for m in modalities}
        # last prune time
        self._last_prune = time.time()
        # cross-modal co-occurrence map
        self.cross_modal_cooccur: Dict[Tuple[str, int], Dict[Tuple[str, int], int]] = {}

    # ---------------------------
    # Helper utilities
    # ---------------------------
    def _dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)

    def _quantize_signature(self, modality: str, signature: Any, bins: int = 16) -> Any:
        """Quantize or canonicalize signatures per modality.

        For discrete signatures, return identity.
        For continuous values, bin into `bins` buckets.

        This method is intended to be simple and easily replaceable by an external
        quantization or LSH function when needed.
        """
        if signature is None:
            return None
        # if already hashable and not a tensor
        if not isinstance(signature, torch.Tensor):
            return signature
        # tensor -> scalar or small vector
        if signature.numel() == 1:
            v = float(signature.item())
            # map to bin index
            idx = int(max(0, min(bins - 1, math.floor((v + 1.0) / 2.0 * bins))))
            return (modality, 'bin', idx)
        # if small vector, compute simple hash by rounding
        arr = signature.detach().cpu().numpy().ravel()
        # reduce to a few ints
        key = tuple([int(round(float(x) * 100)) for x in arr[:min(4, len(arr))]])
        return (modality, 'vec', key)

    # ---------------------------
    # Ingestion from upstream tensors
    # ---------------------------
    def ingest_from_tensors(self,
                             retina_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                             msfb_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                             stride: int = 8,
                             thresholds: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Convert upstream tensors into a list of discrete features for retrieval/update.

        Args:
            retina_outputs: tuple returned by RetinaModel.forward: (x, grad, h, diff, flowvelocity, cropped)
            msfb_outputs: tuple (curv_bank, aspect_bank, orient_bank)
            stride: spatial sampling stride (sample every `stride` pixels to limit features)
            thresholds: per-modality thresholds for activation (magnitudes). Keys: 'edge','hue','diff','flow','curv'

        Returns:
            features: list of dicts with keys: modality, signature, position (x,y), optional 'value' and 'scale'
        """
        if thresholds is None:
            thresholds = {'edge': 0.1, 'hue': 0.0, 'diff': 0.01, 'flow': 0.01, 'curv': 0.01, 'aspect': 0.01, 'orient': 0.0}

        x, grad, h, diff, flowvelocity, cropped = retina_outputs
        curv_bank, aspect_bank, orient_bank = msfb_outputs

        # assume batch size 1
        device = x.device
        features: List[Dict] = []

        # shapes sanity
        # grad: [1,1,H,W,2]
        # h: [1,1,H,W]
        # diff: [1,C,H,W]
        # flowvelocity: [1,C,H,W,4]
        # curv_bank: [1,S,H,W]

        # convert to CPU numpy for iteration when small; keep on torch if large
        # We'll operate on tensors but sample a grid with stride

        _, _, H, W = x.shape
        _, _, Hc, Wc = curv_bank.shape
        assert H == Hc and W == Wc, 'spatial dims of retina and msfb must match'

        # squeeze batch dims
        grad_s = grad.squeeze(0)  # [1,H,W,2] or [1,H,W,2] if channel dim present
        if grad_s.ndim == 4:
            # grad shape [1,H,W,2]
            grad_s = grad_s.squeeze(0)
        # now grad_s [H,W,2]
        h_s = h.squeeze(0).squeeze(0)  # [H,W]
        diff_s = diff.squeeze(0)  # [C,H,W]
        flow_s = flowvelocity.squeeze(0)
        curv_s = curv_bank.squeeze(0)  # [S,H,W]
        aspect_s = aspect_bank.squeeze(0)
        orient_s = orient_bank.squeeze(0)

        # precompute flow magnitude from first two channels if present
        # flow_s may have shape [C,H,W,4] or [C,H,W,4], but we expect last dim axis at end
        if flow_s.ndim == 4 and flow_s.shape[-1] == 4:
            # flow_s: [C,H,W,4] -> take first channel vector components
            vx = flow_s[..., 0]
            vy = flow_s[..., 1]
            flow_mag = torch.sqrt(vx * vx + vy * vy)
            # collapse channel dim by max over channels
            if flow_mag.ndim == 3:
                flow_mag = torch.max(flow_mag, dim=0)[0]
        else:
            # fallback if shape different
            flow_mag = torch.zeros((H, W), device=device)

        # edge: compute magnitude from grad (assume gx,gy)
        if grad_s.ndim == 3 and grad_s.shape[-1] == 2:
            gx = grad_s[..., 0]
            gy = grad_s[..., 1]
            edge_mag = torch.sqrt(gx * gx + gy * gy)
            edge_orient = torch.atan2(gy, gx)  # radians
        else:
            edge_mag = torch.zeros((H, W), device=device)
            edge_orient = torch.zeros((H, W), device=device)

        # diff magnitude: take max across color channels
        if diff_s.ndim == 3:
            diff_mag = torch.max(torch.abs(diff_s), dim=0)[0]
        else:
            diff_mag = torch.zeros((H, W), device=device)

        # iterate sparse grid
        for y in range(0, H, stride):
            for xcoord in range(0, W, stride):
                pos = (float(xcoord), float(y))
                # EDGE
                em = float(edge_mag[y, xcoord].item())
                if em >= thresholds.get('edge', 0.1):
                    # quantize orientation into bins
                    orient = float(edge_orient[y, xcoord].item())
                    bin_idx = int((orient + math.pi) / (2 * math.pi) * 8)  # 8 bins
                    sig = ('edge', bin_idx)
                    features.append({'modality': 'edge', 'signature': sig, 'position': pos, 'value': em})

                # HUE
                hue_val = float(h_s[y, xcoord].item())
                # assume hue in [0,1]
                if abs(hue_val) >= thresholds.get('hue', 0.0):
                    hue_bin = int(hue_val * 16)  # 16 bins
                    sig = ('hue', hue_bin)
                    features.append({'modality': 'hue', 'signature': sig, 'position': pos, 'value': hue_val})

                # DIFF
                dm = float(diff_mag[y, xcoord].item())
                if dm >= thresholds.get('diff', 0.01):
                    sig = ('diff', int(min(255, int(dm * 255))))
                    features.append({'modality': 'diff', 'signature': sig, 'position': pos, 'value': dm})

                # FLOW
                fm = float(flow_mag[y, xcoord].item())
                if fm >= thresholds.get('flow', 0.01):
                    # direction bin as well if available
                    # here we approximate using vx,vy first channel if present
                    try:
                        vxv = float(vx[:, y, xcoord].max().item()) if 'vx' in locals() else 0.0
                        vyv = float(vy[:, y, xcoord].max().item()) if 'vy' in locals() else 0.0
                        ang = math.atan2(vyv, vxv)
                        ang_bin = int((ang + math.pi) / (2 * math.pi) * 8)
                    except Exception:
                        ang_bin = 0
                    sig = ('flow', ang_bin)
                    features.append({'modality': 'flow', 'signature': sig, 'position': pos, 'value': fm})

                # Multi-scale banks: curv/aspect/orient have S channels
                S = curv_s.shape[0]
                for s in range(S):
                    cv = float(curv_s[s, y, xcoord].item())
                    if cv >= thresholds.get('curv', 0.01):
                        sig = ('curv', s, int(min(255, int(cv * 255))))
                        features.append({'modality': 'curv', 'signature': sig, 'position': pos, 'value': cv, 'scale': s})
                    av = float(aspect_s[s, y, xcoord].item())
                    if abs(av) >= thresholds.get('aspect', 0.01):
                        sig = ('aspect', s, int(min(255, int((av + 1.0) * 127))))
                        features.append({'modality': 'aspect', 'signature': sig, 'position': pos, 'value': av, 'scale': s})
                    ov = float(orient_s[s, y, xcoord].item())
                    if abs(ov) >= thresholds.get('orient', 0.0):
                        # orientation normalized [-1,1], quantize to 8 bins
                        obin = int((ov + 1.0) / 2.0 * 8)
                        sig = ('orient', s, obin)
                        features.append({'modality': 'orient', 'signature': sig, 'position': pos, 'value': ov, 'scale': s})

        return features

    # ---------------------------
    # Core ops: retrieve and update
    # ---------------------------
    def retrieve(self, input_features: List[Dict], top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """Retrieve candidate nodes per modality for given input features.

        input_features: list of dicts, each with fields:
           - modality: str
           - signature: hashable key (discrete) OR 'vector' provided
           - position: (x,y)
           - optionally 'value' and 'scale'

        Returns dict: modality -> list of (node_id, score) sorted by score desc.
        Score combines signature match (binary) and spatial proximity.
        """
        candidates: Dict[str, Dict[int, float]] = {m: {} for m in self.modalities}

        for feat in input_features:
            modality = feat['modality']
            if modality not in self.modalities:
                continue
            sig = self._quantize_signature(modality, feat.get('signature'))
            pos = feat.get('position', (0.0, 0.0))

            postings = self.inverted_index.get(modality, {}).get(sig, set())
            # if postings empty, we still may want to return nearby nodes (optional)
            for nid in postings:
                node = self.graphs[modality].nodes.get(nid)
                if node is None:
                    continue
                d = self._dist(pos, node.position)
                spatial_score = max(0.0, 1.0 - d / (self.spatial_threshold + 1e-6))
                # signature match gives base score 1, spatial reduces by distance
                score = 1.0 * spatial_score
                # optionally incorporate node.count or feature value
                if 'value' in feat:
                    score *= (1.0 + min(5.0, float(feat['value']) * 10.0))
                prev = candidates[modality].get(nid, 0.0)
                if score > prev:
                    candidates[modality][nid] = score

        # convert to sorted lists
        out: Dict[str, List[Tuple[int, float]]] = {}
        for m, cmap in candidates.items():
            lst = sorted(cmap.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            out[m] = lst
        return out

    def update(self, input_features: List[Dict], allow_update: bool = True, pos_alpha: float = 0.2, create_if_missing: bool = True) -> None:
        """Update memory with given features. This runs recognition+update synchronously.

        For all active features in the same input, we also update edges between matched/created nodes.
        """
        # early exit if all modalities locked or updates disabled
        if not allow_update:
            return

        # per-modality matched node ids
        matched_nodes_per_mod: Dict[str, List[int]] = {m: [] for m in self.modalities}

        # first pass: match or create nodes per modality
        for feat in input_features:
            modality = feat['modality']
            if modality not in self.modalities:
                continue
            if self.locks.get(modality, False):
                # skip updates for this modality
                continue
            raw_sig = feat.get('signature')
            sig = self._quantize_signature(modality, raw_sig)
            pos = feat.get('position', (0.0, 0.0))

            postings = self.inverted_index.get(modality, {}).get(sig, set())
            best_nid = None
            best_score = 0.0
            for nid in postings:
                node = self.graphs[modality].nodes.get(nid)
                if node is None:
                    continue
                d = self._dist(pos, node.position)
                if d <= self.spatial_threshold and node.count > 0:
                    # score by proximity and count
                    score = (1.0 - d / (self.spatial_threshold + 1e-6)) * (1.0 + math.log(1 + node.count))
                    if score > best_score:
                        best_score = score
                        best_nid = nid

            if best_nid is not None:
                # update existing node
                node = self.graphs[modality].nodes[best_nid]
                node.touch(pos, alpha=pos_alpha)
                matched_nodes_per_mod[modality].append(best_nid)
            else:
                # create new node
                if create_if_missing:
                    nid = self._next_node_id
                    self._next_node_id += 1
                    node = SparseGraphMemory.Node(nid, modality, sig, pos, vector=feat.get('value'))
                    self.graphs[modality].add_node(node)
                    # update inverted index
                    inv = self.inverted_index[modality]
                    inv.setdefault(sig, set()).add(nid)
                    matched_nodes_per_mod[modality].append(nid)

        # second pass: update edges between simultaneously active nodes (across modalities)
        active_nodes: List[Tuple[str, int]] = []
        for mod, lst in matched_nodes_per_mod.items():
            for nid in lst:
                active_nodes.append((mod, nid))

        # intra-modal and cross-modal co-occurrence
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                mod_i, nid_i = active_nodes[i]
                mod_j, nid_j = active_nodes[j]
                if mod_i == mod_j:
                    g = self.graphs[mod_i]
                    g.add_edge(nid_i, nid_j)
                    g.add_edge(nid_j, nid_i)
                # update cross-modal cooccur map
                key_a = (mod_i, nid_i)
                key_b = (mod_j, nid_j)
                self.cross_modal_cooccur.setdefault(key_a, {})
                self.cross_modal_cooccur.setdefault(key_b, {})
                self.cross_modal_cooccur[key_a].setdefault(key_b, 0)
                self.cross_modal_cooccur[key_b].setdefault(key_a, 0)
                self.cross_modal_cooccur[key_a][key_b] += 1
                self.cross_modal_cooccur[key_b][key_a] += 1

    # ---------------------------
    # Maintenance: pruning, lock control, serialization
    # ---------------------------
    def prune(self, now: Optional[float] = None) -> None:
        """Remove low-count or old nodes and weak edges."""
        if now is None:
            now = time.time()
        for mod, g in self.graphs.items():
            # remove nodes with count < min_node_count or old
            for nid, node in list(g.nodes.items()):
                age = now - node.last_seen
                if node.count < self.min_node_count or age > self.pos_window:
                    g.remove_node(nid)
                    # remove from inverted index
                    inv = self.inverted_index[mod]
                    sig = node.signature
                    if sig in inv and nid in inv[sig]:
                        inv[sig].remove(nid)
                        if len(inv[sig]) == 0:
                            del inv[sig]
            # prune weak edges
            for nid, neigh in list(g.adj.items()):
                for nb, e in list(neigh.items()):
                    if e.weight <= 0:
                        del neigh[nb]

        # prune cross_modal_cooccur
        for k, v in list(self.cross_modal_cooccur.items()):
            for kk in list(v.keys()):
                if v[kk] <= 0:
                    del v[kk]
            if len(v) == 0:
                del self.cross_modal_cooccur[k]

    def lock_modality(self, modality: str) -> None:
        if modality in self.locks:
            self.locks[modality] = True

    def unlock_modality(self, modality: str) -> None:
        if modality in self.locks:
            self.locks[modality] = False

    # ---------------------------
    # Debug & inspection helpers
    # ---------------------------
    def inspect_modal(self, modality: str, limit: int = 10) -> Dict:
        """Return summary of a modality graph for debugging."""
        if modality not in self.modalities:
            return {}
        g = self.graphs[modality]
        nodes = list(g.nodes.values())[:limit]
        node_info = [{
            'id': n.id, 'sig': n.signature, 'pos': n.position, 'count': n.count
        } for n in nodes]
        edges = []
        for nid, neigh in list(g.adj.items())[:limit]:
            for nb, e in list(neigh.items())[:limit]:
                edges.append({'src': nid, 'dst': nb, 'w': e.weight})
        return {'nodes': node_info, 'edges': edges}


# ----------------------------
# Demo usage helper (not run automatically)
# ----------------------------

def demo_with_tensors():
    # create fake upstream tensors with small H,W for demo
    B = 1
    H = 32
    W = 32
    device = torch.device('cpu')

    x = torch.randn(B, 3, H, W, device=device)
    # grad: [B,1,H,W,2]
    gx = torch.randn(B, 1, H, W, 2, device=device)
    # hue: values in [0,1]
    h = torch.rand(B, 1, H, W, device=device)
    diff = torch.randn(B, 3, H, W, device=device) * 0.05
    # flowvelocity: [B, C, H, W, 4]
    flow = torch.randn(B, 1, H, W, 4, device=device) * 0.1

    # MultiScale banks S=2
    curv = torch.abs(torch.randn(B, 2, H, W, device=device))
    aspect = torch.randn(B, 2, H, W, device=device)
    orient = torch.tanh(torch.randn(B, 2, H, W, device=device))

    retina_out = (x, gx, h, diff, flow, x)
    msfb_out = (curv, aspect, orient)

    mem = SparseGraphMemory(modalities=['edge', 'hue', 'diff', 'flow', 'curv', 'aspect', 'orient'], spatial_threshold=5.0)
    feats = mem.ingest_from_tensors(retina_out, msfb_out, stride=8)
    print('Generated features:', len(feats))
    # retrieve candidates
    cands = mem.retrieve(feats[:20], top_k=5)
    print('Candidates before update:', {k: len(v) for k, v in cands.items()})
    mem.update(feats, allow_update=True)
    cands2 = mem.retrieve(feats[:20], top_k=5)
    print('Candidates after update:', {k: len(v) for k, v in cands2.items()})
    print('Inspect edge modality:', mem.inspect_modal('edge'))


if __name__ == '__main__':
    demo_with_tensors()

'''
加入:
设定阈值,首先检索,大于阈值检测通过;小于阈值更新记忆,加入新节点

'''
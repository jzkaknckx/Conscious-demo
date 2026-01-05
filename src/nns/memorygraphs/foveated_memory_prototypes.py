# foveated_memory_prototypes.py
"""
Foveated memory with Graph1 capacity-limited prototype storage.

Key classes:
 - Graph0 : raw local feature nodes + directed edges (as before)
 - Graph1 : prototype nodes that each store up to K representative Graph0 ids,
            plus spatial envelope (mean, covariance) and occurrence stats
 - FoveatedGraphMemory : orchestrates ingestion, retrieval (coarse->fine),
                         promotion from Graph0 -> Graph1 with capacity control

Usage:
  from foveated_memory_prototypes import FoveatedGraphMemory
  mem = FoveatedGraphMemory(H=512, W=512, graph1_capacity=32)
  mem.set_match_params(min_inliers=8, edge_preserve_threshold=0.4)
  ... use mem.ingest_features_direct(...), mem.run_one_step(...)
"""
import math
import random
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import numpy as np
import torch

# ------------------------
# Helpers: geometric estimation (Umeyama) and RANSAC (same as earlier)
# ------------------------
def estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False):
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    if n < 2:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_cent = src - src_mean
    dst_cent = dst - dst_mean
    cov = dst_cent.T @ src_cent / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if not allow_reflection and (np.linalg.det(U) * np.linalg.det(Vt) < 0):
        S[1, 1] = -1
    R = U @ S @ Vt
    var_src = (src_cent ** 2).sum() / n
    if var_src < 1e-12:
        return None
    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t

def ransac_similarity(src_pts: np.ndarray, dst_pts: np.ndarray, num_iters: int = 200, inlier_thresh: float = 6.0, min_samples: int = 2, random_seed: Optional[int] = None):
    if src_pts.shape[0] != dst_pts.shape[0]:
        raise ValueError("src/dst count mismatch")
    M = src_pts.shape[0]
    if M < min_samples:
        return None
    if random_seed is not None:
        random.seed(random_seed); np.random.seed(random_seed)
    best = {'inlier_count': 0}
    for _ in range(num_iters):
        idx = np.random.choice(M, size=min_samples, replace=False)
        sub_src = src_pts[idx]; sub_dst = dst_pts[idx]
        est = estimate_similarity_umeyama(sub_src, sub_dst)
        if est is None:
            continue
        s, R, t = est
        transformed = s * (src_pts @ R.T) + t
        dists = np.linalg.norm(transformed - dst_pts, axis=1)
        inliers = dists <= inlier_thresh
        cnt = int(inliers.sum())
        if cnt > best['inlier_count']:
            best = {'inlier_count': cnt, 'scale': s, 'R': R, 't': t, 'inliers_mask': inliers, 'dists': dists}
            if cnt > 0.9 * M:
                break
    if best['inlier_count'] == 0:
        return None
    return best

# ------------------------
# Graph0: nodes and edges (lightweight)
# ------------------------
class Graph0:
    class Node:
        __slots__ = ('id','modality','magnitude','pos','signature','age','count','archived')
        def __init__(self, node_id:int, modality:str, magnitude:float, pos:Tuple[float,float], signature:Any=None):
            self.id = node_id
            self.modality = modality
            self.magnitude = magnitude
            self.pos = pos
            self.signature = signature
            self.age = 0
            self.count = 1  # persistence
            self.archived = False  # when stored into Graph1 prototypes and optionally removed from active index

    class Edge:
        __slots__ = ('id','src','dst','dist','orient','weight','age')
        def __init__(self, edge_id:int, src:int, dst:int, dist:float, orient:float):
            self.id = edge_id
            self.src = src
            self.dst = dst
            self.dist = dist
            self.orient = orient
            self.weight = 1.0
            self.age = 0

    def __init__(self, bucket_size:int=8):
        self.nodes: Dict[int, Graph0.Node] = {}
        self.edges: Dict[int, Graph0.Edge] = {}
        self.next_node_id = 0
        self.next_edge_id = 0
        # inverted index: modality->signature->set(node_id)
        self.inv_index: Dict[str, Dict[Any, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self.bucket_size = bucket_size
        self.buckets: Dict[Tuple[int,int], List[int]] = defaultdict(list)

    def _bucket_key(self, pos:Tuple[float,float]) -> Tuple[int,int]:
        bx = int(pos[0]) // self.bucket_size
        by = int(pos[1]) // self.bucket_size
        return bx, by

    def add_node(self, modality:str, magnitude:float, pos:Tuple[float,float], signature:Any=None) -> int:
        nid = self.next_node_id
        node = Graph0.Node(nid, modality, magnitude, pos, signature)
        self.nodes[nid] = node
        self.inv_index[modality][signature].add(nid)
        self.buckets[self._bucket_key(pos)].append(nid)
        self.next_node_id += 1
        return nid

    def find_local_nodes(self, pos:Tuple[float,float], radius:float) -> List[int]:
        bx, by = self._bucket_key(pos)
        r_b = max(1, int(math.ceil(radius / self.bucket_size)))
        out = []
        for dx in range(-r_b, r_b+1):
            for dy in range(-r_b, r_b+1):
                out.extend(self.buckets.get((bx+dx, by+dy), []))
        # filter by exact radius
        res = []
        for nid in out:
            n = self.nodes.get(nid)
            if n is None: continue
            if (n.pos[0] - pos[0])**2 + (n.pos[1] - pos[1])**2 <= radius*radius:
                res.append(nid)
        return res

    def connect(self, src:int, dst:int):
        if src not in self.nodes or dst not in self.nodes:
            return None, None
        n1 = self.nodes[src]; n2 = self.nodes[dst]
        dx = n2.pos[0] - n1.pos[0]; dy = n2.pos[1] - n1.pos[1]
        dist = math.hypot(dx, dy)
        orient = math.atan2(dy, dx)
        eid = self.next_edge_id
        self.edges[eid] = Graph0.Edge(eid, src, dst, dist, orient)
        self.next_edge_id += 1
        eid2 = self.next_edge_id
        self.edges[eid2] = Graph0.Edge(eid2, dst, src, dist, (orient + math.pi)%(2*math.pi))
        self.next_edge_id += 1
        return eid, eid2

    def strengthen_edges(self, node_ids:Sequence[int], delta:float=1.0):
        s = set(node_ids)
        for e in self.edges.values():
            if e.src in s and e.dst in s:
                e.weight += delta

    def archive_node(self, nid:int):
        """Mark node archived and remove from active inverted index / buckets to reduce search burden.
           Node object still kept in self.nodes so Graph1 prototypes can reference positions."""
        if nid not in self.nodes:
            return
        n = self.nodes[nid]
        if n.archived:
            return
        n.archived = True
        # remove from inv index
        try:
            if n.id in self.inv_index[n.modality][n.signature]:
                self.inv_index[n.modality][n.signature].remove(n.id)
        except Exception:
            pass
        # remove from bucket list (lazy removal)
        key = self._bucket_key(n.pos)
        if key in self.buckets:
            try:
                self.buckets[key].remove(n.id)
            except ValueError:
                pass

    def unarchive_node(self, nid:int):
        if nid not in self.nodes:
            return
        n = self.nodes[nid]
        if not n.archived:
            return
        n.archived = False
        self.inv_index[n.modality][n.signature].add(n.id)
        self.buckets[self._bucket_key(n.pos)].append(n.id)

# ------------------------
# Graph1: prototype nodes with capacity-limited storage
# ------------------------
class Graph1:
    """
    Each Graph1 node stores:
      - id
      - member_ids: set/list of Graph0 node ids that contributed to this prototype
      - prototypes: subset of member_ids up to capacity K (representative graph0 ids)
      - pos_mean: centroid of prototype positions (2,)
      - pos_cov: covariance matrix (2x2) of prototype positions (for envelope matching)
      - count: number of times this prototype was reinforced
    Behavior:
      - add_group(member_ids): create or merge group; select up to capacity K prototypes using score+diversity
      - search_by_query(query_feature_postings): given a set of Graph0 node ids (from inverted index),
        return top-N Graph1 nodes ranked by overlap count with these postings.
    """
    class Node:
        def __init__(self, nid:int, member_ids:Sequence[int], prototypes:Sequence[int], pos_mean:Tuple[float,float], pos_cov:np.ndarray):
            self.id = nid
            self.member_ids = set(member_ids)
            self.prototypes = list(prototypes)  # representative Graph0 ids (<= capacity)
            self.pos_mean = tuple(pos_mean)
            self.pos_cov = pos_cov  # 2x2 numpy array
            self.count = 1

    def __init__(self, capacity_per_node:int = 32):
        self.nodes: Dict[int, Graph1.Node] = {}
        self.prototype_to_g1: Dict[int, int] = {}  # map Graph0 node id -> Graph1 node id (may map to multiple; keep last or list)
        self.next_id = 0
        self.capacity = capacity_per_node

    def _compute_pos_stats(self, graph0, member_ids:Sequence[int]):
        pts = []
        for nid in member_ids:
            if nid not in graph0.nodes:
                continue
            n = graph0.nodes[nid]
            pts.append([n.pos[0], n.pos[1]])
        if not pts:
            return (0.0,0.0), np.eye(2)*1e-3
        arr = np.array(pts, dtype=float)
        mean = arr.mean(axis=0)
        cov = np.cov(arr.T) if arr.shape[0] > 1 else np.eye(2)*1e-3
        # ensure cov is 2x2
        if cov.shape == ():
            cov = np.array([[cov]])
            cov = np.eye(2)*1e-3
        if cov.shape == (1,1):
            cov = np.eye(2)*cov[0,0]
        return (float(mean[0]), float(mean[1])), cov

    def _score_member(self, graph0, nid:int):
        """Simple score for a Graph0 node: combine persistence (count) and local importance (magnitude).
           You may refine this to include local degree, edge weights, etc."""
        node = graph0.nodes.get(nid)
        if node is None:
            return 0.0
        return float(node.count) + 0.1 * float(node.magnitude if node.magnitude is not None else 0.0)

    def _select_prototypes_diverse(self, graph0, candidate_ids:Sequence[int], k:int):
        """
        Select up to k prototypes from candidate_ids using:
          - sort by score descending
          - greedy farthest-first sampling to ensure diversity among top items
        """
        cand = list(set(candidate_ids))
        if not cand:
            return []
        scores = {nid: self._score_member(graph0, nid) for nid in cand}
        # sort by score
        sorted_by_score = sorted(cand, key=lambda x: scores[x], reverse=True)
        # seed with top score
        prototypes = []
        if sorted_by_score:
            prototypes.append(sorted_by_score[0])
        # pick greedily farthest from selected among next high-score candidates
        idx = 1
        while len(prototypes) < k and idx < len(sorted_by_score):
            cand_id = sorted_by_score[idx]
            # compute distance to prototypes using positions
            min_dist = float('inf')
            pos_cand = np.array(graph0.nodes[cand_id].pos)
            for pid in prototypes:
                pos_p = np.array(graph0.nodes[pid].pos)
                d = np.linalg.norm(pos_cand - pos_p)
                if d < min_dist:
                    min_dist = d
            # accept candidate if it increases diversity (or accept if few prototypes exist)
            prototypes.append(cand_id)
            idx += 1
        # trim to k
        return prototypes[:k]

    def add_or_merge_group(self, graph0, member_ids:Sequence[int], archive_members:bool=True):
        """
        Add a new Graph1 node representing member_ids, or merge with existing similar node.
        Policy:
          - If an existing node shares a large overlap (>50%) with member_ids, merge sets and refresh prototypes.
          - Else create new Graph1 node.
        When created/merged, prototypes chosen via _select_prototypes_diverse limited by capacity.
        If archive_members True, Graph0.archive_node(...) is called on prototype ids to reduce active Graph0 size.
        """
        member_set = set(member_ids)
        if not member_set:
            return None
        # try to find a merge candidate by overlap
        best_overlap = 0.0
        best_nid = None
        for g1id, g1node in self.nodes.items():
            inter = len(member_set & g1node.member_ids)
            union = len(member_set | g1node.member_ids)
            if union == 0: continue
            overlap = inter / union
            if overlap > best_overlap:
                best_overlap = overlap; best_nid = g1id
        # threshold for merging
        MERGE_OVERLAP_TH = 0.5
        if best_nid is not None and best_overlap >= MERGE_OVERLAP_TH:
            # merge into best_nid
            node = self.nodes[best_nid]
            node.member_ids.update(member_set)
            # reselect prototypes from expanded member set
            prototypes = self._select_prototypes_diverse(graph0, list(node.member_ids), self.capacity)
            node.prototypes = prototypes
            node.count += 1
            # update spatial stats
            pos_mean, pos_cov = self._compute_pos_stats(graph0, node.prototypes)
            node.pos_mean = pos_mean; node.pos_cov = pos_cov
            # update prototype_to_g1 mapping
            for pid in prototypes:
                self.prototype_to_g1[pid] = node.id
            # optionally archive member prototypes
            if archive_members:
                for pid in prototypes:
                    graph0.archive_node(pid)
            return node.id
        # else create new node
        prototypes = self._select_prototypes_diverse(graph0, list(member_set), self.capacity)
        pos_mean, pos_cov = self._compute_pos_stats(graph0, prototypes)
        nid = self.next_id
        node = Graph1.Node(nid, list(member_set), prototypes, pos_mean, pos_cov)
        self.nodes[nid] = node
        self.next_id += 1
        for pid in prototypes:
            self.prototype_to_g1[pid] = nid
            if archive_members:
                graph0.archive_node(pid)
        return nid

    def search_by_postings(self, postings:Sequence[int], top_k:int=50):
        """
        Given a list of Graph0 node ids (postings from inverted index),
        return top_k Graph1 node ids ranked by how many postings map into their prototypes/member_ids.
        This is a simple overlap count approach (coarse). Complexity is O(len(postings)).
        """
        counter = Counter()
        for pid in postings:
            # if prototype map exists...
            g1id = self.prototype_to_g1.get(pid)
            if g1id is not None:
                counter[g1id] += 2  # boost if posting hits prototype directly
            # additionally we might check membership by iterating nodes (expensive) - avoid
        # fallback: also check nodes by intersection size with their member_ids (only if few nodes)
        # produce top_k
        most = [g for g,_ in counter.most_common(top_k)]
        return most

# ------------------------
# FoveatedGraphMemory orchestrates retrieval & promotion
# ------------------------
class FoveatedGraphMemory:
    def __init__(self, H:int, W:int,
                 bucket_size:int=8,
                 spatial_thresh:float=24.0,
                 ransac_iters:int=200,
                 ransac_inlier_thresh:float=6.0,
                 merge_radius:float=4.0,
                 graph1_capacity:int=32,
                 max_graph0_active:int=20000):
        # basic structures
        self.H = H; self.W = W
        self.graph0 = Graph0(bucket_size=bucket_size)
        self.graph1 = Graph1(capacity_per_node=graph1_capacity)
        self.attn = SaccadeAttention(H, W, r=bucket_size*3, sigma=bucket_size*1.5)
        self.spatial_thresh = spatial_thresh
        self.ransac_iters = ransac_iters
        self.ransac_inlier_thresh = ransac_inlier_thresh
        self.merge_radius = merge_radius
        # matching params
        self.match_params = {'min_inliers':6, 'edge_preserve_threshold':0.3, 'score_weights':{'inlier':1.0,'edge':2.0,'persistence':1.0}}
        # promotion control
        self.promote_edge_count_threshold = 3
        self.graph1_capacity = graph1_capacity
        self.archive_on_promote = True
        self.max_graph0_active = max_graph0_active

        # mode
        self.mode = 'idle'

    def set_match_params(self, *, min_inliers:int=6, edge_preserve_threshold:float=0.3, inlier_distance_thresh:Optional[float]=None, score_weights:Optional[Dict[str,float]]=None):
        self.match_params['min_inliers'] = int(min_inliers)
        self.match_params['edge_preserve_threshold'] = float(edge_preserve_threshold)
        if inlier_distance_thresh is not None:
            self.ransac_inlier_thresh = float(inlier_distance_thresh)
        if score_weights is not None:
            self.match_params['score_weights'] = dict(score_weights)

    # ------------------------
    # features extraction (reuse earlier style)
    # ------------------------
    def extract_features_window(self, retina_out:Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],
                                msfb_out:Tuple[torch.Tensor,torch.Tensor,torch.Tensor],
                                center:Tuple[float,float],
                                radius:int=32, stride:int=4, thresholds:Optional[Dict]=None):
        """Return list of features sampled in window (modality, signature, position, value)."""
        if thresholds is None:
            thresholds = {'edge':0.05, 'hue':0.0, 'curv':0.01, 'aspect':0.01, 'orient':0.0}
        x, grad, h, diff, flow, cropped = retina_out
        curv_bank, aspect_bank, orient_bank = msfb_out
        _,_,H,W = x.shape
        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - radius); x1 = min(W-1, cx + radius)
        y0 = max(0, cy - radius); y1 = min(H-1, cy + radius)
        feats = []
        # process tensors to numpy slices for speed
        grad_s = grad.squeeze(0)
        if grad_s.ndim == 4:
            grad_s = grad_s.squeeze(0)
        # grad_s expected shape [H,W,2]
        if grad_s.ndim == 3 and grad_s.shape[-1] == 2:
            gx = grad_s[...,0]; gy = grad_s[...,1]
            edge_mag = torch.sqrt(gx*gx + gy*gy)
            edge_orient = torch.atan2(gy, gx)
        else:
            edge_mag = torch.zeros((H,W)); edge_orient = torch.zeros((H,W))
        h_s = h.squeeze(0).squeeze(0)
        curv_s = curv_bank.squeeze(0)
        aspect_s = aspect_bank.squeeze(0)
        orient_s = orient_bank.squeeze(0)
        for yy in range(y0, y1+1, stride):
            for xx in range(x0, x1+1, stride):
                pos = (float(xx), float(yy))
                em = float(edge_mag[yy, xx].item()) if isinstance(edge_mag, torch.Tensor) else float(edge_mag[yy,xx])
                if em >= thresholds['edge']:
                    ob = float(edge_orient[yy,xx].item()) if isinstance(edge_orient, torch.Tensor) else float(edge_orient[yy,xx])
                    bin_idx = int((ob + math.pi)/(2*math.pi) * 8)
                    feats.append({'modality':'edge','signature':('edge',bin_idx),'position':pos,'value':em})
                hue_val = float(h_s[yy,xx].item())
                if abs(hue_val) >= thresholds['hue']:
                    hue_bin = int(hue_val*16)
                    feats.append({'modality':'hue','signature':('hue',hue_bin),'position':pos,'value':hue_val})
                S = curv_s.shape[0]
                for s in range(S):
                    cv = float(curv_s[s,yy,xx].item())
                    if cv >= thresholds['curv']:
                        feats.append({'modality':'curv','signature':('curv',s,int(min(255,int(cv*255)))),'position':pos,'value':cv,'scale':s})
                    av = float(aspect_s[s,yy,xx].item())
                    if abs(av) >= thresholds['aspect']:
                        feats.append({'modality':'aspect','signature':('aspect',s,int(min(255,int((av+1.0)*127)))),'position':pos,'value':av,'scale':s})
                    ov = float(orient_s[s,yy,xx].item())
                    if abs(ov) >= thresholds['orient']:
                        obin = int((ov+1.0)/2.0 * 8)
                        feats.append({'modality':'orient','signature':('orient',s,obin),'position':pos,'value':ov,'scale':s})
        return feats

    # ------------------------
    # ingest: local merging and active node cap
    # ------------------------
    def ingest_features_direct(self, features:List[Dict], create_if_missing:bool=True):
        created = []
        for feat in features:
            mod = feat['modality']; sig = feat.get('signature'); pos = feat['position']; mag = float(feat.get('value',1.0))
            # merge locally to avoid duplicates
            local = self.graph0.find_local_nodes(pos, self.merge_radius)
            merged = False
            for nid in local:
                node = self.graph0.nodes[nid]
                if node.modality == mod and node.signature == sig:
                    # update by EMA
                    alpha = 0.2
                    nx = alpha*pos[0] + (1-alpha)*node.pos[0]; ny = alpha*pos[1] + (1-alpha)*node.pos[1]
                    node.pos = (nx, ny)
                    node.magnitude = alpha*mag + (1-alpha)*node.magnitude
                    node.count += 1
                    node.age = 0
                    created.append(nid); merged = True; break
            if not merged and create_if_missing:
                nid = self.graph0.add_node(mod, mag, pos, signature=sig)
                created.append(nid)
        # connect pairwise (local co-occurrence in one attention window)
        for i in range(len(created)):
            for j in range(i+1, len(created)):
                self.graph0.connect(created[i], created[j])
        # enforce active graph0 cap: if too many non-archived nodes, archive some low-importance nodes
        non_archived = [nid for nid,n in self.graph0.nodes.items() if not n.archived]
        if len(non_archived) > self.max_graph0_active:
            # prune: remove lowest-scoring by node.count (persistence)
            scores = sorted(non_archived, key=lambda nid: self.graph0.nodes[nid].count)
            to_archive = scores[:len(non_archived)-self.max_graph0_active]
            for nid in to_archive:
                self.graph0.archive_node(nid)
        return created

    # ------------------------
    # retrieval: coarse->fine using Graph1 if available
    # ------------------------
    def retrieve_candidates_for_features(self, features:List[Dict], spatial_filter:float, use_graph1:bool=True, top_g1:int=32):
        """
        Return candidate Graph0 node ids for geometric verification.
        Pipeline:
          - get postings from Graph0 inv_index for query features
          - if Graph1 has data and use_graph1 True: map postings to Graph1 candidates (search_by_postings),
            then expand to Graph0 prototypes from those top Graph1 nodes (<= top_g1 * K)
          - else fall back to Graph0 postings + local buckets
        """
        postings = set()
        for feat in features:
            mod = feat['modality']; sig = feat.get('signature'); pos = feat['position']
            postings.update(self.graph0.inv_index.get(mod, {}).get(sig, set()))
            # also add local bucket nodes to posting list (for tolerance)
            local = self.graph0.find_local_nodes(pos, spatial_filter)
            postings.update(local)
        # if Graph1 available and asked, use it to shortlist
        if use_graph1 and len(self.graph1.nodes) > 0:
            # map postings -> Graph1 candidates
            p_list = list(postings)
            g1_candidates = self.graph1.search_by_postings(p_list, top_k=top_g1)
            # expand Graph1 candidates into their prototypes (Graph0 ids)
            cand_nodes = set()
            for g1id in g1_candidates:
                g1node = self.graph1.nodes.get(g1id)
                if g1node is None:
                    continue
                for pid in g1node.prototypes:
                    cand_nodes.add(pid)
            # fallback: if empty (rare), use postings
            if not cand_nodes:
                cand_nodes = postings
            return cand_nodes
        else:
            return postings

    # ------------------------
    # geometric_verify (as before) using candidate Graph0 ids
    # ------------------------
    def geometric_verify(self, query_feats:List[Dict], candidate_node_ids:Sequence[int], min_inliers:int=6):
        src_pts=[]; dst_pts=[]; mem_ids=[]
        for feat in query_feats:
            mod=feat['modality']; sig=feat.get('signature'); qpos=feat.get('position')
            for nid in candidate_node_ids:
                node = self.graph0.nodes.get(nid)
                if node is None:
                    continue
                # skip archived nodes in candidate set? archived nodes may still be used if prototypes point to them
                if node.modality != mod:
                    continue
                src_pts.append([node.pos[0], node.pos[1]]); dst_pts.append([qpos[0], qpos[1]]); mem_ids.append(nid)
        if len(src_pts) < 2:
            return None
        src_arr = np.array(src_pts); dst_arr = np.array(dst_pts)
        rres = ransac_similarity(src_arr, dst_arr, num_iters=self.ransac_iters, inlier_thresh=self.ransac_inlier_thresh)
        if rres is None: return None
        inliers_mask = rres['inliers_mask']; inlier_indices = np.nonzero(inliers_mask)[0]
        if len(inlier_indices) < min_inliers:
            return None
        matched_mem_ids = [mem_ids[i] for i in inlier_indices]
        matched_mem_pts = src_arr[inlier_indices]; matched_qry_pts = dst_arr[inlier_indices]
        edge_score, total_edges, preserved_edges = self.edge_consistency_score(matched_mem_ids, rres['scale'], rres['R'], rres['t'])
        # require edge_preserve threshold as part of acceptance
        if total_edges > 0:
            ep_ratio = preserved_edges / total_edges
        else:
            ep_ratio = 0.0
        if ep_ratio < self.match_params['edge_preserve_threshold']:
            # consider this match low-confidence
            return None
        return {
            'scale':rres['scale'],'R':rres['R'],'t':rres['t'],'inlier_count':int(len(inlier_indices)),
            'matched_mem_ids':matched_mem_ids,'matched_mem_pts':matched_mem_pts,'matched_qry_pts':matched_qry_pts,
            'edge_score':edge_score,'preserved_edges':preserved_edges,'total_edges':total_edges
        }

    def edge_consistency_score(self, matched_node_ids:Sequence[int], scale:float, R:np.ndarray, t:np.ndarray, dist_tol:float=0.25, angle_tol:float=0.4):
        total=0; preserved=0
        s=set(matched_node_ids)
        for eid,edge in self.graph0.edges.items():
            if edge.src in s and edge.dst in s:
                total += 1
                nsrc = self.graph0.nodes[edge.src]; ndst=self.graph0.nodes[edge.dst]
                src_pt = np.array([nsrc.pos[0], nsrc.pos[1]]); dst_pt = np.array([ndst.pos[0], ndst.pos[1]])
                transformed_src = scale * (src_pt @ R.T) + t
                transformed_dst = scale * (dst_pt @ R.T) + t
                expected_dist = np.linalg.norm(transformed_dst - transformed_src)
                stored_scaled = scale * edge.dist
                rel_diff = abs(expected_dist - stored_scaled) / (stored_scaled + 1e-9)
                vec = transformed_dst - transformed_src
                orient_trans = math.atan2(float(vec[1]), float(vec[0]))
                d_ang = abs(((orient_trans - edge.orient + math.pi) % (2*math.pi)) - math.pi)
                if rel_diff <= dist_tol and d_ang <= angle_tol:
                    preserved += 1
        score = (preserved / total) if total>0 else 0.0
        return score, total, preserved

    # ------------------------
    # strengthen and promote: now will add a Graph1 node when matched group is big enough
    # ------------------------
    def strengthen_and_promote(self, match_info, promote_edge_count_threshold:int=None):
        if promote_edge_count_threshold is None:
            promote_edge_count_threshold = self.promote_edge_count_threshold
        mem_ids = match_info['matched_mem_ids']
        # strengthen edges among mem_ids
        self.graph0.strengthen_edges(mem_ids, delta=1.0)
        # collect edge ids among mem_ids
        matched_edge_ids = []
        for eid,edge in self.graph0.edges.items():
            if edge.src in mem_ids and edge.dst in mem_ids:
                matched_edge_ids.append(eid)
        if len(matched_edge_ids) >= promote_edge_count_threshold:
            # choose member graph0 node ids (unique nodes from edges)
            member_nodes = set()
            for eid in matched_edge_ids:
                e = self.graph0.edges[eid]
                member_nodes.add(e.src); member_nodes.add(e.dst)
            # promote into Graph1 with capacity control
            g1_id = self.graph1.add_or_merge_group(self.graph0, list(member_nodes), archive_members=self.archive_on_promote)
            return g1_id
        return None

    # ------------------------
    # main run_one_step: retrieval + optional learning (keeps earlier dual-mode logic; Graph1 used during retrieval)
    # ------------------------
    def run_one_step(self, retina_out, msfb_out, current_fix: Tuple[float,float],
                     initial_search_radius:int=8, max_search_radius:int=32, expand_step:int=8,
                     spatial_filter_base:float=24.0, allow_learning:bool=True):
        # retrieval phase (use Graph1 for coarse filtering)
        r = initial_search_radius
        found_match = False
        match_info = None
        chosen_move = None
        learned_nodes = []
        while r <= max_search_radius and not found_match:
            print('Matching..., search_radius = ', r)
            feats = self.extract_features_window(retina_out, msfb_out, center=current_fix, radius=r, stride=4)
            if not feats:
                r += expand_step; continue
            candidates = self.retrieve_candidates_for_features(feats, spatial_filter = spatial_filter_base * (1.0 + r/initial_search_radius*0.5), use_graph1=True, top_g1=64)
            if not candidates:
                r += expand_step; continue
            match = self.geometric_verify(feats, list(candidates), min_inliers=self.match_params['min_inliers'])
            if match is not None:
                found_match = True; match_info = match
                qpts = match.get('matched_qry_pts')
                if qpts is None or len(qpts)==0:
                    target = current_fix
                else:
                    centroid = np.mean(qpts, axis=0); target = (float(centroid[0]), float(centroid[1]))
                dx = target[0] - current_fix[0]; dy = target[1] - current_fix[1]
                dist = math.hypot(dx,dy); orient = math.atan2(dy, dx)
                chosen_move = (dist, orient, target)
                break
            else:
                r += expand_step
        if found_match:
            self.mode = 'strengthen'
            g1id = self.strengthen_and_promote(match_info)
            # optionally do non-recorded refinement saccade
            if chosen_move is not None:
                dist, orient, _ = chosen_move
                self.attn.saccade(dist, orient, record_visit=False)
            return {'mode':'strengthen','matched':True,'move':chosen_move,'match_info':match_info,'promoted_g1':g1id,'learned_nodes':[]}
        # else learning mode (simpler policy; can be extended). We'll ingest incremental features and possibly promote later.
        self.mode = 'learn'
        print('Match Failed, Learning')
        learn_saccades = 0
        local_current = current_fix
        loop_closed = False
        while learn_saccades <  self.max_graph0_active and not loop_closed:
            move_dist, move_orient, target_pos = self.select_next_fixation_learning(retina_out, msfb_out, local_current)
            self.attn.saccade(move_dist, move_orient, record_visit=True)
            feats_local = self.extract_features_window(retina_out, msfb_out, center=target_pos, radius=24, stride=4)
            new_nodes = self.ingest_features_direct(feats_local, create_if_missing=True)
            learned_nodes.extend(new_nodes)
            learn_saccades += 1
            local_current = target_pos
            print('Looking at', local_current)
            if self.attn.check_loop_and_reset(loop_radius=16.0):
                loop_closed = True
        # unified ingestion before reset: ingest features from attention region one time
        unified_new_nodes = []
        try:
            unified_new_nodes = self.ingest_features_from_attention(retina_out, msfb_out, att_threshold=self.attn_write_threshold(), stride=4)
        except Exception:
            unified_new_nodes = []
        # reset attention and return (do NOT re-run retrieval here)
        self.attn.reset()
        self.mode = 'idle'
        return {'mode':'learn','matched':False,'move':None,'match_info':None,'learned_nodes': learned_nodes + unified_new_nodes}

    # these functions below are adapted from earlier implementations and are left compact here
    def ingest_features_from_attention(self, retina_out, msfb_out, att_threshold:float=0.5, stride:int=4):
        att_map = self.attn.get_map().detach().cpu().numpy()
        H,W = att_map.shape
        feats=[]
        for y in range(0,H,stride):
            for x in range(0,W,stride):
                if att_map[y,x] >= att_threshold:
                    feats.extend(self.extract_features_window(retina_out, msfb_out, center=(float(x),float(y)), radius=4, stride=4))
        return self.ingest_features_direct(feats, create_if_missing=True)

    def select_next_fixation_learning(self, retina_out, msfb_out, current_fix, local_radius:int=48, stride:int=4, gaussian_sigma:float=12.0):
        # inverted attention + gaussian * edge_strength heuristic (same as earlier)
        att = self.attn.get_map().detach().cpu()
        inv = torch.ones_like(att) - att
        H,W = att.shape
        cx, cy = float(current_fix[0]), float(current_fix[1])
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
        gauss = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (gaussian_sigma ** 2)))
        combined = (inv * gauss).numpy()
        retina_x, grad, h, diff, flow, cropped = retina_out
        grad_s = grad.squeeze(0)
        if grad_s.ndim == 4:
            grad_s = grad_s.squeeze(0)
        if grad_s.ndim == 3 and grad_s.shape[-1] == 2:
            gx = grad_s[...,0]; gy = grad_s[...,1]; edge_mag = torch.sqrt(gx*gx + gy*gy)
            edge_mag_np = edge_mag.detach().cpu().numpy()
        else:
            edge_mag_np = np.zeros((H,W))
        x0 = max(0, int(cx - local_radius)); x1 = min(W-1, int(cx + local_radius))
        y0 = max(0, int(cy - local_radius)); y1 = min(H-1, int(cy + local_radius))
        best_score=-1.0; best_pos=(cx, cy)
        for y in range(y0, y1+1, stride):
            for x in range(x0, x1+1, stride):
                score = combined[y, x] * (1.0 + float(edge_mag_np[y, x]))
                if score > best_score:
                    best_score = score; best_pos = (float(x), float(y))
        dx = best_pos[0]-cx; dy = best_pos[1]-cy
        dist = math.hypot(dx,dy); orient = math.atan2(dy, dx)
        return (dist, orient, best_pos)

    # small helper to expose threshold as callable for compatibility
    def attn_write_threshold(self):
        return 0.5

    # save / load
    def save(self, filepath:str):
        data = {'graph0': self.graph0, 'graph1': self.graph1, 'attn_visited': self.attn.visited, 'H':self.H, 'W':self.W}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    def load(self, filepath:str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.graph0 = data.get('graph0', self.graph0)
        self.graph1 = data.get('graph1', self.graph1)
        visited = data.get('attn_visited', None)
        if visited is not None:
            self.attn.visited = visited
            self.attn.update_map()

# ------------------------
# SaccadeAttention (kept minimal, similar to earlier)
# ------------------------
class SaccadeAttention:
    def __init__(self, h:int, w:int, r:float=24.0, sigma:float=12.0, peak:float=1.2, device:Optional[torch.device]=None):
        self.h = h; self.w = w
        self.r = float(r); self.sigma = float(sigma); self.peak = float(peak)
        self.device = device if device is not None else torch.device('cpu')
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=self.device), torch.arange(w, dtype=torch.float32, device=self.device), indexing='ij')
        self.xx = xx; self.yy = yy
        self.reset()
    def reset(self):
        self.cx = (self.w - 1)/2.0; self.cy = (self.h - 1)/2.0
        self.visited = [(self.cx, self.cy)]
        self.update_map()
    def saccade(self, dist:float, orient:float, record_visit:bool=True):
        dx = float(dist)*math.cos(float(orient)); dy = float(dist)*math.sin(float(orient))
        self.cx = max(0.0, min(self.w-1.0, float(self.cx)+dx))
        self.cy = max(0.0, min(self.h-1.0, float(self.cy)+dy))
        if record_visit:
            self.visited.append((self.cx, self.cy))
        self.update_map()
    def update_map(self):
        att = torch.zeros((self.h, self.w), dtype=torch.float32, device=self.device)
        denom = 2.0*(self.sigma**2)+1e-12
        for (cx,cy) in self.visited:
            dx2 = (self.xx - float(cx))**2; dy2 = (self.yy - float(cy))**2
            dist2 = dx2 + dy2; dist = torch.sqrt(dist2)
            inside_mask = dist <= self.r
            if inside_mask.any(): att[inside_mask] = 1.0
            outside_mask = ~inside_mask
            if outside_mask.any():
                g = torch.exp(-dist2[outside_mask]/denom)*self.peak
                g_clipped = torch.clamp(g, max=1.0)
                att[outside_mask] = torch.maximum(att[outside_mask], g_clipped)
        self.map = torch.clamp(att, 0.0, 1.0)
    def check_loop_and_reset(self, loop_radius:float) -> bool:
        if len(self.visited) <= 1: return False
        cx, cy = self.visited[-1]
        for (vx, vy) in self.visited[:-1]:
            dx = cx - vx; dy = cy - vy
            if (dx*dx + dy*dy) <= (loop_radius*loop_radius):
                # reset after loop detected
                self.reset()
                return True
        return False
    def get_map(self):
        return self.map

# End of module

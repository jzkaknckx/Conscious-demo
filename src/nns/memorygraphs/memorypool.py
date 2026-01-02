# memorypool.py
import math
import random
import time
import pickle
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# ---------------------------
# --- Utility / RANSAC etc.
# ---------------------------
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
    for it in range(num_iters):
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

# ---------------------------
# --- SaccadeAttention (as before, but add optional flag to not record visited)
# ---------------------------
class SaccadeAttention:
    def __init__(self, h: int, w: int, r: float = 24.0, sigma: float = 12.0, peak: float = 1.2, device: Optional[torch.device] = None):
        self.h = h; self.w = w; self.r = float(r); self.sigma = float(sigma); self.peak = float(peak)
        self.device = device if device is not None else torch.device('cpu')
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.device),
            torch.arange(w, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        self.xx = xx; self.yy = yy
        self.reset()

    def reset(self):
        self.cx = (self.w - 1) / 2.0; self.cy = (self.h - 1) / 2.0
        self.visited = [(self.cx, self.cy)]
        self.update_map()

    def saccade(self, dist: float, orient: float, record_visit: bool = True):
        dx = float(dist) * math.cos(float(orient)); dy = float(dist) * math.sin(float(orient))
        self.cx = max(0.0, min(self.w - 1.0, float(self.cx) + dx))
        self.cy = max(0.0, min(self.h - 1.0, float(self.cy) + dy))
        if record_visit:
            self.visited.append((self.cx, self.cy))
        self.update_map()

    def update_map(self):
        att = torch.zeros((self.h, self.w), dtype=torch.float32, device=self.device)
        denom = 2.0 * (self.sigma ** 2) + 1e-12
        for (cx, cy) in self.visited:
            dx2 = (self.xx - float(cx)) ** 2
            dy2 = (self.yy - float(cy)) ** 2
            dist2 = dx2 + dy2
            dist = torch.sqrt(dist2)
            inside_mask = dist <= self.r
            if inside_mask.any():
                att[inside_mask] = 1.0
            outside_mask = ~inside_mask
            if outside_mask.any():
                g = torch.exp(-dist2[outside_mask] / denom) * self.peak
                g_clipped = torch.clamp(g, max=1.0)
                att[outside_mask] = torch.maximum(att[outside_mask], g_clipped)
        self.map = torch.clamp(att, 0.0, 1.0)

    def check_loop_and_reset(self, loop_radius: float) -> bool:
        if len(self.visited) <= 1:
            return False
        cx, cy = self.visited[-1]
        for (vx, vy) in self.visited[:-1]:
            dx = cx - vx; dy = cy - vy
            if (dx * dx + dy * dy) <= (loop_radius * loop_radius):
                self.reset()
                return True
        return False

    def get_map(self) -> torch.Tensor:
        return self.map

# ---------------------------
# --- Graph0/Graph1/Graph2 (simplified versions; similar to earlier)
# ---------------------------
class Graph0:
    class Node:
        def __init__(self, node_id: int, modality: str, magnitude: float, pos: Tuple[float,float], signature: Optional[Tuple]=None):
            self.id = node_id; self.modality = modality; self.magnitude = magnitude; self.pos = pos; self.signature = signature
            self.age = 0; self.count = 1
    class Edge:
        def __init__(self, edge_id: int, src: int, dst: int, dist: float, orient: float):
            self.id = edge_id; self.src = src; self.dst = dst; self.dist = dist; self.orient = orient
            self.weight = 1.0; self.age = 0
    def __init__(self, bucket_size: int = 8):
        from collections import defaultdict
        self.nodes = {}; self.edges = {}; self.next_node_id = 0; self.next_edge_id = 0
        self.inv_index = defaultdict(lambda: defaultdict(set))  # modality->signature->set(node_ids)
        self.bucket_size = bucket_size; self.buckets = defaultdict(list)
    def _bucket_key(self, pos):
        return (int(pos[0]) // self.bucket_size, int(pos[1]) // self.bucket_size)
    def add_node(self, modality, magnitude, pos, signature=None):
        nid = self.next_node_id; n = Graph0.Node(nid, modality, magnitude, pos, signature)
        self.nodes[nid] = n; self.inv_index[modality][signature].add(nid)
        self.buckets[self._bucket_key(pos)].append(nid); self.next_node_id += 1; return nid
    def find_local_nodes(self, pos, radius):
        bx,by = self._bucket_key(pos); r_b = max(1, int(math.ceil(radius / self.bucket_size)))
        out=[]
        for dx in range(-r_b, r_b+1):
            for dy in range(-r_b, r_b+1):
                out.extend(self.buckets.get((bx+dx, by+dy), []))
        return out
    def connect(self, src, dst):
        n1=self.nodes[src]; n2=self.nodes[dst]
        dx = n2.pos[0]-n1.pos[0]; dy = n2.pos[1]-n1.pos[1]; dist = math.hypot(dx,dy); orient = math.atan2(dy,dx)
        eid = self.next_edge_id; self.edges[eid] = Graph0.Edge(eid, src, dst, dist, orient); self.next_edge_id +=1
        eid2 = self.next_edge_id; self.edges[eid2] = Graph0.Edge(eid2, dst, src, dist, (orient+math.pi)%(2*math.pi)); self.next_edge_id+=1
        return eid, eid2
    def strengthen_edges(self, node_ids: Sequence[int], delta: float = 1.0):
        s=set(node_ids)
        for e in self.edges.values():
            if e.src in s and e.dst in s:
                e.weight += delta

class Graph1:
    class Node:
        def __init__(self, nid:int, edge_ids: Tuple[int,...]): self.id=nid; self.edge_ids=tuple(sorted(edge_ids)); self.count=1; self.signature=None
    def __init__(self):
        self.nodes = {}; self.next_id=0
    def add_or_increment(self, edge_ids: Sequence[int]):
        key = tuple(sorted(edge_ids))
        if key in self.nodes:
            self.nodes[key].count += 1; return self.nodes[key].id
        nid = self.next_id; self.nodes[key] = Graph1.Node(nid, key); self.nodes[key].signature = tuple(key); self.next_id+=1; return nid

class Graph2:
    class Node:
        def __init__(self, nid:int, g1_ids: Tuple[int,...]): self.id=nid; self.g1_ids=tuple(sorted(g1_ids)); self.count=1
    def __init__(self):
        self.nodes = {}; self.next_id=0
    def add_or_increment(self, g1_ids: Sequence[int]):
        key = tuple(sorted(g1_ids))
        if key in self.nodes:
            self.nodes[key].count += 1; return self.nodes[key].id
        nid = self.next_id; self.nodes[key] = Graph2.Node(nid, key); self.next_id +=1; return nid

# ---------------------------
# --- Main class that implements retrieval + decision + learning + save/load
# ---------------------------
class FoveatedGraphMemory:
    def __init__(self, H:int, W:int,
                 bucket_size:int=8,
                 spatial_thresh:float=24.0,
                 ransac_iters:int=200,
                 ransac_inlier_thresh:float=6.0,
                 merge_radius:float=4.0):
        self.H=H; self.W=W
        self.graph0 = Graph0(bucket_size=bucket_size)
        self.graph1 = Graph1(); self.graph2 = Graph2()
        self.attn = SaccadeAttention(H, W, r=bucket_size*3, sigma=bucket_size*1.5)
        self.spatial_thresh = spatial_thresh
        self.ransac_iters = ransac_iters
        self.ransac_inlier_thresh = ransac_inlier_thresh
        self.merge_radius = merge_radius

    # --- feature extraction limited to a window around center (does not modify attention) ---
    def extract_features_window(self, retina_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], msfb_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], center: Tuple[float,float], radius:int=32, stride:int=4, thresholds:Optional[Dict]=None):
        """
        Return list of feature dicts {'modality','signature','position','value'} sampled inside window centered at center.
        retina_out = (x, grad, h, diff, flow, cropped) ; msfb_out = (curv, aspect, orient)
        Only use grad (edge), h (hue), curv/aspect/orient.
        """
        if thresholds is None:
            thresholds = {'edge':0.05, 'hue':0.0, 'curv':0.01, 'aspect':0.01, 'orient':0.0}
        x, grad, h, diff, flow, cropped = retina_out
        curv_bank, aspect_bank, orient_bank = msfb_out
        # shapes assumed: grad [1,1,H,W,2], h [1,1,H,W], curv_bank [1,S,H,W], ...
        _,_,H,W = x.shape
        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - radius); x1 = min(W-1, cx + radius)
        y0 = max(0, cy - radius); y1 = min(H-1, cy + radius)
        feats=[]
        # squeeze
        grad_s = grad.squeeze(0); grad_s = grad_s.squeeze(0) if grad_s.ndim==4 else grad_s  # [H,W,2]
        if grad_s.ndim==3 and grad_s.shape[-1]==2:
            gx = grad_s[...,0]; gy = grad_s[...,1]; edge_mag = torch.sqrt(gx*gx + gy*gy)
            edge_orient = torch.atan2(gy, gx)
        else:
            edge_mag = torch.zeros((H,W)); edge_orient = torch.zeros((H,W))
        h_s = h.squeeze(0).squeeze(0)
        curv_s = curv_bank.squeeze(0)
        aspect_s = aspect_bank.squeeze(0)
        orient_s = orient_bank.squeeze(0)
        for y in range(y0, y1+1, stride):
            for xcoord in range(x0, x1+1, stride):
                pos = (float(xcoord), float(y))
                em = float(edge_mag[y, xcoord].item())
                if em >= thresholds['edge']:
                    ob = float(edge_orient[y,xcoord].item())
                    bin_idx = int((ob + math.pi)/(2*math.pi) * 8)  # 8 bins
                    feats.append({'modality':'edge','signature':('edge',bin_idx),'position':pos,'value':em})
                hue_val = float(h_s[y,xcoord].item())
                if abs(hue_val) >= thresholds['hue']:
                    hue_bin = int(hue_val*16)
                    feats.append({'modality':'hue','signature':('hue',hue_bin),'position':pos,'value':hue_val})
                S = curv_s.shape[0]
                for s in range(S):
                    cv = float(curv_s[s,y,xcoord].item())
                    if cv >= thresholds['curv']:
                        feats.append({'modality':'curv','signature':('curv',s,int(min(255,int(cv*255)))),'position':pos,'value':cv,'scale':s})
                    av = float(aspect_s[s,y,xcoord].item())
                    if abs(av) >= thresholds['aspect']:
                        feats.append({'modality':'aspect','signature':('aspect',s,int(min(255,int((av+1.0)*127)))),'position':pos,'value':av,'scale':s})
                    ov = float(orient_s[s,y,xcoord].item())
                    if abs(ov) >= thresholds['orient']:
                        obin = int((ov+1.0)/2.0 * 8)
                        feats.append({'modality':'orient','signature':('orient',s,obin),'position':pos,'value':ov,'scale':s})
        return feats

    # --- ingest (local merge) similar to earlier ---
    def ingest_features_direct(self, features: List[Dict], create_if_missing:bool=True):
        created=[]
        for feat in features:
            mod=feat['modality']; sig=feat.get('signature'); pos=feat['position']; mag=float(feat.get('value',1.0))
            local = self.graph0.find_local_nodes(pos, self.merge_radius)
            merged=False
            for nid in local:
                node = self.graph0.nodes[nid]
                if node.modality==mod and node.signature==sig:
                    # merge (EMA)
                    alpha=0.2
                    nx = alpha*pos[0] + (1-alpha)*node.pos[0]; ny = alpha*pos[1] + (1-alpha)*node.pos[1]
                    node.pos = (nx,ny); node.magnitude = alpha*mag + (1-alpha)*node.magnitude
                    node.count += 1; node.age = 0
                    created.append(nid); merged=True; break
            if not merged and create_if_missing:
                nid = self.graph0.add_node(mod, mag, pos, signature=sig); created.append(nid)
        # connect pairwise
        for i in range(len(created)):
            for j in range(i+1, len(created)):
                self.graph0.connect(created[i], created[j])
        return created

    # --- Phase I retrieval: candidate selection via inverted index + spatial filter (on given features) ---
    def retrieve_candidates_for_features(self, features: List[Dict], spatial_filter:float):
        candidates = set()
        for feat in features:
            mod = feat['modality']; sig = feat.get('signature'); pos = feat['position']
            postings = self.graph0.inv_index.get(mod, {}).get(sig, set())
            for nid in postings:
                node = self.graph0.nodes.get(nid)
                if node is None: continue
                d = math.hypot(node.pos[0]-pos[0], node.pos[1]-pos[1])
                if d <= spatial_filter:
                    candidates.add(nid)
            # also add bucket neighbors (tolerate signature mismatch)
            local = self.graph0.find_local_nodes(pos, spatial_filter)
            for nid in local: candidates.add(nid)
        return candidates

    # --- Phase II geometric verification (same idea as earlier) ---
    def geometric_verify(self, query_feats:List[Dict], candidate_node_ids:Sequence[int], min_inliers:int=6):
        src_pts=[]; dst_pts=[]; mem_ids=[]
        for feat in query_feats:
            mod=feat['modality']; sig=feat.get('signature'); qpos=feat.get('position')
            for nid in candidate_node_ids:
                node = self.graph0.nodes[nid]
                if node.modality != mod: continue
                src_pts.append([node.pos[0], node.pos[1]]); dst_pts.append([qpos[0], qpos[1]]); mem_ids.append(nid)
        if len(src_pts) < 2: return None
        src_arr = np.array(src_pts); dst_arr = np.array(dst_pts)
        rres = ransac_similarity(src_arr, dst_arr, num_iters=self.ransac_iters, inlier_thresh=self.ransac_inlier_thresh)
        if rres is None: return None
        inliers_mask = rres['inliers_mask']; inlier_indices = np.nonzero(inliers_mask)[0]
        if len(inlier_indices) < min_inliers: return None
        matched_mem_ids = [mem_ids[i] for i in inlier_indices]
        matched_mem_pts = src_arr[inlier_indices]; matched_qry_pts = dst_arr[inlier_indices]
        edge_score, total_edges, preserved_edges = self.edge_consistency_score(matched_mem_ids, rres['scale'], rres['R'], rres['t'])
        return {'scale':rres['scale'],'R':rres['R'],'t':rres['t'],'inlier_count':int(len(inlier_indices)),
                'matched_mem_ids':matched_mem_ids,'matched_mem_pts':matched_mem_pts,'matched_qry_pts':matched_qry_pts,
                'edge_score':edge_score,'preserved_edges':preserved_edges,'total_edges':total_edges}

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

    # --- Strengthen and promote when a verified match is found ---
    def strengthen_and_promote(self, match_info, promote_edge_count_threshold:int=3):
        mem_ids = match_info['matched_mem_ids']; mem_set=set(mem_ids)
        matched_edge_ids=[]
        for eid,edge in self.graph0.edges.items():
            if edge.src in mem_set and edge.dst in mem_set:
                edge.weight += 1.0
                matched_edge_ids.append(eid)
        if len(matched_edge_ids) >= promote_edge_count_threshold:
            g1_id = self.graph1.add_or_increment(matched_edge_ids)
            self.graph2.add_or_increment([g1_id])
        # store differing local features: features near matched points but not matched nodes
        # (this is executed by caller via ingest if desired)

    # --- Selection policy when no match found: simple saliency in small local patch ---
    def choose_new_fixation_by_saliency(self, retina_out, msfb_out, current_fix, local_radius=32, stride=4, alpha=16.0):
        # compute features in small local window and score each by edge_strength * exp(-r/alpha)
        feats = self.extract_features_window(retina_out, msfb_out, center=current_fix, radius=local_radius, stride=stride)
        # compute saliency map per sampled point
        best_score=-1; best_pos=current_fix
        for f in feats:
            pos=f['position']; r = math.hypot(pos[0]-current_fix[0], pos[1]-current_fix[1])
            edge_strength = f['value'] if f['modality']=='edge' else 0.0
            # saliency uses exp(-r/alpha) * (1 + edge_strength)
            sal = math.exp(-r/alpha) * (1.0 + edge_strength)
            if sal > best_score:
                best_score = sal; best_pos = pos
        # movement vector from current_fix to best_pos
        dx = best_pos[0] - current_fix[0]; dy = best_pos[1] - current_fix[1]
        dist = math.hypot(dx,dy); orient = math.atan2(dy,dx)
        return (dist, orient), best_pos

    # --- run one step: retrieval (with expanding radius) then optional learning ---
    def run_one_step(self, retina_out, msfb_out, current_fix: Tuple[float,float],
                     initial_search_radius:int=8, max_search_radius:int=160, expand_step:int=8,
                     spatial_filter_base:float=24.0, allow_learning:bool=True, learning_record_in_attention:bool=True):
        """
        Process one cycle given upstream tensors and current fixation.
        Returns dict with keys:
          - 'matched' : bool
          - 'move': (dist,orient)
          - 'match_info' : match info or None
          - 'learned_nodes' : list of node ids added/updated
        Notes:
          - retrieval-induced eye movement does not update self.attn.visited by default (we implement that)
          - if allow_learning True, after a successful match we ingest local features at the target location and optionally record that saccade in attention (learning_record_in_attention)
        """
        # progressive search
        r = initial_search_radius
        found_match = False; match_info=None; chosen_move=None; learned_nodes=[]
        while r <= max_search_radius and not found_match:
            # extract features within window radius r around current_fix
            feats = self.extract_features_window(retina_out, msfb_out, center=current_fix, radius=r, stride=4)
            if len(feats)==0:
                r += expand_step; continue
            # candidate retrieval
            candidates = self.retrieve_candidates_for_features(feats, spatial_filter = spatial_filter_base * (1.0 + r/initial_search_radius*0.5))
            if not candidates:
                r += expand_step; continue
            # geometric verify
            match = self.geometric_verify(feats, list(candidates), min_inliers=6)
            if match is not None:
                # success
                found_match=True; match_info=match
                # compute movement vector: move toward centroid of matched query points (relative to current_fix)
                qpts = match['matched_qry_pts']
                if qpts is None or len(qpts)==0:
                    target = current_fix
                else:
                    centroid = np.mean(qpts, axis=0)
                    target = (float(centroid[0]), float(centroid[1]))
                dx = target[0] - current_fix[0]; dy = target[1] - current_fix[1]
                dist = math.hypot(dx,dy); orient = math.atan2(dy,dx)
                chosen_move = (dist, orient, target)
                # strengthen and promote memory based on match
                self.strengthen_and_promote(match)
                # store differing local features: find features in window around target which are not matched mem nodes
                if allow_learning:
                    # extract a small local window around target and ingest features
                    learn_feats = self.extract_features_window(retina_out, msfb_out, center=target, radius=24, stride=4)
                    # Optionally filter out features that already correspond to matched memory nodes:
                    # We'll ingest all; ingest handles local merging.
                    learned_nodes = self.ingest_features_direct(learn_feats, create_if_missing=True)
                    # if learning should be recorded in attention, do a real saccade record
                    if learning_record_in_attention:
                        # move attention to target and record
                        dist_to_move, orient_to_move = dist, orient
                        self.attn.saccade(dist_to_move, orient_to_move, record_visit=True)
                break
            else:
                r += expand_step
        # if not found, choose new fixation by saliency and learn there
        if not found_match:
            move, pos = self.choose_new_fixation_by_saliency(retina_out, msfb_out, current_fix, local_radius=48, stride=4)
            chosen_move = (move[0], move[1], pos)
            if allow_learning:
                learned_nodes = self.ingest_features_direct(self.extract_features_window(retina_out, msfb_out, center=pos, radius=24, stride=4), create_if_missing=True)
                # record saccade in attention if desired
                self.attn.saccade(move[0], move[1], record_visit=True)
        return {'matched': found_match, 'move': chosen_move, 'match_info': match_info, 'learned_nodes': learned_nodes}

    # --- training helper to run multiple steps on a stream of inputs ---
    def train(self, data_generator, num_steps:int=1000, allow_learning:bool=True, save_every:int=200, save_path:str='fove_mem.pkl'):
        """
        data_generator: yields tuples (retina_out, msfb_out) for each time step (e.g., consecutive frames).
        num_steps: number of steps to run
        This training function simulates an agent that at each step:
           - does retrieval around current fixation,
           - optionally learns (ingest) and records saccade into attention
        """
        current_fix = ((self.W-1)/2.0, (self.H-1)/2.0)
        for step in range(num_steps):
            try:
                retina_out, msfb_out = next(data_generator)
            except StopIteration:
                break
            res = self.run_one_step(retina_out, msfb_out, current_fix, allow_learning=allow_learning, learning_record_in_attention=True)
            # update current_fix according to returned move (but note: run_one_step already may have recorded the saccade)
            if res['move'] is not None:
                _, _, target = res['move']
                current_fix = (float(target[0]), float(target[1]))
            if (step+1) % save_every == 0:
                self.save(save_path)
                print(f"[train] step {step+1}: saved memory to {save_path}")
        # final save
        self.save(save_path)
        print("[train] finished and saved memory.")

    # --- save/load graphs to file (pickle) ---
    def save(self, filepath:str):
        data = {
            'graph0': self.graph0,
            'graph1': self.graph1,
            'graph2': self.graph2,
            'attn_visited': self.attn.visited,
            'H': self.H, 'W': self.W
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    def load(self, filepath:str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.graph0 = data.get('graph0', self.graph0)
        self.graph1 = data.get('graph1', self.graph1)
        self.graph2 = data.get('graph2', self.graph2)
        visited = data.get('attn_visited', None)
        if visited is not None:
            self.attn.visited = visited
            self.attn.update_map()

# End of module
'''
当前的问题:
    run_one_step中  每次眼跳:检测一次->学习一次
    最好是          眼跳一次->检测--(match)-->加强
                              |-(unmatch)-->学习(多次眼跳)
    即每次检测进入模式识别 match后加强并继续眼跳; unmatch后学习,允许多次眼跳,直到跳出循环
    
    眼跳的范围动态调整 
    学习时attention矩阵
    扫视范围跳过attention,在矩阵外侧选取 首先取反,叠加以注视点为中心的高斯函数
    
    attention_map的更新问题,新图/新注视点触发更新 写明触发点
    
    position参数的意义是否存在 edge中存储方向利于匹配算法 
    
    现在的学习算法是否太敏感,持续学习纹理信息而抓不住主要的边缘.

'''

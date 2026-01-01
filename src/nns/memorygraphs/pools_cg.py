
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import math


# ----------------------------
# === Sub-modality Encoders ===
# ----------------------------
# Each encoder maps its modality tensor to a fixed-size embedding vector.
# We provide small CNN/MLP encoders; these are intentionally simple and meant
# to be swapped with heavier architectures if desired.

class SpatialEncoder(nn.Module):
    """Encoder for foveated RGB patch `x` (shape: [B, C, H, W])"""
    def __init__(self, in_channels=3, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, emb_dim)
    def forward(self, x):
        # x: [B, C, H, W]
        f = self.conv(x).view(x.size(0), -1)  # [B, 64]
        return self.fc(f)

class EdgeEncoder(nn.Module):
    """Encoder for gradient/edge output `grad` expected shape [..., 2] in last dim"""
    def __init__(self, in_channels=2, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, emb_dim)
    def forward(self, grad):
        # grad: [B, 1, H, W, 2] or [B, 2, H, W]
        if grad.dim() == 5:
            grad = grad.squeeze(1).permute(0, 3, 1, 2)  # -> [B, 2, H, W]
        return self.fc(self.conv(grad).view(grad.size(0), -1))

class HueEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, emb_dim)
    def forward(self, h):
        # h: [B, 1, H, W]
        out = self.conv(h).view(h.size(0), -1)
        return self.fc(out)

class DiffEncoder(nn.Module):
    def __init__(self, in_channels=3, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, emb_dim)
    def forward(self, diff):
        # diff: [B, C, H, W]
        f = self.conv(diff).view(diff.size(0), -1)
        return self.fc(f)

class FlowEncoder(nn.Module):
    def __init__(self, in_channels=4, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, emb_dim)
    def forward(self, flow):
        # flow: [B, C, H, W, 4] or [B, 4, H, W]
        if flow.dim() == 5:
            # keep last dim as channel
            flow = flow.squeeze(1).permute(0, 4, 2, 3)  # [B, 4, H, W]
        f = self.conv(flow).view(flow.size(0), -1)
        return self.fc(f)

# ----------------------------
# === Fusion & Projection ===
# ----------------------------
class MultiModalProjector(nn.Module):
    def __init__(self, emb_dims: Dict[str,int]):
        super().__init__()
        total = sum(emb_dims.values())
        self.fc = nn.Sequential(
            nn.Linear(total, total//2),
            nn.ReLU(),
            nn.Linear(total//2, 256)
        )
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # features: dict of modality_name -> embedding [B, D]
        xs = [features[k] for k in sorted(features.keys())]
        x = torch.cat(xs, dim=1)
        return self.fc(x)  # [B, 256]

# ----------------------------
# === Contrastive Predictive Coding (CPC)-style module ===
# ----------------------------
class CPCModel(nn.Module):
    """Encoder + Autoregressive context + predictor heads
    We follow the idea in Oord et al. (2018) CPC:
    - encode each glimpse into z_t
    - build a context c_t via an RNN/GRU over z_<=t
    - predict future z_{t+k} via critic that scores similarity between predicted and true z
    """
    def __init__(self, encoder_projector: nn.Module, z_dim=256, ctx_dim=256, pred_steps=3):
        super().__init__()
        self.encoder_projector = encoder_projector
        self.gru = nn.GRU(z_dim, ctx_dim, batch_first=True)
        self.predictors = nn.ModuleList([nn.Linear(ctx_dim, z_dim) for _ in range(pred_steps)])
        self.pred_steps = pred_steps

    def encode(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # features batched across time: each value [B, T, D_mod]
        # We'll flatten time into batch for encoder_projector which expects [B, D]
        B, T = next(iter(features.values())).shape[:2]
        flat = {}
        for k, v in features.items():
            # v shape [B, T, D'] -> [B*T, D']
            flat[k] = v.view(B*T, -1)
        z = self.encoder_projector(flat)  # expects dict of 2D tensors -> returns [B*T, z_dim]
        z = z.view(B, T, -1)
        return z  # [B, T, z_dim]

    def forward(self, z_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_seq: [B, T, z_dim]
        B, T, D = z_seq.shape
        out_ctx, _ = self.gru(z_seq)  # [B, T, ctx_dim]
        losses = []
        for k in range(1, self.pred_steps+1):
            if T - k <= 0:
                break
            pred = self.predictors[k-1](out_ctx[:, :-k, :])  # predict z_{t+k}
            target = z_seq[:, k:, :]
            # InfoNCE loss between pred and target (use dot-product logits)
            # Flatten for compute: [B*(T-k), D]
            p = pred.contiguous().view(-1, D)
            t = target.contiguous().view(-1, D)
            # positive logits
            pos_logits = torch.sum(p * t, dim=1, keepdim=True)
            # negative logits: use all other samples in batch as negatives
            neg_logits = torch.matmul(p, t.t())  # [N, N]
            # build labels where diagonal are positives
            N = p.size(0)
            labels = torch.arange(N, device=p.device)
            # compute cross entropy over logits where pos is diagonal entry
            logits = neg_logits
            loss = F.cross_entropy(logits, labels)
            losses.append(loss)
        return sum(losses) / max(1, len(losses))

# ----------------------------
# === Adapter: make encoder_projector take dict(batch_of_features) -> vector ===
# ----------------------------
class DictEncoderProjector(nn.Module):
    def __init__(self, spatial_enc, edge_enc, hue_enc, diff_enc, flow_enc):
        super().__init__()
        self.spatial = spatial_enc
        self.edge = edge_enc
        self.hue = hue_enc
        self.diff = diff_enc
        self.flow = flow_enc
        # define output dims assumed by projector
        emb_dims = {
            'edge': 256,
            'hue': 128,
            'diff': 256,
            'flow': 128,
            'spatial': 256
        }
        self.projector = MultiModalProjector(emb_dims)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # features keys map to modality tensors [B, D']
        # simply dispatch to encoders
        feats = {}
        feats['spatial'] = self.spatial(features['spatial'])
        feats['edge'] = self.edge(features['edge'])
        feats['hue'] = self.hue(features['hue'])
        feats['diff'] = self.diff(features['diff'])
        feats['flow'] = self.flow(features['flow'])
        return self.projector(feats)

# ----------------------------
# === Simple Graph-based Memory Store (sparse, in-PyTorch) ===
# ----------------------------
class GraphMemory:
    """Lightweight memory graph for storing node embeddings and edges.
    Node structure: {
      'id': int,
      'emb': Tensor [D],
      'count': int,
      'meta': dict
    }
    Edges stored as adjacency dict: edges[u] = {v: weight, ...}
    """
    def __init__(self, emb_dim=256, device='cpu', max_nodes=10000, merge_threshold=0.7):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: Dict[int, Dict[int, float]] = {}
        self.emb_dim = emb_dim
        self.device = device
        self.max_nodes = max_nodes
        self.merge_threshold = merge_threshold

    def query(self, emb: torch.Tensor, topk=5) -> List[Tuple[int, float]]:
        # emb: [D]
        if len(self.nodes) == 0:
            return []
        mats = torch.stack([n['emb'] for n in self.nodes], dim=0).to(emb.device)
        # cosine similarity
        embn = emb / (emb.norm() + 1e-6)
        matsn = mats / (mats.norm(dim=1, keepdim=True) + 1e-6)
        sims = torch.matmul(matsn, embn)
        vals, idx = torch.topk(sims, min(topk, len(self.nodes)))
        return [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]

    def upsert(self, emb: torch.Tensor, meta: Dict[str, Any]=None):
        # either merge with nearest if similarity > merge_threshold, else create new node
        q = self.query(emb, topk=1)
        if q and q[0][1] > self.merge_threshold:
            nid = q[0][0]
            node = self.nodes[nid]
            # exponential moving average update
            alpha = 0.2
            node['emb'] = (1-alpha)*node['emb'] + alpha*emb.detach().cpu()
            node['count'] += 1
            if meta:
                node['meta'].update(meta)
            return nid
        else:
            if len(self.nodes) >= self.max_nodes:
                # simple LRU-like removal based on count (could be improved)
                lru = min(range(len(self.nodes)), key=lambda i: self.nodes[i]['count'])
                del self.nodes[lru]
                # reindex edges
                new_edges = {}
                for u, adj in self.edges.items():
                    if u == lru: continue
                    newu = u - 1 if u > lru else u
                    newadj = {}
                    for v, w in adj.items():
                        if v == lru: continue
                        newv = v - 1 if v > lru else v
                        newadj[newv] = w
                    new_edges[newu] = newadj
                self.edges = new_edges
            nid = len(self.nodes)
            self.nodes.append({'id': nid, 'emb': emb.detach().cpu(), 'count': 1, 'meta': meta or {}})
            self.edges[nid] = {}
            return nid

    def add_edge(self, u: int, v: int, weight=1.0, directed=True):
        if u not in self.edges: self.edges[u] = {}
        self.edges[u][v] = self.edges[u].get(v, 0.0) + weight
        if not directed:
            if v not in self.edges: self.edges[v] = {}
            self.edges[v][u] = self.edges[v].get(u, 0.0) + weight

    def decay_edges(self, factor=0.99):
        for u, adj in self.edges.items():
            for v in list(adj.keys()):
                adj[v] *= factor
                if adj[v] < 1e-3:
                    del adj[v]

# ----------------------------
# === Saccade controller (stub) ===
# ----------------------------
class SaccadeController(nn.Module):
    """A simple controller that scores candidate fixation points using
    the fused embedding and local memory context. This is a placeholder
    for training either by imitation or RL.
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # score
        )

    def forward(self, fused_emb: torch.Tensor, neighbor_emb: torch.Tensor) -> torch.Tensor:
        # fused_emb [B, D], neighbor_emb [B, K, D] -> returns scores [B, K]
        B, K, D = neighbor_emb.shape
        fused = fused_emb.unsqueeze(1).expand(-1, K, -1)
        inp = torch.cat([fused, neighbor_emb], dim=2).view(B*K, 2*D)
        s = self.mlp(inp).view(B, K)
        return s

# ----------------------------
# === Training utilities (high-level pseudocode functions) ===
# ----------------------------

def infonce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # simple dot-product InfoNCE: pred [N, D], target [N, D]
    N = pred.size(0)
    logits = torch.matmul(pred, target.t())  # [N,N]
    labels = torch.arange(N, device=pred.device)
    return F.cross_entropy(logits, labels)

# Example training step for feature learning using consecutive glimpses
# inputs:
#   retina: RetinaModel
#   cpc: CPCModel
#   optimizer: optimizer
#   batch_of_frames: frames tensor [B, T, C, H, W]
#   centers: list of center coords per frame

def train_step_feature_learning(retina: RetinaModel, cpc: CPCModel, optimizer, frames: torch.Tensor, centers: List[Tuple[int,int]]):
    # frames: [B, T, C, H, W]
    B, T = frames.shape[0], frames.shape[1]
    device = next(cpc.parameters()).device
    # collect modality features per timestep
    features = { 'spatial': [], 'edge': [], 'hue': [], 'diff': [], 'flow': [] }
    for t in range(T):
        # example uses first center for simplicity; in practice centers could vary per frame
        x = frames[:, t].to(device)
        cx, cy = centers[t]
        with torch.no_grad():
            x_proj, grad, h, diff, flowvelocity, cropped = retina(x, cx, cy)
        # ensure shapes expected by encoders
        features['spatial'].append(cropped)
        features['edge'].append(grad)
        features['hue'].append(h)
        features['diff'].append(diff)
        features['flow'].append(flowvelocity)
    # stack time dimension -> [B, T, ...]
    for k in features.keys():
        features[k] = torch.stack(features[k], dim=1)
    # flatten each per-time feature to vector form expected by DictEncoderProjector
    # e.g., features['spatial'] is [B, T, C, H, W], slice to [B*T, C, H, W] in encoder
    # cpc.encode expects features where each value is [B, T, ...]
    z_seq = cpc.encode(features)  # [B, T, z_dim]
    loss = cpc(z_seq)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# End of file

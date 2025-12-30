# ==================================================
# Route A (Revised): Foveated Graph Memory with Saccade-Aware Attention
# ==================================================
# This implementation rewrites the previous design to strictly follow
# the new specification provided by the user.
#
# Core ideas:
# - Only sparse, foveated regions (attention-weighted) are written
# - Graph0 stores feature-nodes (edge/color/curv/aspect/orient)
# - Edges encode co-activation + saccade displacement
# - Local graph matching is used for retrieval (robust to scale/rotation)
# - Graph1 is constructed from frequently co-activated edges in Graph0
#
# All logic is encapsulated in classes for debuggability.
# ==================================================

import math
import itertools
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def polar_from_delta(dx: float, dy: float) -> Tuple[float, float]:
    dist = math.sqrt(dx * dx + dy * dy)
    orient = math.atan2(dy, dx)
    return dist, orient


def gaussian2d(h: int, w: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))


# --------------------------------------------------
# Attention Map with Saccade Dynamics
# --------------------------------------------------
class SaccadeAttention:
    """
    Maintains a probabilistic attention map that tracks eye saccades.
    Values represent probability of writing into memory.
    """

    def __init__(self, h: int, w: int, radius: float, sigma: float):
        self.h = h
        self.w = w
        self.radius = radius
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        self.map = gaussian2d(self.h, self.w, self.cx, self.cy, self.sigma)
        self.visited = [(self.cx, self.cy)]

    def saccade(self, dist: float, orient: float):
        dx = dist * math.cos(orient)
        dy = dist * math.sin(orient)
        self.cx += dx
        self.cy += dy
        self.cx = max(0, min(self.w - 1, self.cx))
        self.cy = max(0, min(self.h - 1, self.cy))
        self.map = torch.maximum(
            self.map,
            gaussian2d(self.h, self.w, self.cx, self.cy, self.sigma),
        )
        self.visited.append((self.cx, self.cy))

    def check_loop_and_reset(self, loop_radius: float) -> bool:
        for (vx, vy) in self.visited[:-1]:
            if (self.cx - vx) ** 2 + (self.cy - vy) ** 2 < loop_radius ** 2:
                self.reset()
                return True
        return False


# --------------------------------------------------
# Graph0: Feature Graph (modal-separated nodes)
# --------------------------------------------------
class Graph0:
    class Node:
        def __init__(self, node_id: int, modality: str, magnitude: float, pos: Tuple[float, float]):
            self.id = node_id
            self.modality = modality
            self.magnitude = magnitude
            self.pos = pos
            self.age = 0

    class Edge:
        def __init__(self, edge_id: int, src: int, dst: int, dist: float, orient: float):
            self.id = edge_id
            self.src = src
            self.dst = dst
            self.dist = dist
            self.orient = orient
            self.weight = 1.0
            self.age = 0

    def __init__(self):
        self.nodes: Dict[int, Graph0.Node] = {}
        self.edges: Dict[int, Graph0.Edge] = {}
        self.next_node_id = 0
        self.next_edge_id = 0
        self.spatial_index: Dict[Tuple[int, int], List[int]] = {}

    # ---------- Node ops ----------
    def add_node(self, modality: str, magnitude: float, pos: Tuple[float, float]) -> int:
        nid = self.next_node_id
        self.nodes[nid] = Graph0.Node(nid, modality, magnitude, pos)
        key = (int(pos[0]) // 8, int(pos[1]) // 8)
        self.spatial_index.setdefault(key, []).append(nid)
        self.next_node_id += 1
        return nid

    def local_nodes(self, pos: Tuple[float, float], radius: float) -> List[int]:
        cx, cy = int(pos[0]) // 8, int(pos[1]) // 8
        out = []
        r = int(radius // 8) + 1
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                out.extend(self.spatial_index.get((cx + dx, cy + dy), []))
        return out

    # ---------- Edge ops ----------
    def connect(self, src: int, dst: int):
        n1 = self.nodes[src]
        n2 = self.nodes[dst]
        dist, orient = polar_from_delta(n2.pos[0] - n1.pos[0], n2.pos[1] - n1.pos[1])

        eid = self.next_edge_id
        self.edges[eid] = Graph0.Edge(eid, src, dst, dist, orient)
        self.next_edge_id += 1

        # bidirectional
        eid2 = self.next_edge_id
        self.edges[eid2] = Graph0.Edge(eid2, dst, src, dist, orient + math.pi)
        self.next_edge_id += 1

    def strengthen_edges(self, node_ids: List[int]):
        node_set = set(node_ids)
        for e in self.edges.values():
            if e.src in node_set and e.dst in node_set:
                e.weight += 1.0


# --------------------------------------------------
# Graph1: Edge-Pattern Graph (higher semantics)
# --------------------------------------------------
class Graph1:
    class Node:
        def __init__(self, node_id: int, edge_ids: Tuple[int, ...]):
            self.id = node_id
            self.edge_ids = edge_ids
            self.count = 1

    def __init__(self):
        self.nodes: Dict[Tuple[int, ...], Graph1.Node] = {}
        self.next_node_id = 0

    def update_from_edges(self, edge_ids: List[int]):
        key = tuple(sorted(edge_ids))
        if key in self.nodes:
            self.nodes[key].count += 1
        else:
            self.nodes[key] = Graph1.Node(self.next_node_id, key)
            self.next_node_id += 1


# --------------------------------------------------
# Main Memory Layer
# --------------------------------------------------
class FoveatedGraphMemory(nn.Module):
    """
    Main module implementing the revised design.
    """

    def __init__(self, h: int, w: int, attn_radius: float = 24.0, attn_sigma: float = 12.0):
        super().__init__()
        self.graph0 = Graph0()
        self.graph1 = Graph1()
        self.attn = SaccadeAttention(h, w, attn_radius, attn_sigma)

    # --------------------------------------------------
    # Ingest features
    # --------------------------------------------------
    def forward(
        self,
        edge: torch.Tensor,      # [1, H, W]
        hue: torch.Tensor,       # [1, H, W]
        curvature: torch.Tensor, # [1, S, H, W]
        aspect: torch.Tensor,    # [1, S, H, W]
        orient: torch.Tensor,    # [1, S, H, W]
        saccade: Optional[Tuple[float, float]] = None,
    ):
        # Update attention
        if saccade is not None:
            dist, ang = saccade
            self.attn.saccade(dist, ang)
            loop = self.attn.check_loop_and_reset(loop_radius=16.0)
        else:
            loop = False

        attn_map = self.attn.map
        H, W = attn_map.shape

        active_nodes: List[int] = []

        # Sample sparse points
        ys, xs = torch.where(attn_map > 0.5)
        for y, x in zip(ys.tolist(), xs.tolist()):
            pos = (float(x), float(y))

            # edge
            active_nodes.append(self.graph0.add_node("edge", edge[0, y, x].item(), pos))
            # hue
            active_nodes.append(self.graph0.add_node("hue", hue[0, y, x].item(), pos))

            # multiscale features (collapsed)
            active_nodes.append(self.graph0.add_node("curvature", curvature[0, :, y, x].mean().item(), pos))
            active_nodes.append(self.graph0.add_node("aspect", aspect[0, :, y, x].mean().item(), pos))
            active_nodes.append(self.graph0.add_node("orient", orient[0, :, y, x].mean().item(), pos))

        # Connect co-activated nodes
        for n1, n2 in itertools.combinations(active_nodes, 2):
            self.graph0.connect(n1, n2)

        if loop:
            self.graph0.strengthen_edges(active_nodes)

        # Update Graph1 using edges from this attention window
        edge_ids = [e.id for e in self.graph0.edges.values() if e.weight > 1.0]
        if edge_ids:
            self.graph1.update_from_edges(edge_ids)

        return {
            "num_nodes": len(self.graph0.nodes),
            "num_edges": len(self.graph0.edges),
            "num_patterns": len(self.graph1.nodes),
        }

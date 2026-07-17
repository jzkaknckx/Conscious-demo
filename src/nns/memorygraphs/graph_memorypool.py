import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# Unified Config (v5.1)
# =============================
class MemoryConfig:
    H = 256
    W = 256
    device = torch.device('cpu')

    # 极坐标采样配置 (Log-Polar Config)
    lp_rho_bins = 64
    lp_theta_bins = 64
    lp_base_radius = 2.0

    # 广义距离变换形变约束 (GDT Constraints)
    default_lambda_rho = 1.5
    default_gamma_theta = 2.0
    
    # 模态权重平衡 (Modality Normalization)
    modality_weights = {
        0: 0.7,
        1: 0.6,
        2: 0.3,
        3: 0.3,
        4: 0.3,
    }
    
    feature_types = {
        0: {'metric': 'Vector_Magnitude', 'paradigm': 'A', 'dominant_channel': 0},
        1: {'metric': 'Euclidean_Absolute', 'paradigm': 'A', 'dominant_metric': 'Isurf'},
        2: {'metric': 'Euclidean_Absolute', 'paradigm': 'B'},
        3: {'metric': 'Euclidean_Absolute', 'paradigm': 'C'},
        4: {'metric': 'Periodic_Property', 'paradigm': 'C', 'periods': {0: 1.0, 1: 1.0, 2: 1.0}}
    }
    
    # [修改] 通道属性划分 (STRENGTH vs CONTINUITY_SURFACE vs CONTINUITY_TRACE vs IGNORE)
    feature_attributes = {
        0: {0: 'STRENGTH', 1: 'CONTINUITY_TRACE'},   
        1: {0: 'CONTINUITY_SURFACE', 1: 'CONTINUITY_SURFACE', 2: 'CONTINUITY_SURFACE'}, 
        2: {0: 'STRENGTH', 1: 'STRENGTH', 2: 'STRENGTH'},     
        3: {0: 'CONTINUITY_TRACE', 1: 'CONTINUITY_TRACE', 2: 'CONTINUITY_TRACE'},  
        4: {0: 'CONTINUITY_TRACE', 1: 'CONTINUITY_TRACE', 2: 'CONTINUITY_TRACE'},  
    }
    sigma_surf = 0.5
    sigma_trace = 0.5
    
    # [新增] 动态写入门限
    tau_str_gate = 0.8
    tau_surf_gate = 0.99
    tau_trace_gate = 0.8

    # Neuron Dynamics GmemI
    T_excite_I = 0.8
    T_inject_I = 0.5
    tau_active_I = 5
    tau_refractory_I = 10
    decay_rate_I = 0.6
    
    # Neuron Dynamics GmemII
    T_excite_II = 2
    T_inject_II = 0.5
    tau_active_II = 15
    tau_refractory_II = 30
    decay_rate_II = 0.95
    
    # Neuron Dynamics GmemIII
    T_excite_III = 3.0
    T_inject_III = 0.5
    tau_active_III = 20
    tau_refractory_III = 50
    decay_rate_III = 0.98
    
    # Interest Map New Params
    interest_margin_radius = 24
    alpha_local = 0.5
    beta_periphery = 0.8
    gamma_remote = 0.2
    saccade_sigma = 50
    foveal_sigma = 25
    eta_foveal = 0.25
    epsilon_foveal = 0.75
    
    # Optimizer weights
    w_base = 1.0
    w_sp_i = 1.2
    w_sp_ii = 1.5
    w_exp = 1.0
    w_guide = 2
    
    # Topology Mount Params
    k_depth = 2
    
    # Other new params
    guide_padding = 5
    guide_kernel_size = 11
    eta_exp = 2.0
    
    # Semantic Mask Params
    alpha_mask = 15.0
    beta_mask = 5.0
    sigma_mask = 100.0
    epsilon_breakout = 0.05
    lambda_con1 = 2.0
    lambda_con2 = 2.0   
    lambda_str = 0.7
    mask_dilation_steps = 20000
    mask_edge_dilation = 0
    sigma_micro = 50.0
    alpha_ior = 0.5
    gamma_dist = 0.001
    gamma_macro_dist = 0.001
    sigma_consume = 10.0
    alpha_consume = 0.8

    # Hyperedge / relation graph params
    relation_merge_threshold = 0.45
    relation_confidence_decay = 0.98
    self_edge_min_span = 1.0

    # LEARN_MICRO topological intention params
    micro_min_interior_samples = 3
    micro_repeat_to_boundary = 3
    micro_min_area_for_boundary = 128.0
    theta_color_var = 0.02
    theta_res = 0.18
    theta_bd_cover = 0.65
    theta_boundary = 0.05
    micro_w_color = 1.0
    micro_w_residual = 0.8
    micro_w_blue = 0.6
    micro_w_boundary = 1.2
    micro_w_unseen_boundary = 0.9
    micro_w_edge = 1.0
    micro_w_corner = 0.5
    micro_w_junction = 0.4
    micro_w_terminator = 0.4
    micro_w_semantic_prior = 0.0
    sigma_seek = 28.0
    seek_step = 60.0
    sigma_trace_step = 18.0
    trace_step = 24.0
    sigma_trace_angle = 0.9
    sigma_boundary_visit = 8.0


# =============================
# Controller State Machine
# =============================
class NeuronState(Enum):
    REFRACTORY = -1
    CALM = 0
    ACTIVE = 1

class MainPhase(Enum):
    REVIEW = 1
    LEARN_MACRO = 2
    LEARN_MICRO = 3

class MicroIntention(Enum):
    SURFACE_CONFIRM = 1
    INTERIOR_SAMPLE = 2
    BOUNDARY_SEEK = 3
    CONTOUR_TRACE = 4
    RESUME_OR_EXIT = 5


def _angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


@dataclass
class RelationEdge:
    """A multigraph relation edge stored inside a semantic/entity hyperedge."""
    edge_id: int
    src_id: int
    dst_id: int
    rho_offset: float
    theta_offset: float
    lambda_rho: float
    gamma_theta: float
    role: str = 'peripheral'
    count: int = 1
    confidence: float = 1.0
    span: float = 0.0
    area: float = 0.0

    def distance_to(self, rho_offset: float, theta_offset: float) -> float:
        dr = rho_offset - self.rho_offset
        dt = _angle_delta(theta_offset, self.theta_offset)
        return math.sqrt((self.lambda_rho * dr) ** 2 + (self.gamma_theta * dt) ** 2)

    def update(self, rho_offset: float, theta_offset: float, span: float = 0.0, area: float = 0.0):
        new_count = self.count + 1
        alpha = 1.0 / new_count
        dtheta = _angle_delta(theta_offset, self.theta_offset)
        self.rho_offset = (1.0 - alpha) * self.rho_offset + alpha * rho_offset
        self.theta_offset = self.theta_offset + alpha * dtheta
        self.span = (1.0 - alpha) * self.span + alpha * span
        self.area = (1.0 - alpha) * self.area + alpha * area
        self.count = new_count
        self.confidence = min(1.0, self.confidence + alpha * (1.0 - self.confidence))

    def as_legacy_link(self) -> Dict[str, float]:
        return {
            'rho_offset': self.rho_offset,
            'theta_offset': self.theta_offset,
            'lambda_rho': self.lambda_rho,
            'gamma_theta': self.gamma_theta,
        }


# =============================
# Gmemory: Modality Nodes & Semantic Nodes
# =============================
class ModalityNode:
    """模态孤立特征层节点 (GI)"""
    def __init__(self, node_id: int, modality_id: int, prototype: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.node_id = node_id
        self.modality_id = modality_id
        self.prototype = prototype
        self.mask = mask if mask is not None else torch.ones_like(prototype)
        
        self.count = 1
        self.last_seen = 0
        self.stability_score = 1.0
        
        # 激活状态
        self.activation_level = 0.0
        self.state_flag = NeuronState.CALM
        self.timer = 0

    def tick_update(self, E_input: float, cfg: MemoryConfig):
        if self.state_flag == NeuronState.REFRACTORY:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.CALM
                self.activation_level = 0.0
            else:
                self.activation_level = -1.0 # 强制设为 -1，释放厌恶足迹
        elif self.state_flag == NeuronState.ACTIVE and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.REFRACTORY
                self.timer = cfg.tau_refractory_I
                self.activation_level = -1.0
        else: # CALM
            self.activation_level = self.activation_level * cfg.decay_rate_I + E_input
            if self.activation_level > cfg.T_excite_I:
                self.state_flag = NeuronState.ACTIVE
                self.activation_level = 1.0
                self.timer = cfg.tau_active_I


class SemanticNode:
    """锚点-外围拓扑层节点 (GII，语义对象)"""
    def __init__(self, node_id: int, anchor_id: int):
        self.node_id = node_id
        self.anchor_id = anchor_id
        # Legacy first-edge view kept for older callers. Full topology is in relation_edges.
        self.peripheral_links: Dict[int, Dict[str, float]] = {}
        self.child_node_ids: Set[int] = {anchor_id}
        self.relation_edges: Dict[int, RelationEdge] = {}
        self.relation_edges_by_pair: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.relation_edge_ids: List[int] = []
        self.self_edge_ids: List[int] = []
        self.boundary_edge_ids: List[int] = []
        
        self.activation_level = 0.0
        self.state_flag = NeuronState.CALM
        self.timer = 0
        
        # 内生标度
        self.explore_time = 0.0
        self.max_span = 0.0
        self.is_completed = False
        self.color_mean: Optional[torch.Tensor] = None
        self.color_m2: Optional[torch.Tensor] = None
        self.color_samples = 0
        self.boundary_coverage = 0.0
        self.color_residual = 1.0
        self.is_closed_contour = False
        self.has_hole_candidate = False

    def related_node_ids(self) -> Set[int]:
        ids = set(self.child_node_ids)
        ids.add(self.anchor_id)
        for edge in self.relation_edges.values():
            ids.add(edge.src_id)
            ids.add(edge.dst_id)
        return ids

    def iter_relation_edges(self, role: Optional[str] = None):
        for edge in self.relation_edges.values():
            if role is None or edge.role == role:
                yield edge

    def tick_update(self, E_input: float, cfg: MemoryConfig):
        if self.state_flag == NeuronState.REFRACTORY:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.CALM
                self.activation_level = 0.0
            else:
                self.activation_level = -1.0
        elif self.state_flag == NeuronState.ACTIVE and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.REFRACTORY
                self.timer = cfg.tau_refractory_II
                self.activation_level = -1.0
        else:
            self.activation_level = self.activation_level * cfg.decay_rate_II + E_input
            if self.activation_level > cfg.T_excite_II:
                self.state_flag = NeuronState.ACTIVE
                self.activation_level = 1.0
                self.timer = cfg.tau_active_II


class EntityNode:
    """宏观空间实体层节点 (GIII，格式塔实体)"""
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.components: Dict[int, Dict[str, float]] = {} # sid -> { 'dx': float, 'dy': float }
        self.component_edges: Dict[int, Dict[str, float]] = {}
        self.component_edges_by_semantic: Dict[int, List[int]] = defaultdict(list)
        self.next_component_edge_id = 0
        
        self.activation_level = 0.0
        self.state_flag = NeuronState.CALM
        self.timer = 0

    def tick_update(self, E_input: float, cfg: MemoryConfig):
        if self.state_flag == NeuronState.REFRACTORY:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.CALM
                self.activation_level = 0.0
            else:
                self.activation_level = -1.0
        elif self.state_flag == NeuronState.ACTIVE and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.REFRACTORY
                self.timer = cfg.tau_refractory_III
                self.activation_level = -1.0
        else:
            self.activation_level = self.activation_level * cfg.decay_rate_III + E_input
            if self.activation_level > cfg.T_excite_III:
                self.state_flag = NeuronState.ACTIVE
                self.activation_level = 1.0
                self.timer = cfg.tau_active_III


class GmemoryI:
    """Gmemory I 层：模态孤立特征层"""
    def __init__(self):
        self.nodes: Dict[int, ModalityNode] = {}
        self.next_node_id = 0
        
    def add_node(self, modality_id: int, prototype: torch.Tensor, mask: Optional[torch.Tensor] = None) -> ModalityNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = ModalityNode(nid, modality_id, prototype, mask)
        self.nodes[nid] = node
        return node


class GmemoryII:
    """Gmemory II 层：锚点-外围拓扑层"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.semantic_nodes: Dict[int, SemanticNode] = {}
        self.next_node_id = 0
        self.next_relation_edge_id = 0

    def add_semantic_node(self, anchor_id: int) -> SemanticNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = SemanticNode(nid, anchor_id)
        self.semantic_nodes[nid] = node
        return node

    def add_relation_edge(self, semantic_node: SemanticNode, src_id: int, dst_id: int,
                          rho_offset: float, theta_offset: float,
                          lambda_rho: Optional[float] = None, gamma_theta: Optional[float] = None,
                          role: str = 'peripheral', span: float = 0.0, area: float = 0.0) -> RelationEdge:
        """Add or merge a relation edge in the semantic multigraph."""
        l_rho = lambda_rho if lambda_rho is not None else self.cfg.default_lambda_rho
        g_theta = gamma_theta if gamma_theta is not None else self.cfg.default_gamma_theta
        pair = (src_id, dst_id)
        for edge_id in semantic_node.relation_edges_by_pair.get(pair, []):
            edge = semantic_node.relation_edges[edge_id]
            if edge.role == role and edge.distance_to(rho_offset, theta_offset) < self.cfg.relation_merge_threshold:
                edge.update(rho_offset, theta_offset, span=span, area=area)
                return edge

        edge_id = self.next_relation_edge_id
        self.next_relation_edge_id += 1
        edge = RelationEdge(
            edge_id=edge_id,
            src_id=src_id,
            dst_id=dst_id,
            rho_offset=rho_offset,
            theta_offset=theta_offset,
            lambda_rho=l_rho,
            gamma_theta=g_theta,
            role=role,
            span=span,
            area=area,
        )
        semantic_node.relation_edges[edge_id] = edge
        semantic_node.relation_edges_by_pair[pair].append(edge_id)
        semantic_node.relation_edge_ids.append(edge_id)
        semantic_node.child_node_ids.update([src_id, dst_id])
        if role == 'self_extent':
            semantic_node.self_edge_ids.append(edge_id)
        if role in {'boundary', 'contour', 'terminator', 'junction'}:
            semantic_node.boundary_edge_ids.append(edge_id)
        if dst_id not in semantic_node.peripheral_links and src_id != dst_id:
            semantic_node.peripheral_links[dst_id] = edge.as_legacy_link()
        return edge

    def add_peripheral(self, semantic_node: SemanticNode, peri_id: int,
                       rho_offset: float, theta_offset: float,
                       lambda_rho: Optional[float] = None, gamma_theta: Optional[float] = None,
                       role: str = 'peripheral', span: float = 0.0, area: float = 0.0):
        """添加外围特征拓扑。保留旧接口，并委托给多重关系边。"""
        return self.add_relation_edge(
            semantic_node,
            semantic_node.anchor_id,
            peri_id,
            rho_offset,
            theta_offset,
            lambda_rho=lambda_rho,
            gamma_theta=gamma_theta,
            role=role,
            span=span,
            area=area,
        )


class GmemoryIII:
    """Gmemory III 层：宏观实体拓扑层"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.entity_nodes: Dict[int, EntityNode] = {}
        self.next_node_id = 0
        
    def add_entity_node(self) -> EntityNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = EntityNode(nid)
        self.entity_nodes[nid] = node
        return node
        
    def add_component(self, entity_node: EntityNode, sem_id: int, dx: float, dy: float):
        edge_id = entity_node.next_component_edge_id
        entity_node.next_component_edge_id += 1
        component = {'sem_id': sem_id, 'dx': dx, 'dy': dy, 'count': 1}
        entity_node.component_edges[edge_id] = component
        entity_node.component_edges_by_semantic[sem_id].append(edge_id)
        # Legacy representative component for older consumers.
        entity_node.components.setdefault(sem_id, {'dx': dx, 'dy': dy})
        return component


# =============================
# Similarity Engine
# =============================
class SimilarityEngine:
    @staticmethod
    def sim_euclidean(X: torch.Tensor, W: torch.Tensor, mask_X: torch.Tensor, mask_W: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
        """欧氏绝对坐标: 高斯RBF核"""
        M = mask_X * mask_W
        if X.dim() == 4:
            M = M.view(1, -1, 1, 1)
        
        diff_sq = torch.sum(((X - W) ** 2) * M, dim=1, keepdim=True)
        valid_channels = torch.sum(M, dim=1, keepdim=True).clamp(min=1e-6)
        dist_sq = diff_sq / valid_channels
        return torch.exp(-dist_sq / (2 * sigma**2))

    @staticmethod
    def sim_vector(X: torch.Tensor, W: torch.Tensor, mask_X: torch.Tensor, mask_W: torch.Tensor) -> torch.Tensor:
        """矢量占有度: 点积"""
        M = mask_X * mask_W
        if X.dim() == 4:
            M = M.view(1, -1, 1, 1)
        
        return torch.sum(X * W * M, dim=1, keepdim=True)

    @staticmethod
    def sim_periodic(X: torch.Tensor, W: torch.Tensor, mask_X: torch.Tensor, mask_W: torch.Tensor, periods: Dict[int, float], gamma: float = 2.0) -> torch.Tensor:
        """周期属性: 高阶余弦"""
        M = mask_X * mask_W
        if X.dim() == 4:
            M = M.view(1, -1, 1, 1)
        
        sum_sim = 0
        count = 0
        for c, k in periods.items():
            if c >= M.shape[1]:
                continue
            Mc = M[:, c:c+1]
            Xc = X[:, c:c+1]
            Wc = W[:, c:c+1]
            
            S_raw = torch.cos(k * (Xc - Wc))
            S_prop = torch.pow(torch.clamp(S_raw, min=0.0), gamma)
            sum_sim = sum_sim + S_prop * Mc
            count = count + Mc
            
        count = count.clamp(min=1e-6)
        return sum_sim / count

    @staticmethod
    def calculate_similarity(X: torch.Tensor, W: torch.Tensor, cfg: 'MemoryConfig', mod_id: int, mask_X: Optional[torch.Tensor] = None, mask_W: Optional[torch.Tensor] = None) -> torch.Tensor:
        finfo = getattr(cfg, 'feature_types', {}).get(mod_id, {})
        metric = finfo.get('metric', 'Vector_Magnitude')
        
        if mask_X is None:
            mask_X = torch.ones(1, X.shape[1], device=X.device)
        if mask_W is None:
            mask_W = torch.ones(1, W.shape[1], device=W.device)
            
        # Ensure dimensions match
        if mask_X.shape[1] > X.shape[1]:
            mask_X = mask_X[:, :X.shape[1]]
        if mask_W.shape[1] > W.shape[1]:
            mask_W = mask_W[:, :W.shape[1]]
            
        if metric == 'Euclidean_Absolute':
            sigma = finfo.get('sigma', 0.05)
            return SimilarityEngine.sim_euclidean(X, W, mask_X, mask_W, sigma)
        elif metric == 'Vector_Magnitude':
            return SimilarityEngine.sim_vector(X, W, mask_X, mask_W)
        elif metric == 'Periodic_Property':
            periods = finfo.get('periods', {0: 1.0})
            gamma = finfo.get('gamma', 2.0)
            return SimilarityEngine.sim_periodic(X, W, mask_X, mask_W, periods, gamma)
        else:
            return SimilarityEngine.sim_vector(X, W, mask_X, mask_W)


# =============================
# Gposition: Dynamic Projection Retrieval Network
# =============================
class Gposition:
    """动态投影检索网络 (L1 & L2)"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.device = cfg.device

    def l1_modality_routing_projection(self, X_subspaces: Dict[int, torch.Tensor], 
                                       target_nodes: List[ModalityNode]) -> Dict[int, torch.Tensor]:
        """
        L1层：模态路由子空间投影 (Modality Routing Subspace Projection)
        仅检索不处于REFRACTORY且激活电位强于阈值的活跃节点！
        """
        S_maps = {}
        active_nodes = [n for n in target_nodes if n.activation_level > self.cfg.T_inject_I and n.state_flag != NeuronState.REFRACTORY]
        for node in active_nodes:
            mod_id = node.modality_id
            X_m = X_subspaces.get(mod_id)
            if X_m is None:
                continue
                
            w_m = node.prototype.to(self.device).view(1, -1, 1, 1)
            mask_W = node.mask.to(self.device).view(1, -1)
            B, C_m, H, W = X_m.shape
            
            S_m = SimilarityEngine.calculate_similarity(X_m, w_m, self.cfg, mod_id, mask_W=mask_W)
            
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            S_maps[node.node_id] = S_m * weight
            
        return S_maps

    def l1_footprint_projection(self, X_subspaces: Dict[int, torch.Tensor], 
                                target_nodes: List[ModalityNode]) -> Dict[int, torch.Tensor]:
        """
        生成足迹专用投影：不论是否处于不应期，只要脱离了平静态 (A!=0)，均提供整图响应投射以制造区域脚印。
        (过滤逻辑由 Controller 传递 target_nodes 时负责)
        """
        S_maps = {}
        for node in target_nodes:
            mod_id = node.modality_id
            X_m = X_subspaces.get(mod_id)
            if X_m is None:
                continue
                
            w_m = node.prototype.to(self.device).view(1, -1, 1, 1)
            mask_W = node.mask.to(self.device).view(1, -1)
            S_m = SimilarityEngine.calculate_similarity(X_m, w_m, self.cfg, mod_id, mask_W=mask_W)
            
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            S_maps[node.node_id] = S_m * weight
            
        return S_maps

    def _create_log_polar_grid(self, H: int, W: int, xc: int, yc: int) -> torch.Tensor:
        rho_bins = self.cfg.lp_rho_bins
        theta_bins = self.cfg.lp_theta_bins
        rmax = math.sqrt(max(xc, W - xc)**2 + max(yc, H - yc)**2)
        
        rho_grid = torch.linspace(math.log(self.cfg.lp_base_radius), math.log(max(rmax, self.cfg.lp_base_radius + 1e-5)), rho_bins, device=self.device)
        theta_grid = torch.linspace(-math.pi, math.pi, theta_bins, device=self.device)
        
        theta_mesh, rho_mesh = torch.meshgrid(theta_grid, rho_grid, indexing='ij')
        r_mesh = torch.exp(rho_mesh)
        
        grid_x = xc + r_mesh * torch.cos(theta_mesh)
        grid_y = yc + r_mesh * torch.sin(theta_mesh)
        
        grid_x = (grid_x / (W - 1)) * 2 - 1
        grid_y = (grid_y / (H - 1)) * 2 - 1
        
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        return grid

    def _generalized_distance_transform(self, S_LP: torch.Tensor, rho_offset: float, theta_offset: float, 
                                        lambda_rho: float, gamma_theta: float) -> torch.Tensor:
        _, C, theta_bins, rho_bins = S_LP.shape
        padding = (2, 2, 2, 2)
        S_padded = F.pad(S_LP, padding, mode='replicate')
        kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25
        D_map = F.conv2d(S_padded, kernel)
        return D_map

    def l2_structure_synthesis(self, S_maps: Dict[int, torch.Tensor], semantic_node: SemanticNode):
        S_anc = S_maps.get(semantic_node.anchor_id)
        if S_anc is None:
            return None
        
        B, C, H, W = S_anc.shape
        pooled = F.max_pool2d(S_anc, kernel_size=7, stride=1, padding=3)
        peaks = (S_anc == pooled) & (S_anc > 0.1)
        
        peak_indices = torch.nonzero(peaks[0, 0])
        if peak_indices.numel() == 0:
            return None
            
        best_idx = torch.argmax(S_anc[0, 0, peak_indices[:, 0], peak_indices[:, 1]])
        yc, xc = int(peak_indices[best_idx, 0]), int(peak_indices[best_idx, 1])

        grid = self._create_log_polar_grid(H, W, xc, yc)
        D_maps_sum = torch.zeros((1, 1, self.cfg.lp_theta_bins, self.cfg.lp_rho_bins), device=self.device)
        
        for edge in semantic_node.iter_relation_edges():
            if edge.src_id == edge.dst_id:
                continue
            S_peri = S_maps.get(edge.dst_id)
            if S_peri is not None:
                S_LP_peri = F.grid_sample(S_peri, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                D_n = self._generalized_distance_transform(
                    S_LP_peri, edge.rho_offset, edge.theta_offset,
                    edge.lambda_rho, edge.gamma_theta
                )
                D_maps_sum += D_n

        best_resonance = torch.max(D_maps_sum)
        max_idx = torch.argmax(D_maps_sum)
        
        theta_idx = max_idx // self.cfg.lp_rho_bins
        rho_idx = max_idx % self.cfg.lp_rho_bins
        
        delta_rho = float(rho_idx) / self.cfg.lp_rho_bins * 2.0 - 1.0  
        delta_theta = float(theta_idx) / self.cfg.lp_theta_bins * math.pi
        
        s_star = math.exp(delta_rho)
        theta_star = delta_theta
        
        return {
            'x': xc, 'y': yc,
            's_star': s_star,
            'theta_star': theta_star,
            'score': float(S_anc[0, 0, yc, xc].item() + best_resonance.item())
        }


# =============================
# Interest Optimizer (场动力引擎)
# =============================
class InterestOptimizer:
    """InterestOptimizer: 负责多场矩阵叠加出实时的视觉兴趣地貌 I_map"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        self.base_interest: Optional[torch.Tensor] = None
        self.I_exp: Dict[int, torch.Tensor] = {} # Topological expectations
        
        # 供调试观测的开放变量
        self.sum_I_spatial_I: Optional[torch.Tensor] = None
        self.sum_I_spatial_II: Optional[torch.Tensor] = None
        self.sum_I_exp: Optional[torch.Tensor] = None
        self.I_guide: Optional[torch.Tensor] = None

        self.H_orig: Optional[int] = None
        self.W_orig: Optional[int] = None
        self.r: Optional[int] = None

        self.I_str: Optional[torch.Tensor] = None
        self.I_con1: Optional[torch.Tensor] = None
        self.I_con2: Optional[torch.Tensor] = None
        
    def initialize_base_interest(self, X_subspaces: Dict[int, torch.Tensor], H_orig: int = None, W_orig: int = None):
        self.H_orig = H_orig
        self.W_orig = W_orig
        self.r = self.cfg.interest_margin_radius
        B, C, H, W = list(X_subspaces.values())[0].shape
        self.base_interest = torch.zeros((B, 1, H, W), device=self.device)
        
        I_str = torch.zeros((B, 1, H, W), device=self.device)
        I_con1 = torch.zeros((B, 1, H, W), device=self.device)
        I_con2 = torch.zeros((B, 1, H, W), device=self.device)


        for mod_id, X_m in X_subspaces.items():
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            attrs = self.cfg.feature_attributes.get(mod_id, {})
            
            C_m = X_m.shape[1]
            for c in range(C_m):
                attr = attrs.get(c, 'STRENGTH')
                if attr == 'IGNORE':
                    continue
                    
                channel_X = X_m[:, c:c+1, :, :]
                
                if attr == 'STRENGTH':
                    I_str += weight * torch.abs(channel_X)
                
                elif attr == 'CONTINUITY_SURFACE':
                    # Sobel-like simple spatial difference for gradient
                    dx = channel_X[:, :, :, 1:] - channel_X[:, :, :, :-1]
                    dy = channel_X[:, :, 1:, :] - channel_X[:, :, :-1, :]
                    dx = F.pad(dx, (0, 1, 0, 0))
                    dy = F.pad(dy, (0, 0, 0, 1))
                    grad_norm_sq = dx**2 + dy**2
                    I_con1 += weight * torch.exp(-grad_norm_sq / (2 * self.cfg.sigma_surf**2))

                elif attr == 'CONTINUITY_TRACE':
                    kernel_size = 3
                    padding = 1
                    v_x = torch.cos(channel_X)
                    v_y = torch.sin(channel_X)
                    blur_vx = F.avg_pool2d(v_x, kernel_size, stride=1, padding=padding)
                    blur_vy = F.avg_pool2d(v_y, kernel_size, stride=1, padding=padding)
                    coherence = torch.sqrt(blur_vx**2 + blur_vy**2 + 1e-6)
                    I_con2 += weight * coherence

        self.I_str = I_str
        self.I_con1 = I_con1
        self.I_con2 = I_con2
        self.base_interest = I_str + I_con1 + I_con2

        if self.H_orig is not None and self.W_orig is not None:
            # 保留兴趣压制：初始化时压低外围兴趣
            h1, h2 = max(0, H//2-(self.H_orig-self.r)//2), min(H, H//2+(self.H_orig-self.r)//2)
            w1, w2 = max(0, W//2-(self.W_orig-self.r)//2), min(W, W//2+(self.W_orig-self.r)//2)
            self.base_interest[:, :, :w1, :] = 0; self.base_interest[:, :, w2:, :] = 0
            self.base_interest[:, :, :, :h1] = 0; self.base_interest[:, :, :, h2:] = 0

        b_max = self.base_interest.max() + 1e-5
        self.base_interest /= b_max

    def generate_I_guide(self, M_resp: torch.Tensor, pt: Tuple[int, int]) -> torch.Tensor:
        H, W = M_resp.shape[-2], M_resp.shape[-1]
        cx, cy = pt
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        sigma_local = self.cfg.saccade_sigma
        G_pt = torch.exp(-dist_sq / (2 * sigma_local**2)).view(1, 1, H, W)
        
        M_local = M_resp * G_pt
        
        padding = self.cfg.guide_padding
        kernel_size = self.cfg.guide_kernel_size
        blurred = F.avg_pool2d(M_resp, kernel_size, stride=1, padding=padding)
        M_periphery = F.relu(blurred - M_resp)
        M_remote = F.relu(M_resp - M_local)
        
        alpha, beta = self.cfg.alpha_local, self.cfg.beta_periphery
        return alpha * M_periphery + beta * M_remote

    def calculate_I_map(self, active_gmem_i: List['ModalityNode'], active_gmem_ii: List['SemanticNode'], 
                        S_maps_I: Dict[int, torch.Tensor], pt: Tuple[int, int]) -> torch.Tensor:
        if self.base_interest is None:
            return torch.zeros((1, 1, 1, 1), device=self.device)
            
        I_map = self.cfg.w_base * self.base_interest.clone()
        H, W = I_map.shape[-2], I_map.shape[-1]
        
        self.sum_I_spatial_I = torch.zeros_like(I_map)
        self.sum_I_spatial_II = torch.zeros_like(I_map)
        self.sum_I_exp = torch.zeros_like(I_map)
        self.I_guide = torch.zeros_like(I_map)
        
        cx, cy = pt
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        sigma_local_I = self.cfg.saccade_sigma
        sigma_foveal_I = getattr(self.cfg, 'foveal_sigma', 30.0)
        G_local_I = torch.exp(-dist_sq / (2 * sigma_local_I**2)).view(1, 1, H, W)
        G_foveal_I = torch.exp(-dist_sq / (2 * sigma_foveal_I**2)).view(1, 1, H, W)
        
        sigma_local_II = self.cfg.saccade_sigma * 2.0
        sigma_foveal_II = getattr(self.cfg, 'foveal_sigma', 30.0) * 2.0
        G_local_II = torch.exp(-dist_sq / (2 * sigma_local_II**2)).view(1, 1, H, W)
        G_foveal_II = torch.exp(-dist_sq / (2 * sigma_foveal_II**2)).view(1, 1, H, W)
        
        # + w_sp_I * sum I_spatial_I
        for node in active_gmem_i:
            if node.state_flag != NeuronState.CALM and node.node_id in S_maps_I:
                A_k = node.activation_level
                M_resp = S_maps_I[node.node_id]
                Phi_I = A_k * G_local_I - abs(A_k) * G_foveal_I
                I_spatial = M_resp * Phi_I
                self.sum_I_spatial_I += I_spatial
                I_map += self.cfg.w_sp_i * I_spatial
                
        # + w_sp_II * sum I_spatial_II
        for node in active_gmem_ii:
            if node.state_flag != NeuronState.CALM:
                A_k = node.activation_level
                M_resp_sum = torch.zeros((1, 1, H, W), device=self.device)
                count = 0
                if node.anchor_id in S_maps_I:
                    M_resp_sum += S_maps_I[node.anchor_id]
                    count += 1
                for pid in node.related_node_ids():
                    if pid == node.anchor_id:
                        continue
                    if pid in S_maps_I:
                        M_resp_sum += S_maps_I[pid]
                        count += 1
                if count > 0:
                    M_resp_sum /= count
                    Phi_II = A_k * G_local_II - abs(A_k) * G_foveal_II
                    I_spatial = M_resp_sum * Phi_II
                    self.sum_I_spatial_II += I_spatial
                    I_map += self.cfg.w_sp_ii * I_spatial
                
        # + w_exp * sum I_exp
        for exp_map in self.I_exp.values():
            self.sum_I_exp += exp_map
            I_map += self.cfg.w_exp * exp_map
                
        # + w_guide * I_guide
        self.I_guide = self.generate_I_guide(self.base_interest, pt)
        I_map += self.cfg.w_guide * self.I_guide
        
        if self.H_orig is not None and self.W_orig is not None:
            # 保留兴趣压制：初始化时压低外围兴趣
            h1, h2 = max(0, H//2-(self.H_orig-self.r)//2), min(H, H//2+(self.H_orig-self.r)//2)
            w1, w2 = max(0, W//2-(self.W_orig-self.r)//2), min(W, W//2+(self.W_orig-self.r)//2)
            I_map[:, :, :w1, :] = 0; I_map[:, :, w2:, :] = 0
            I_map[:, :, :, :h1] = 0; I_map[:, :, :, h2:] = 0

        return F.relu(I_map)

    def add_expectation(self, node_id: int, pt: Tuple[int, int], H: int, W: int):
        if node_id not in self.I_exp:
            self.I_exp[node_id] = torch.zeros((1, 1, H, W), device=self.device)
            
        cx, cy = pt
        cx, cy = max(0, min(cx, W-1)), max(0, min(cy, H-1))
        
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        sigma_exp = self.cfg.saccade_sigma
        
        eta_exp = self.cfg.eta_exp
        
        sigma_surround = sigma_exp
        sigma_crater = sigma_exp / 2.0
        
        surround = torch.exp(-dist_sq / (2 * sigma_surround**2))
        crater = torch.exp(-dist_sq / (2 * sigma_crater**2))
        
        # 大感受野正，小极坑负 (带有 -1 不应权)
        dog = surround - 2.0 * crater
        self.I_exp[node_id] += eta_exp * dog.view(1, 1, H, W)

    def clear_node_fields(self, node_id: int, is_semantic: bool = False):
        if not is_semantic:
            self.I_exp.pop(node_id, None)


# =============================
# Controller
# =============================
class Controller:
    """控制器：基于动态兴趣矩阵与双塔交互的检视引擎"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        self.state = MainPhase.REVIEW
        self.optimizer = InterestOptimizer(cfg)
        
        self.fixation_point = (cfg.W // 2, cfg.H // 2)
        self.prev_fixation_point = self.fixation_point
        
        self.active_semantic_id: Optional[int] = None 
        self.active_entity_id: Optional[int] = None
        self.anchor_position: Optional[Tuple[int, int]] = None
        self.micro_explore_time: float = 0.0
        self.micro_max_span: float = 0.0
        self.current_step = 0

        self.current_interest_map = None
        
        # Semantic Mask / Macro-Micro Saccade
        self.semantic_mask: Optional[torch.Tensor] = None
        self.semantic_history: Optional[torch.Tensor] = None
        self.sigma_micro: float = getattr(self.cfg, 'sigma_micro', 50.0)
        self.micro_intention = MicroIntention.SURFACE_CONFIRM
        self.micro_repeat_count = 0
        self.interior_sample_count = 0
        self.color_mean: Optional[torch.Tensor] = None
        self.color_m2: Optional[torch.Tensor] = None
        self.color_residual = 1.0
        self.boundary_coverage = 0.0
        self.boundary_visited: Optional[torch.Tensor] = None
        self.last_boundary_tangent: Optional[Tuple[float, float]] = None
        self.boundary_start_point: Optional[Tuple[int, int]] = None

        # Run-step specific maps and buffers
        self.ior_map: Optional[torch.Tensor] = None
        self.m_aversion: Optional[torch.Tensor] = None
        self.peripheral_buffer: List[Dict] = []
        self.suspend_stack: List[Dict] = []
        self.E_input_gmem_i_acc = defaultdict(float)
        self.E_input_gmem_ii_acc = defaultdict(float)
        self.E_input_gmem_iii_acc = defaultdict(float)

        # trail
        self.matrix1 = None
        self.matrix2 = None
        self.matrix3 = None
        self.matrix4 = None

    def _extract_and_match_local_features(self, X_subspaces: Dict[int, torch.Tensor], 
                                          gmem_i: GmemoryI, x: int, y: int, 
                                          similarity_threshold: float = 0.85) -> List[ModalityNode]:
        active_nodes = []
        for mod_id, X_m in X_subspaces.items():
            C_m = X_m.shape[1]
            H, W = X_m.shape[-2], X_m.shape[-1]
            mod_attrs = self.cfg.feature_attributes.get(mod_id, {})
            finfo = self.cfg.feature_types.get(mod_id, {})
            paradigm = finfo.get('paradigm', 'C')
            
            valid_mask = torch.zeros((1, C_m), device=self.device)
            sc_values = []
            
            # Evaluate Write Gating Threshold
            for c in range(C_m):
                attr = mod_attrs.get(c, 'STRENGTH')
                if attr == 'IGNORE':
                    continue
                vc = X_m[0, c, y, x].item()
                if attr == 'STRENGTH':
                    if abs(vc) > self.cfg.tau_str_gate:
                        valid_mask[0, c] = 1.0
                elif attr == 'CONTINUITY_SURFACE':
                    dx = X_m[0, c, y, min(x+1, W-1)].item() - vc
                    dy = X_m[0, c, min(y+1, H-1), x].item() - vc
                    sc = math.exp(-(dx**2 + dy**2) / (2 * self.cfg.sigma_surf**2))
                    sc_values.append(sc)
                    if sc > self.cfg.tau_surf_gate:
                        valid_mask[0, c] = 1.0
                elif attr == 'CONTINUITY_TRACE':
                    padding = 1
                    y1, y2 = max(0, y-padding), min(H, y+padding+1)
                    x1, x2 = max(0, x-padding), min(W, x+padding+1)
                    local_window = X_m[0, c, y1:y2, x1:x2]
                    v_x = torch.cos(local_window)
                    v_y = torch.sin(local_window)
                    coherence = torch.sqrt(torch.sum(v_x)**2 + torch.sum(v_y)**2) / (local_window.numel() + 1e-6)
                    if coherence.item() > self.cfg.tau_trace_gate:
                        valid_mask[0, c] = 1.0
                        
            if paradigm == 'A':
                if finfo.get('dominant_metric') == 'Isurf':
                    if len(sc_values) > 0:
                        isurf = sum(sc_values) / len(sc_values)
                        if isurf > self.cfg.tau_surf_gate:
                            valid_mask = torch.ones((1, C_m), device=self.device)
                        else:
                            continue
                    else:
                        continue
                else:
                    dom_c = finfo.get('dominant_channel', 0)
                    if dom_c < C_m and valid_mask[0, dom_c].item() == 0:
                        continue
                    else:
                        valid_mask = torch.ones((1, C_m), device=self.device)
                    
            if torch.sum(valid_mask) == 0:
                continue

            local_vec = X_m[0, :, y, x].unsqueeze(0)
            
            matched_node = None
            best_sim = -1.0
            
            for node in gmem_i.nodes.values():
                if node.modality_id == mod_id:
                    w_node = node.prototype.unsqueeze(0)
                    sim = SimilarityEngine.calculate_similarity(local_vec, w_node, self.cfg, mod_id, mask_X=valid_mask, mask_W=node.mask.unsqueeze(0)).item()
                    if sim > best_sim:
                        best_sim = sim
                        matched_node = node
                        
            if matched_node and best_sim >= similarity_threshold:
                matched_node.count += 1
                matched_node.last_seen = time.time()
                active_nodes.append(matched_node)
            else:
                new_node = gmem_i.add_node(mod_id, local_vec.squeeze(0).detach().clone(), mask=valid_mask.squeeze(0).detach().clone())
                active_nodes.append(new_node)
                
        return active_nodes

    def _modality_topology(self, modality_id: int) -> str:
        attrs = self.cfg.feature_attributes.get(modality_id, {})
        values = set(attrs.values())
        if 'CONTINUITY_SURFACE' in values:
            return 'Surface_2D'
        if 'STRENGTH' in values:
            return 'Boundary_Edge'
        if 'CONTINUITY_TRACE' in values:
            return 'Trace_1D'
        return 'Unknown'

    def _node_topology(self, node: ModalityNode) -> str:
        return self._modality_topology(node.modality_id)

    def _normalize_map(self, value: Optional[torch.Tensor], ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        if value is None:
            if ref is None:
                return torch.zeros((1, 1, 1, 1), device=self.device)
            return torch.zeros_like(ref)
        out = torch.nan_to_num(value.clone(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        vmax = out.max()
        if vmax > 1e-6:
            out = out / vmax
        return out

    def _first_map_shape(self, X_subspaces: Dict[int, torch.Tensor]) -> Tuple[int, int, int, int]:
        sample = list(X_subspaces.values())[0]
        return sample.shape

    def _ensure_runtime_maps(self, H: int, W: int):
        if self.ior_map is None:
            self.ior_map = torch.zeros((1, 1, H, W), device=self.device)
        if self.m_aversion is None:
            self.m_aversion = torch.zeros((1, 1, H, W), device=self.device)
        if self.semantic_history is None:
            self.semantic_history = torch.zeros((1, 1, H, W), device=self.device)
        if self.boundary_visited is None:
            self.boundary_visited = torch.zeros((1, 1, H, W), device=self.device)

    def _xy_grids(self, H: int, W: int):
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        return x_grid, y_grid

    def _distance_sq_from(self, pt: Tuple[int, int], H: int, W: int) -> torch.Tensor:
        cx, cy = pt
        x_grid, y_grid = self._xy_grids(H, W)
        return (x_grid - cx) ** 2 + (y_grid - cy) ** 2

    def _argmax_point(self, drive: torch.Tensor) -> Tuple[int, int]:
        H, W = drive.shape[-2], drive.shape[-1]
        if drive.max() < 1e-6:
            idx = torch.randint(0, H * W, (1,), device=self.device).item()
        else:
            idx = torch.argmax(drive).item()
        return int(idx % W), int(idx // W)

    def _mark_map_visited(self, target: torch.Tensor, pt: Tuple[int, int], sigma: float, weight: float = 1.0) -> torch.Tensor:
        H, W = target.shape[-2], target.shape[-1]
        dist_sq = self._distance_sq_from(pt, H, W)
        mark = weight * torch.exp(-dist_sq / (2 * sigma ** 2)).view(1, 1, H, W)
        return torch.max(target, mark.clamp(max=1.0))

    def _reset_micro_context(self, H: int, W: int, intention: MicroIntention = MicroIntention.SURFACE_CONFIRM):
        self.micro_intention = intention
        self.micro_repeat_count = 0
        self.interior_sample_count = 0
        self.color_mean = None
        self.color_m2 = None
        self.color_residual = 1.0
        self.boundary_coverage = 0.0
        self.boundary_visited = torch.zeros((1, 1, H, W), device=self.device)
        self.last_boundary_tangent = None
        self.boundary_start_point = None
        self.micro_explore_time = 0.0
        self.micro_max_span = 0.0
        self.m_aversion = torch.zeros((1, 1, H, W), device=self.device)

    def _color_variance(self) -> float:
        if self.color_m2 is None or self.interior_sample_count < 2:
            return float('inf')
        return float(torch.mean(self.color_m2 / max(self.interior_sample_count - 1, 1)).item())

    def _bfs_recognize(self, local_nodes, gmem_ii: GmemoryII, k_depth: int = 2):
        from collections import deque
        queue = deque([(n.node_id, 0) for n in local_nodes])
        visited_modality = set([n.node_id for n in local_nodes])
        
        modality_to_semantics = defaultdict(list)
        for sid, sem in gmem_ii.semantic_nodes.items():
            modality_to_semantics[sem.anchor_id].append(sid)
            for pid in sem.related_node_ids():
                if pid == sem.anchor_id:
                    continue
                modality_to_semantics[pid].append(sid)
                
        recognized_semantics = set()
        while queue:
            mid, depth = queue.popleft()
            
            sids = modality_to_semantics[mid]
            for sid in sids:
                recognized_semantics.add(sid)
                if depth < k_depth:
                    sem = gmem_ii.semantic_nodes[sid]
                    neighbors = list(sem.related_node_ids())
                    for neighbor_mid in neighbors:
                        if neighbor_mid not in visited_modality:
                            visited_modality.add(neighbor_mid)
                            queue.append((neighbor_mid, depth + 1))
                            
        if recognized_semantics:
            return list(recognized_semantics)[0]
        return None

    def _generate_semantic_mask(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition, 
                                anchor_nodes: List[ModalityNode], p_macro: Tuple[int, int]) -> torch.Tensor:
        S_maps = gpos.l1_footprint_projection(X_subspaces, anchor_nodes)
        
        B, C, H, W = list(X_subspaces.values())[0].shape
        M_resp_con = torch.zeros((1, 1, H, W), device=self.device)
        count = 0
        
        for node in anchor_nodes:
            if node.node_id in S_maps:
                M_resp_con += S_maps[node.node_id]
                count += 1
                
        if count > 0:
            M_resp_con /= count

        # 1. Permeability Map (Conductivity)
        M_GposI = M_resp_con
        if M_GposI.max() > 1e-6:
            M_GposI = M_GposI / M_GposI.max()
        M_GposI[M_GposI < 0.1] = 0.0

        # 2. Seed Initialization
        cx, cy = p_macro
        V = torch.zeros((B, 1, H, W), device=self.device)
        V[0, 0, cy, cx] = 1.0

        # 3. Iterative Matrix Dilation
        pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for _ in range(self.cfg.mask_dilation_steps):
            V_next = pool(V) * M_GposI
            A_V = torch.sum(V).item()
            A_Vn = torch.sum(V_next).item()
            if (A_Vn - A_V) / (A_V + 1e-6) < 1e-3:
                break
            V = V_next
            
        edge_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for _ in range(getattr(self.cfg, 'mask_edge_dilation', 5)):
            V = edge_pool(V)

        # 4. Final dynamic energy output
        E_dynamic = V
        E_dynamic[E_dynamic < 0.05] = 0
        return E_dynamic

    def _current_semantic_node(self, gmem_ii: GmemoryII) -> Optional[SemanticNode]:
        if self.active_semantic_id is None:
            return None
        return gmem_ii.semantic_nodes.get(self.active_semantic_id)

    def _gradient_magnitude(self, value: torch.Tensor) -> torch.Tensor:
        dx = value[:, :, :, 1:] - value[:, :, :, :-1]
        dy = value[:, :, 1:, :] - value[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

    def _combine_node_maps_by_topology(self, S_maps: Dict[int, torch.Tensor], gmem_i: GmemoryI,
                                       topologies: Set[str], ref: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(ref)
        count = 0
        for nid, resp in S_maps.items():
            node = gmem_i.nodes.get(nid)
            if node is None:
                continue
            if self._node_topology(node) in topologies:
                out += self._normalize_map(resp, ref)
                count += 1
        if count > 0:
            out /= count
        return out

    def _build_micro_gpos_fields(self, S_maps: Dict[int, torch.Tensor], gmem_i: GmemoryI,
                                 gmem_ii: GmemoryII) -> Dict[str, torch.Tensor]:
        sem_node = self._current_semantic_node(gmem_ii)
        ref = self.semantic_mask
        if ref is None:
            if S_maps:
                ref = next(iter(S_maps.values()))
            elif self.optimizer.base_interest is not None:
                ref = self.optimizer.base_interest
            else:
                return {}

        mask = self._normalize_map(self.semantic_mask, ref)
        if sem_node and sem_node.anchor_id in S_maps:
            surface_resp = self._normalize_map(S_maps[sem_node.anchor_id], ref)
        else:
            surface_resp = mask

        color_resp = surface_resp
        if sem_node:
            color_maps = []
            for nid in sem_node.related_node_ids():
                node = gmem_i.nodes.get(nid)
                if node is not None and self._node_topology(node) == 'Surface_2D' and nid in S_maps:
                    color_maps.append(self._normalize_map(S_maps[nid], ref))
            if color_maps:
                color_resp = torch.stack(color_maps, dim=0).max(dim=0).values

        edge_resp = self._combine_node_maps_by_topology(S_maps, gmem_i, {'Boundary_Edge', 'Trace_1D'}, ref)
        if edge_resp.max() <= 1e-6 and self.optimizer.I_str is not None:
            edge_resp = self._normalize_map(self.optimizer.I_str, ref)
        corner_resp = self._normalize_map(self._gradient_magnitude(edge_resp), ref)
        boundary_from_mask = self._normalize_map(self._gradient_magnitude(mask), ref)
        boundary_i = boundary_from_mask * self._normalize_map(
            self.cfg.micro_w_edge * edge_resp + self.cfg.micro_w_corner * corner_resp, ref
        )
        boundary_gpos = self._normalize_map(boundary_i, ref)
        color_residual = ((1.0 - color_resp).clamp(0.0, 1.0)) * mask
        residual_den = torch.sum(mask).item() + 1e-6
        self.color_residual = float(torch.sum(color_residual).item() / residual_den)

        if self.boundary_visited is None:
            self.boundary_visited = torch.zeros_like(ref)
        coverage_den = torch.sum(boundary_gpos).item() + 1e-6
        self.boundary_coverage = float(torch.sum(self.boundary_visited * boundary_gpos).item() / coverage_den)

        return {
            'mask': mask,
            'surface': surface_resp,
            'color': color_resp,
            'color_residual': color_residual,
            'edge': edge_resp,
            'corner': corner_resp,
            'boundary': boundary_gpos,
            'semantic_prior': torch.zeros_like(ref),
        }

    def _update_color_stats_from_nodes(self, local_nodes: List[ModalityNode]):
        surface_nodes = [n for n in local_nodes if self._node_topology(n) == 'Surface_2D']
        if not surface_nodes:
            return
        sample = surface_nodes[0].prototype.detach().float().view(-1)
        if self.color_mean is None:
            self.color_mean = sample.clone()
            self.color_m2 = torch.zeros_like(sample)
            self.interior_sample_count = 1
            return
        self.interior_sample_count += 1
        delta = sample - self.color_mean
        self.color_mean = self.color_mean + delta / self.interior_sample_count
        delta2 = sample - self.color_mean
        self.color_m2 = self.color_m2 + delta * delta2

    def _mark_micro_fixation(self):
        if self.semantic_mask is None:
            return
        self.m_aversion = self._mark_map_visited(
            self.m_aversion,
            self.fixation_point,
            getattr(self.cfg, 'sigma_consume', 10.0),
            getattr(self.cfg, 'alpha_consume', 0.8),
        )

    def _micro_surface_confirm_target(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = fields['mask']
        drive = mask * fields['surface'] * (1.0 - self.m_aversion)
        area = torch.sum(mask).item()
        if self.micro_repeat_count >= 1 or area >= self.cfg.micro_min_area_for_boundary:
            self.micro_intention = MicroIntention.INTERIOR_SAMPLE
        return drive

    def _micro_interior_sample_target(self, fields: Dict[str, torch.Tensor], local_nodes: List[ModalityNode]) -> torch.Tensor:
        self._update_color_stats_from_nodes(local_nodes)
        mask = fields['mask']
        blue = (1.0 - self.m_aversion).clamp(0.0, 1.0)
        drive = mask * (
            self.cfg.micro_w_color * fields['color']
            + self.cfg.micro_w_residual * fields['color_residual']
            + self.cfg.micro_w_blue * blue
        )
        color_var = self._color_variance()
        if (
            self.interior_sample_count >= self.cfg.micro_min_interior_samples
            and (color_var < self.cfg.theta_color_var or self.color_residual < self.cfg.theta_res)
        ) or self.micro_repeat_count >= self.cfg.micro_repeat_to_boundary:
            self.micro_intention = MicroIntention.BOUNDARY_SEEK
        return drive * blue

    def _micro_seek_gate(self, ref: torch.Tensor) -> torch.Tensor:
        H, W = ref.shape[-2], ref.shape[-1]
        dist_sq = self._distance_sq_from(self.fixation_point, H, W)
        dist = torch.sqrt(dist_sq + 1e-6).view(1, 1, H, W)
        return torch.exp(-((dist - self.cfg.seek_step) ** 2) / (2 * self.cfg.sigma_seek ** 2))

    def _micro_span_gate(self, ref: torch.Tensor) -> torch.Tensor:
        if self.anchor_position is None:
            return torch.ones_like(ref)
        H, W = ref.shape[-2], ref.shape[-1]
        dist = torch.sqrt(self._distance_sq_from(self.anchor_position, H, W) + 1e-6).view(1, 1, H, W)
        max_dist = dist.max().clamp(min=1e-6)
        return (dist / max_dist).clamp(0.0, 1.0)

    def _micro_boundary_seek_target(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        boundary = fields['boundary']
        unseen = (1.0 - self.boundary_visited).clamp(0.0, 1.0)
        drive = (
            self.cfg.micro_w_boundary * boundary
            + self.cfg.micro_w_unseen_boundary * boundary * unseen
        )
        drive = drive * self._micro_span_gate(boundary) * unseen * self._micro_seek_gate(boundary)
        if drive.max() > self.cfg.theta_boundary:
            self.micro_intention = MicroIntention.CONTOUR_TRACE
            if self.boundary_start_point is None:
                self.boundary_start_point = self.fixation_point
        return drive

    def _micro_tangent_gate(self, ref: torch.Tensor) -> torch.Tensor:
        if self.last_boundary_tangent is None:
            return torch.ones_like(ref)
        tx, ty = self.last_boundary_tangent
        H, W = ref.shape[-2], ref.shape[-1]
        cx, cy = self.fixation_point
        x_grid, y_grid = self._xy_grids(H, W)
        vx = (x_grid - cx).float()
        vy = (y_grid - cy).float()
        norm = torch.sqrt(vx ** 2 + vy ** 2 + 1e-6)
        cosang = ((vx * tx + vy * ty) / norm).clamp(-1.0, 1.0)
        angle = torch.acos(cosang)
        dist_gate = torch.exp(-((norm - self.cfg.trace_step) ** 2) / (2 * self.cfg.sigma_trace_step ** 2))
        angle_gate = torch.exp(-(angle ** 2) / (2 * self.cfg.sigma_trace_angle ** 2))
        return (dist_gate * angle_gate).view(1, 1, H, W)

    def _micro_contour_trace_target(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        boundary = fields['boundary']
        self.boundary_visited = self._mark_map_visited(
            self.boundary_visited,
            self.fixation_point,
            self.cfg.sigma_boundary_visit,
            1.0,
        )
        contour = (
            self.cfg.micro_w_edge * fields['edge']
            + self.cfg.micro_w_corner * fields['corner']
        )
        drive = boundary * contour * self._micro_tangent_gate(boundary) * (1.0 - self.boundary_visited)
        if self.boundary_coverage >= self.cfg.theta_bd_cover and self.color_residual < self.cfg.theta_res:
            self.micro_intention = MicroIntention.RESUME_OR_EXIT
        elif drive.max() < self.cfg.theta_boundary:
            self.micro_intention = MicroIntention.BOUNDARY_SEEK
        return drive

    def _micro_resume_or_exit_target(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.boundary_coverage >= self.cfg.theta_bd_cover and self.color_residual < self.cfg.theta_res:
            return torch.zeros_like(fields['mask'])
        if self.color_residual >= self.cfg.theta_res:
            self.micro_intention = MicroIntention.INTERIOR_SAMPLE
            return self._micro_interior_sample_target(fields, [])
        self.micro_intention = MicroIntention.BOUNDARY_SEEK
        return self._micro_boundary_seek_target(fields)

    def _decide_next_saccade_micro_by_intention(self, S_maps: Dict[int, torch.Tensor],
                                                local_nodes: List[ModalityNode],
                                                gmem_i: GmemoryI,
                                                gmem_ii: GmemoryII) -> Tuple[Tuple[int, int], torch.Tensor]:
        if self.semantic_mask is None:
            ref = self.optimizer.base_interest
            if ref is None:
                return self.fixation_point, torch.zeros((1, 1, 1, 1), device=self.device)
            return self.fixation_point, ref

        self._mark_micro_fixation()
        fields = self._build_micro_gpos_fields(S_maps, gmem_i, gmem_ii)
        if not fields:
            return self.fixation_point, self.semantic_mask

        if self.micro_intention == MicroIntention.SURFACE_CONFIRM:
            drive = self._micro_surface_confirm_target(fields)
        elif self.micro_intention == MicroIntention.INTERIOR_SAMPLE:
            drive = self._micro_interior_sample_target(fields, local_nodes)
        elif self.micro_intention == MicroIntention.BOUNDARY_SEEK:
            drive = self._micro_boundary_seek_target(fields)
        elif self.micro_intention == MicroIntention.CONTOUR_TRACE:
            drive = self._micro_contour_trace_target(fields)
        else:
            drive = self._micro_resume_or_exit_target(fields)

        next_pt = self._argmax_point(drive)
        self.matrix1 = drive
        print(drive.max())
        if self.micro_intention == MicroIntention.CONTOUR_TRACE:
            dx = float(next_pt[0] - self.fixation_point[0])
            dy = float(next_pt[1] - self.fixation_point[1])
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 1e-6:
                self.last_boundary_tangent = (dx / norm, dy / norm)
        return next_pt, drive

    def _decide_next_saccade_review(self, I_map: torch.Tensor) -> Tuple[int, int]:
        H, W = I_map.shape[-2], I_map.shape[-1]
        cx, cy = self.fixation_point
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        
        # update IoR map
        ior_sigma = self.cfg.saccade_sigma
        self.ior_map = self.ior_map * 0.9  # Decay
        self.ior_map += torch.exp(-dist_sq / (2 * ior_sigma**2)).view(1, 1, H, W)
        
        alpha_ior = getattr(self.cfg, 'alpha_ior', 0.5)
        gamma_dist = getattr(self.cfg, 'gamma_dist', 0.001)
        
        drive = I_map - alpha_ior * self.ior_map - gamma_dist * torch.sqrt(dist_sq).view(1, 1, H, W)
        if drive.max() < -1e3:
            idx = torch.randint(0, H*W, (1,)).item()
        else:
            idx = torch.argmax(drive).item()
        return (int(idx % W), int(idx // W))

    def _decide_next_saccade_macro(self, gmem_ii: GmemoryII, S_maps: Dict[int, torch.Tensor]) -> Tuple[Tuple[int, int], torch.Tensor]:
        I_con1 = self.optimizer.I_con1
        H, W = I_con1.shape[-2], I_con1.shape[-1]
        
        sum_I_spatial_II = torch.zeros((1, 1, H, W), device=self.device)
        for node in gmem_ii.semantic_nodes.values():
            M_resp_sum = torch.zeros((1, 1, H, W), device=self.device)
            if node.anchor_id in S_maps:
                M_resp_sum += S_maps[node.anchor_id]
            for pid in node.related_node_ids():
                if pid == node.anchor_id:
                    continue
                if pid in S_maps:
                    M_resp_sum += S_maps[pid]
            sum_I_spatial_II += M_resp_sum * node.activation_level
            # sum_I_spatial_II += M_resp_sum
        self.matrix4 = sum_I_spatial_II
        sum_I_spatial_II = sum_I_spatial_II.clamp(min=0.0)
        sum_log = torch.log1p(sum_I_spatial_II)
        if sum_log.max() > 1e-6:
            mask_II = sum_log / sum_log.max()
        else:
            mask_II = torch.zeros_like(sum_log)
            
        semantic_history_norm = torch.zeros_like(mask_II)
        if self.semantic_history is not None:
            if self.semantic_history.max() > 1e-6:
                semantic_history_norm = self.semantic_history / self.semantic_history.max()
                
        c_suppress = getattr(self.cfg, 'c_suppress', 1.0)
        M_suppress = torch.max(mask_II, c_suppress * semantic_history_norm)
        
        I_macro = I_con1 * (1.0 - M_suppress)
        
        if self.optimizer.H_orig is not None and self.optimizer.W_orig is not None:
            # 保留兴趣压制：压低外围兴趣
            h1, h2 = max(0, H//2-(self.optimizer.H_orig-self.optimizer.r)//2), min(H, H//2+(self.optimizer.H_orig-self.optimizer.r)//2)
            w1, w2 = max(0, W//2-(self.optimizer.W_orig-self.optimizer.r)//2), min(W, W//2+(self.optimizer.W_orig-self.optimizer.r)//2)
            I_macro[:, :, :w1, :] = 0; I_macro[:, :, w2:, :] = 0
            I_macro[:, :, :, :h1] = 0; I_macro[:, :, :, h2:] = 0
        
        kernel_size = 15
        padding = kernel_size // 2
        K_lowpass = F.avg_pool2d(I_macro, kernel_size, stride=1, padding=padding)
        
        # Distance penalty for jump
        cx, cy = self.fixation_point
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        gamma_dist = getattr(self.cfg, 'gamma_macro_dist', 0.001)

        drive = I_macro * K_lowpass - gamma_dist * torch.sqrt(dist_sq).view(1, 1, H, W)
        idx = torch.argmax(drive).item()
        return (int(idx % W), int(idx // W)), drive

    def _decide_next_saccade_micro(self, base_interest: torch.Tensor, semantic_mask: torch.Tensor) -> Tuple[int, int]:
        if semantic_mask is None:
            return self.fixation_point
            
        H, W = semantic_mask.shape[-2], semantic_mask.shape[-1]
        cx, cy = self.fixation_point
        
        sigma_consume = getattr(self.cfg, 'sigma_consume', 10.0)
        alpha = getattr(self.cfg, 'alpha_consume', 0.8)
        
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2

        # Deplete the dynamic energy matrix around the current fixation point
        depletion = alpha * torch.exp(-dist_sq / (2 * sigma_consume**2)).view(1, 1, H, W)
        self.semantic_mask = torch.max(torch.zeros_like(self.semantic_mask), self.semantic_mask - depletion)
        # We also keep m_aversion up-to-date just for visuals or other modules overlapping
        self.m_aversion = torch.max(self.m_aversion, torch.exp(-dist_sq / (2 * sigma_consume**2)).view(1, 1, H, W))
        
        # Next saccade point is the max of remaining dynamic energy
        drive = self.semantic_mask
        if drive.max() < 1e-6:
            idx = torch.randint(0, H*W, (1,)).item()
        else:
            idx = torch.argmax(drive).item()
        return (int(idx % W), int(idx // W))

    def _start_micro_learning(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                              gmem_ii: GmemoryII, local_nodes: List[ModalityNode],
                              start_pt: Tuple[int, int], source_label: str):
        if not local_nodes:
            return
        B, C, H, W = self._first_map_shape(X_subspaces)
        new_sem_node = gmem_ii.add_semantic_node(local_nodes[0].node_id)
        self.active_semantic_id = new_sem_node.node_id
        self.anchor_position = start_pt
        self.semantic_mask = self._generate_semantic_mask(X_subspaces, gpos, local_nodes, start_pt)
        self._reset_micro_context(H, W, MicroIntention.SURFACE_CONFIRM)
        self.peripheral_buffer = []
        self.state = MainPhase.LEARN_MICRO

    def _relation_role_for_node(self, node: ModalityNode) -> str:
        topology = self._node_topology(node)
        if topology == 'Boundary_Edge':
            return 'boundary'
        if topology == 'Trace_1D':
            return 'contour'
        return 'peripheral'

    def _append_relation_observation(self, nid: int, rho: float, theta: float, role: str, span: float = 0.0, area: float = 0.0):
        self.peripheral_buffer.append({
            'nid': nid,
            'rho': rho,
            'theta': theta,
            'role': role,
            'span': span,
            'area': area,
        })

    def _record_micro_observation(self, local_nodes: List[ModalityNode], sem_node: SemanticNode):
        if self.anchor_position is None:
            return
        ax, ay = self.anchor_position
        cx, cy = self.fixation_point
        r = math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) + 1e-5
        rho = math.log(r)
        theta = math.atan2(cy - ay, cx - ax)
        step_dx = cx - self.prev_fixation_point[0]
        step_dy = cy - self.prev_fixation_point[1]
        step_span = math.sqrt(step_dx ** 2 + step_dy ** 2)
        saw_anchor = False

        for node in local_nodes:
            if node.node_id == sem_node.anchor_id:
                saw_anchor = True
                if step_span >= self.cfg.self_edge_min_span and self._node_topology(node) in {'Surface_2D', 'Trace_1D'}:
                    step_theta = math.atan2(step_dy, step_dx)
                    self._append_relation_observation(
                        node.node_id,
                        math.log(step_span + 1e-5),
                        step_theta,
                        'self_extent',
                        span=step_span,
                        area=max(step_span, 1.0),
                    )
                continue
            self._append_relation_observation(
                node.node_id,
                rho,
                theta,
                self._relation_role_for_node(node),
                span=r,
                area=1.0,
            )

        self.micro_repeat_count = self.micro_repeat_count + 1 if saw_anchor else 0

    def _active_semantic_nodes(self, gmem_ii: GmemoryII) -> List[SemanticNode]:
        return [
            n for n in gmem_ii.semantic_nodes.values()
            if n.state_flag != NeuronState.CALM or self.E_input_gmem_ii_acc[n.node_id] > 0
        ]

    def _check_micro_exit(self, semantic_mask: torch.Tensor, local_nodes: List[ModalityNode] = None, gmem_ii: 'GmemoryII' = None, gmem_i: 'GmemoryI' = None) -> str:
        if semantic_mask is None:
            return 'empty'
        sum_energy = torch.sum(semantic_mask).item()
        # Exit if energy drops below a small threshold
        if sum_energy < 1.0:
            return 'depleted'
        if (
            self.micro_intention == MicroIntention.RESUME_OR_EXIT
            and self.boundary_coverage >= self.cfg.theta_bd_cover
            and self.color_residual < self.cfg.theta_res
        ):
            return 'completed'
            
        if local_nodes and gmem_ii and gmem_i and self.active_semantic_id is not None:
            sem_node = gmem_ii.semantic_nodes.get(self.active_semantic_id)
            if sem_node:
                anchor_mod_id = gmem_i.nodes[sem_node.anchor_id].modality_id
                if local_nodes[0].modality_id != anchor_mod_id and self.micro_intention not in {
                    MicroIntention.BOUNDARY_SEEK,
                    MicroIntention.CONTOUR_TRACE,
                    MicroIntention.RESUME_OR_EXIT,
                }:
                    return 'mutation'
        return 'continue'

    def _transition_to_macro(self, gmem_i: GmemoryI, gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII', exit_type: str = 'depleted'):
        if getattr(self, 'active_entity_id', None) is None:
            new_ent = gmem_iii.add_entity_node()
            self.active_entity_id = new_ent.node_id
            
        if self.active_semantic_id is not None:
            sem_node = gmem_ii.semantic_nodes.get(self.active_semantic_id)
            if sem_node:
                # 1. Batch relation processing. Multiple spatial relations between the same
                # endpoints are merged only when their relation metric is close enough.
                for p in self.peripheral_buffer:
                    if p['nid'] in gmem_i.nodes:
                        if p.get('role') == 'self_extent':
                            gmem_ii.add_relation_edge(
                                sem_node,
                                p['nid'],
                                p['nid'],
                                p['rho'],
                                p['theta'],
                                role='self_extent',
                                span=p.get('span', 0.0),
                                area=p.get('area', 0.0),
                            )
                        else:
                            gmem_ii.add_peripheral(
                                sem_node,
                                p['nid'],
                                p['rho'],
                                p['theta'],
                                role=p.get('role', 'peripheral'),
                                span=p.get('span', 0.0),
                                area=p.get('area', 0.0),
                            )
                        
                # 记录 GmemIII 组件关系 (dx, dy 相对于某种空间锚点或单纯记录位量向量)
                if self.anchor_position is not None:
                    ent_node = gmem_iii.entity_nodes[self.active_entity_id]
                    ax, ay = self.anchor_position
                    gmem_iii.add_component(ent_node, self.active_semantic_id, ax, ay)
                    
                # 更新自身内在模态属性
                sem_node.explore_time = getattr(self, 'micro_explore_time', 0.0)
                sem_node.max_span = getattr(self, 'micro_max_span', 0.0)
                sem_node.color_mean = self.color_mean.detach().clone() if self.color_mean is not None else sem_node.color_mean
                sem_node.color_m2 = self.color_m2.detach().clone() if self.color_m2 is not None else sem_node.color_m2
                sem_node.color_samples = self.interior_sample_count
                sem_node.boundary_coverage = self.boundary_coverage
                sem_node.color_residual = self.color_residual
                
                # 2. Suspend check & status update
                if exit_type == 'mutation':
                    sem_node.is_completed = False
                    self.suspend_stack.append({
                        'sid': self.active_semantic_id,
                        'anchor': self.anchor_position,
                        'mask': self.semantic_mask,
                        'explore_time': sem_node.explore_time,
                        'max_span': sem_node.max_span
                    })
                else:
                    sem_node.is_completed = True

        if self.semantic_mask is not None:
            # semantic_mask_norm = self.semantic_mask / self.semantic_mask.max()
            semantic_mask_bin = (self.semantic_mask > 0.0).float()
            self.semantic_history += semantic_mask_bin

        self.peripheral_buffer.clear()
        
        # 3. Resume & Complete
        if exit_type == 'depleted' and len(self.suspend_stack) > 0:
            resume_ctx = self.suspend_stack.pop()
            resume_id = resume_ctx['sid']
            resume_node = gmem_ii.semantic_nodes.get(resume_id)
            if resume_node and not resume_node.is_completed:
                self.active_semantic_id = resume_id
                self.anchor_position = resume_ctx['anchor']
                self.semantic_mask = resume_ctx['mask']
                self.micro_explore_time = resume_ctx['explore_time']
                self.micro_max_span = resume_ctx['max_span']
                
                # Assume we resume and the state is immediately LEARN_MICRO
                self.state = MainPhase.LEARN_MICRO
                if self.semantic_mask is not None:
                    H, W = self.semantic_mask.shape[-2], self.semantic_mask.shape[-1]
                    self._reset_micro_context(H, W, MicroIntention.BOUNDARY_SEEK)
                return
                
        self.active_semantic_id = None
        self.anchor_position = None
        self.semantic_mask = None
        self.state = MainPhase.LEARN_MACRO
        self.micro_explore_time = 0.0
        self.micro_max_span = 0.0
        B, C, H, W = self.m_aversion.shape
        self.m_aversion = torch.zeros((B, C, H, W), device=self.device)
        self.micro_intention = MicroIntention.SURFACE_CONFIRM
        self.boundary_visited = torch.zeros((B, C, H, W), device=self.device)

    def _process_review_observation(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                                    local_nodes: List[ModalityNode], gmem_i: GmemoryI,
                                    gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII',
                                    S_maps_footprint: Dict[int, torch.Tensor], H: int, W: int):
        recognized_sem_id = self._bfs_recognize(local_nodes, gmem_ii, self.cfg.k_depth)
        if recognized_sem_id is not None:
            sem_node = gmem_ii.semantic_nodes[recognized_sem_id]
            if sem_node.state_flag != NeuronState.ACTIVE:
                sem_node.activation_level += 2.0
                sem_node.state_flag = NeuronState.ACTIVE
                sem_node.timer = self.cfg.tau_active_II
            if self.anchor_position is None:
                self.anchor_position = self.fixation_point
            ax, ay = self.anchor_position
            for edge in sem_node.iter_relation_edges():
                if edge.src_id == edge.dst_id:
                    continue
                pred_r = math.exp(edge.rho_offset)
                pred_theta = edge.theta_offset
                pred_x = int(ax + pred_r * math.cos(pred_theta))
                pred_y = int(ay + pred_r * math.sin(pred_theta))
                pred_x, pred_y = max(0, min(pred_x, W - 1)), max(0, min(pred_y, H - 1))
                self.optimizer.add_expectation(edge.dst_id, (pred_x, pred_y), H, W)
            for n in local_nodes:
                self.optimizer.I_exp.pop(n.node_id, None)
            return

        if local_nodes:
            self._start_micro_learning(X_subspaces, gpos, gmem_ii, local_nodes, self.fixation_point, 'review')
        else:
            self._transition_to_macro(gmem_i, gmem_ii, gmem_iii)

    def _process_micro_observation(self, local_nodes: List[ModalityNode], gmem_ii: GmemoryII,
                                   gmem_i: GmemoryI, gmem_iii: 'GmemoryIII'):
        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is None or self.anchor_position is None:
            self._transition_to_macro(gmem_i, gmem_ii, gmem_iii)
            return
        self._record_micro_observation(local_nodes, sem_node)

    def _process_macro_observation(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                                   local_nodes: List[ModalityNode], gmem_ii: GmemoryII):
        self.anchor_position = self.fixation_point
        if local_nodes:
            self._start_micro_learning(X_subspaces, gpos, gmem_ii, local_nodes, self.fixation_point, 'macro')

    def _process_phase_observation(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                                   local_nodes: List[ModalityNode], gmem_i: GmemoryI,
                                   gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII',
                                   S_maps_footprint: Dict[int, torch.Tensor], H: int, W: int):
        if self.state == MainPhase.REVIEW:
            self._process_review_observation(
                X_subspaces, gpos, local_nodes, gmem_i, gmem_ii, gmem_iii, S_maps_footprint, H, W
            )
        elif self.state == MainPhase.LEARN_MICRO:
            self._process_micro_observation(local_nodes, gmem_ii, gmem_i, gmem_iii)
        elif self.state == MainPhase.LEARN_MACRO:
            self._process_macro_observation(X_subspaces, gpos, local_nodes, gmem_ii)

    def _decide_next_by_phase(self, local_nodes: List[ModalityNode], gmem_i: GmemoryI,
                              gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII',
                              active_gmem_i: List[ModalityNode], active_gmem_ii: List[SemanticNode],
                              S_maps_footprint: Dict[int, torch.Tensor]):
        cx, cy = self.fixation_point
        self.prev_fixation_point = self.fixation_point

        if self.state == MainPhase.REVIEW:
            I_map = self.optimizer.calculate_I_map(active_gmem_i, active_gmem_ii, S_maps_footprint, (cx, cy))
            self.fixation_point = self._decide_next_saccade_review(I_map)
            self.current_interest_map = I_map
            return

        if self.state == MainPhase.LEARN_MACRO:
            self.fixation_point, I_macro = self._decide_next_saccade_macro(gmem_ii, S_maps_footprint)
            self.current_interest_map = I_macro
            return

        if self.state == MainPhase.LEARN_MICRO:
            exit_type = self._check_micro_exit(self.semantic_mask, local_nodes, gmem_ii, gmem_i)
            if exit_type != 'continue':
                self._transition_to_macro(gmem_i, gmem_ii, gmem_iii, exit_type)
                if self.state == MainPhase.LEARN_MACRO:
                    self.fixation_point, I_macro = self._decide_next_saccade_macro(gmem_ii, S_maps_footprint)
                    self.current_interest_map = I_macro
                else:
                    self.fixation_point, drive = self._decide_next_saccade_micro_by_intention(
                        S_maps_footprint, local_nodes, gmem_i, gmem_ii
                    )
                    self.current_interest_map = drive
                return

            self.fixation_point, drive = self._decide_next_saccade_micro_by_intention(
                S_maps_footprint, local_nodes, gmem_i, gmem_ii
            )
            self.current_interest_map = drive
            self.micro_explore_time = getattr(self, 'micro_explore_time', 0.0) + 1.0
            if self.anchor_position is not None:
                ax, ay = self.anchor_position
                dist = math.sqrt((self.fixation_point[0] - ax) ** 2 + (self.fixation_point[1] - ay) ** 2)
                self.micro_max_span = max(getattr(self, 'micro_max_span', 0.0), dist)

    def _update_semantic_energy_from_overlap(self, gmem_ii: GmemoryII, S_maps_footprint: Dict[int, torch.Tensor],
                                             cx: int, cy: int):
        if self.state == MainPhase.LEARN_MICRO and self.active_semantic_id is not None:
            self.E_input_gmem_ii_acc[self.active_semantic_id] += 1.0

        for sem_node in gmem_ii.semantic_nodes.values():
            overlap_val = 0.0
            for nid in sem_node.related_node_ids():
                if nid in S_maps_footprint:
                    overlap_val += S_maps_footprint[nid][0, 0, cy, cx].item()
            if overlap_val > 0.0:
                self.E_input_gmem_ii_acc[sem_node.node_id] += overlap_val

    def _tick_all_nodes(self, gmem_i: GmemoryI, gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII'):
        for n in gmem_i.nodes.values():
            n.tick_update(self.E_input_gmem_i_acc[n.node_id], self.cfg)
        for n in gmem_ii.semantic_nodes.values():
            n.tick_update(self.E_input_gmem_ii_acc[n.node_id], self.cfg)
        for n in gmem_iii.entity_nodes.values():
            n.tick_update(self.E_input_gmem_iii_acc[n.node_id], self.cfg)

        self.E_input_gmem_i_acc.clear()
        self.E_input_gmem_ii_acc.clear()
        self.E_input_gmem_iii_acc.clear()

    def run_step(self, X_subspaces: Dict[int, torch.Tensor],
                 gmem_i: GmemoryI, gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII', gpos: Gposition):
        self.current_step += 1
        cx, cy = self.fixation_point
        B, C, H, W = self._first_map_shape(X_subspaces)
        cx, cy = max(0, min(cx, W-1)), max(0, min(cy, H-1))
        self.fixation_point = (cx, cy)
        self._ensure_runtime_maps(H, W)

        # 1. Local extraction and node stimulation
        local_nodes = self._extract_and_match_local_features(X_subspaces, gmem_i, cx, cy)
        for n in local_nodes:
            self.E_input_gmem_i_acc[n.node_id] += 1.0

        # 2. GposI response maps for currently active or stimulated GmemI nodes
        active_gmem_i = [n for n in gmem_i.nodes.values() if n.state_flag != NeuronState.CALM or self.E_input_gmem_i_acc[n.node_id] > 0]
        S_maps_footprint = gpos.l1_footprint_projection(X_subspaces, active_gmem_i)
        active_gmem_ii = self._active_semantic_nodes(gmem_ii)

        # 3. Cross-layer stimulation and current-state observation
        self._update_semantic_energy_from_overlap(gmem_ii, S_maps_footprint, cx, cy)
        self._process_phase_observation(
            X_subspaces, gpos, local_nodes, gmem_i, gmem_ii, gmem_iii, S_maps_footprint, H, W
        )
        active_gmem_ii = self._active_semantic_nodes(gmem_ii)

        # 4. Intention-specific decision and unified end-of-frame dynamics
        self._decide_next_by_phase(
            local_nodes, gmem_i, gmem_ii, gmem_iii, active_gmem_i, active_gmem_ii, S_maps_footprint
        )
        self._tick_all_nodes(gmem_i, gmem_ii, gmem_iii)


# =============================
# MultilevelCoordinator
# =============================
class MultilevelCoordinator:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.gmem_i = GmemoryI()
        self.gmem_ii = GmemoryII(cfg)
        self.gmem_iii = GmemoryIII(cfg)
        self.gpos = Gposition(cfg)
        self.controller = Controller(cfg)
        self.current_view = self.controller.fixation_point
        self.viewroute = []
        
    def handle_new_view(self, features: dict, H_orig: int = None, W_orig: int = None):
        X_subspaces = {}
        if 'grad' in features and features['grad'] is not None:
             X_subspaces[0] = features['grad'].detach()
        if 'hue' in features and features['hue'] is not None:
             X_subspaces[1] = features['hue'].detach()
        if 'curvature' in features and features['curvature'] is not None:
             X_subspaces[2] = features['curvature'].detach()
        if 'aspect' in features and features['aspect'] is not None:
             X_subspaces[3] = features['aspect'].detach()
        if 'orientation' in features and features['orientation'] is not None:
             X_subspaces[4] = features['orientation'].detach()
             
        if not X_subspaces:
            return
        
        if self.controller.optimizer.base_interest is None:
            B, C, H, W = list(X_subspaces.values())[0].shape
            H_o = H_orig if H_orig is not None else H
            W_o = W_orig if W_orig is not None else W
            self.controller.optimizer.initialize_base_interest(X_subspaces, H_o, W_o)
            
        self.controller.run_step(X_subspaces, self.gmem_i, self.gmem_ii, self.gmem_iii, self.gpos)
        self.current_view = self.controller.fixation_point
        self.viewroute.append(self.current_view)

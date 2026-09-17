import math
import time
from dataclasses import dataclass, field
from collections import Counter, deque
import copy
import heapq
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, OrderedDict

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
        0: {'metric': 'Vector_Occupancy', 'paradigm': 'A', 'dominant_channel': 0},
        1: {'metric': 'Euclidean_Absolute', 'paradigm': 'A', 'dominant_metric': 'Isurf'},
        2: {'metric': 'Euclidean_Absolute', 'paradigm': 'B'},
        3: {'metric': 'Euclidean_Absolute', 'paradigm': 'C'},
        4: {'metric': 'Periodic_Property', 'paradigm': 'C', 'periods': {0: 2.0, 1: 2.0, 2: 2.0}}
    }
    
    # [修改] 通道属性划分 (STRENGTH vs CONTINUITY_SURFACE vs CONTINUITY_TRACE vs IGNORE)
    feature_attributes = {
        0: {0: 'STRENGTH', 1: 'CONTEXT', 2: 'CONTINUITY_TRACE', 3: 'CONTINUITY_TRACE'},
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

    # Channel-contract gates and grouped similarity params
    min_valid_sim_channels = 1
    tau_grad_str = 0.8
    tau_grad_coh = 0.6
    grad_coh_radius = 1
    grad_axis_k = 2
    gamma_grad = 2.0
    tau_rgb_var = 0.02
    rgb_gate_radius = 1
    tau_curv = 0.05
    tau_aps = 0.05
    tau_ori_coh = 0.6
    ori_coh_radius = 1
    ori_axis_k = 2
    gamma_ori = 2.0

    feature_specs = {
        0: {
            'name': 'grad',
            'topology': 'Boundary_Edge',
            'channels': {
                0: {'name': 'abs_strength', 'role': 'gate_only', 'gate': 'grad_strength'},
                1: {'name': 'theta', 'role': 'context_only', 'gate': 'grad_coherence'},
                2: {'name': 'gradx', 'role': 'sim_only', 'metric_group': 'grad_vec', 'parents': ['grad_strength', 'grad_coherence']},
                3: {'name': 'grady', 'role': 'sim_only', 'metric_group': 'grad_vec', 'parents': ['grad_strength', 'grad_coherence']},
            },
            'metric_groups': {
                'grad_vec': {
                    'metric': 'Vector_Occupancy',
                    'channels': [2, 3],
                    'gamma': 2.0,
                    'polarity': 'axis',
                    'theta_match': 0.75,
                    'min_valid_channels': 2,
                }
            },
        },
        1: {
            'name': 'RGB',
            'topology': 'Surface_2D',
            'channels': {
                0: {'name': 'R', 'role': 'gate_and_sim', 'metric_group': 'rgb', 'parents': ['rgb_surface']},
                1: {'name': 'G', 'role': 'gate_and_sim', 'metric_group': 'rgb', 'parents': ['rgb_surface']},
                2: {'name': 'B', 'role': 'gate_and_sim', 'metric_group': 'rgb', 'parents': ['rgb_surface']},
            },
            'metric_groups': {
                'rgb': {
                    'metric': 'Euclidean_Absolute',
                    'channels': [0, 1, 2],
                    'sigma': [0.06, 0.06, 0.06],
                    'theta_match': 0.85,
                    'min_valid_channels': 1,
                }
            },
        },
        2: {
            'name': 'Curv',
            'topology': 'Trace_1D',
            'channels': {},
            'metric_groups': {
                'curv': {
                    'metric': 'Euclidean_Absolute',
                    'channels': 'all',
                    'sigma': 0.08,
                    'theta_match': 0.80,
                    'min_valid_channels': 1,
                }
            },
        },
        3: {
            'name': 'aps',
            'topology': 'Surface_2D',
            'channels': {},
            'metric_groups': {
                'aps': {
                    'metric': 'Euclidean_Absolute',
                    'channels': 'all',
                    'sigma': 0.10,
                    'theta_match': 0.80,
                    'min_valid_channels': 1,
                }
            },
        },
        4: {
            'name': 'ori',
            'topology': 'Trace_1D',
            'channels': {},
            'metric_groups': {
                'ori': {
                    'metric': 'Periodic_Property',
                    'channels': 'all',
                    'gamma': 2.0,
                    'k': 2.0,
                    'theta_match': 0.75,
                    'min_valid_channels': 1,
                }
            },
        },
    }

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
    theta_boundary_hit_enter = 0.04
    theta_boundary_hit_exit = 0.015
    theta_trace_fwd_enter = 0.03
    theta_trace_fwd_exit = 0.01
    theta_boundary_role = 0.04
    boundary_hit_radius = 3
    boundary_pending_radius = 4.0
    trace_min_age = 3
    sigma_boundary_reject = 10.0
    boundary_reject_decay = 0.92
    sigma_boundary_suppression = 4.0
    boundary_coverage_suppression = 0.35

    # OnceGraphBuilder: distances are measured in feature-map pixels.
    once_support_threshold = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05}
    once_seed_threshold = 0.6
    once_seed_nms_radius = 4
    once_similarity_min = {0: 0.75, 1: 0.85, 2: 0.8, 3: 0.8, 4: 0.75}
    once_affinity_min = 0.01
    once_tangent_min = 0.75
    once_alignment_min = 0.45
    once_ell_parallel = 2.0
    once_ell_perp = 0.65
    once_surface_sigma = 2.0
    once_direction_coherence = 0.35
    once_thin_gradient_ridges = True
    once_surface_barrier = 0.8
    once_range_max = {1: 2.0, 3: 2.0}
    once_min_support = {0: 4, 1: 16, 2: 3, 3: 8, 4: 4}
    once_region_budget = 256
    once_sample_budget = {0: 1024, 1: 1024, 2: 512, 3: 512, 4: 512}
    once_samples_per_region = 64
    once_curve_spacing = 8.0
    once_curve_error = 1.5
    once_surface_cover_radius = 12.0
    once_boundary_spacing = 8.0
    once_contact_radius = 4
    once_contact_spacing = 8.0
    once_contacts_per_pair = 8
    once_contact_budget = 1024
    once_geometry_epsilon = 1e-5
    once_node_reuse_mode = 'sample_instance'
    once_write_local_contacts = False
    once_rejection_examples = 32

    # Hierarchical hypergraph retrieval: translation-only first implementation.
    graph_recall_threshold = 0.45
    graph_bit_coverage_min = 0.1
    graph_verify_threshold = 0.65
    graph_learn_threshold = 0.78
    graph_match_margin = 0.04
    graph_evaluable_min = 0.7
    graph_coverage_min = 0.7
    graph_geometry_radius = 2
    graph_geometry_sigma = 2.0
    graph_cycle_tolerance = 1.0
    graph_nms_radius = 6.0
    graph_query_stride = 4
    graph_events_per_class = 64
    graph_candidate_budget = 128
    graph_poses_per_node = 8
    graph_pose_bin = 3.0
    graph_response_cache = 32
    graph_query_chunk = 2048
    graph_evidence_prior = 1.0
    graph_maturity_support = 3.0
    graph_stable_support = 3.0
    graph_stable_variance = 9.0
    graph_member_radius = 96.0
    graph_max_members = 6
    graph_min_members = 2
    graph_proposal_budget = 64
    graph_geometry_update_limit = 4.0
    graph_candidate_expiry = 50
    graph_candidate_memory_budget = 512
    graph_member_remove_reliability = 0.25


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
    evidence: Any = field(default_factory=lambda: EvidenceStats())

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


@dataclass
class FeatureDescriptor:
    modality_id: int
    values: torch.Tensor
    sim_mask: torch.Tensor
    gate_mask: torch.Tensor
    context_mask: torch.Tensor
    gate_context: Dict[str, float]
    topology_context: Dict[str, Any]
    match_threshold: float


# =============================
# Gmemory: Modality Nodes & Semantic Nodes
# =============================
class ModalityNode:
    """模态孤立特征层节点 (GI)"""
    def __init__(self, node_id: int, modality_id: int, prototype: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 gate_context: Optional[Dict[str, float]] = None,
                 topology_context: Optional[Dict[str, Any]] = None,
                 match_threshold: Optional[float] = None):
        self.node_id = node_id
        self.modality_id = modality_id
        self.prototype = prototype
        self.mask = mask if mask is not None else torch.ones_like(prototype)
        self.gate_context = gate_context or {}
        self.topology_context = topology_context or {}
        self.match_threshold = match_threshold
        
        self.count = 1
        self.last_seen = 0
        self.stability_score = 1.0
        
        # 激活状态
        self.activation_level = 0.0
        self.state_flag = NeuronState.CALM
        self.timer = 0

    def update_from_descriptor(self, descriptor: FeatureDescriptor):
        sim_mask = descriptor.sim_mask.squeeze(0).to(self.prototype.device)
        values = descriptor.values.squeeze(0).to(self.prototype.device)
        if values.shape[0] == self.prototype.shape[0]:
            alpha = 1.0 / (self.count + 1)
            self.prototype = self.prototype * (1.0 - sim_mask) + (
                (1.0 - alpha) * self.prototype + alpha * values
            ) * sim_mask
            self.mask = torch.max(self.mask, sim_mask)
        for key, value in descriptor.gate_context.items():
            prev = self.gate_context.get(key)
            self.gate_context[key] = value if prev is None else 0.9 * prev + 0.1 * value
        self.topology_context.update(descriptor.topology_context)
        self.match_threshold = descriptor.match_threshold

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
        self.kind = 'region'
        self.version = 0
        self.evidence = EvidenceStats()
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
        self.root_slot = None
        self.relation_edges = {}
        self.evidence = EvidenceStats()
        self.status = 'candidate'
        self.version = 0
        self.last_observed = 0
        self.pending_members = {}
        
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

    def add_node(self, modality_id: int, prototype: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 gate_context: Optional[Dict[str, float]] = None,
                 topology_context: Optional[Dict[str, Any]] = None,
                 match_threshold: Optional[float] = None) -> ModalityNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = ModalityNode(
            nid,
            modality_id,
            prototype,
            mask,
            gate_context=gate_context,
            topology_context=topology_context,
            match_threshold=match_threshold,
        )
        self.nodes[nid] = node
        return node


class GmemoryII:
    """Gmemory II 层：锚点-外围拓扑层"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.semantic_nodes: Dict[int, SemanticNode] = {}
        self.next_node_id = 0
        self.next_relation_edge_id = 0
        self.parent_entities = defaultdict(set)
        self.region_relations = {}

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
        self.observations = 0
        
    def add_entity_node(self) -> EntityNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = EntityNode(nid)
        self.entity_nodes[nid] = node
        return node
        
    def add_component(self, entity_node: EntityNode, sem_id: int, dx: float, dy: float):
        edge_id = entity_node.next_component_edge_id
        entity_node.next_component_edge_id += 1
        component = {'sem_id': sem_id, 'dx': float(dx), 'dy': float(dy), 'count': 1,
                     'slot_id': edge_id, 'scale': 1.0, 'angle': 0.0,
                     'evidence': EvidenceStats()}
        entity_node.component_edges[edge_id] = component
        entity_node.component_edges_by_semantic[sem_id].append(edge_id)
        # Legacy representative component for older consumers.
        entity_node.components.setdefault(sem_id, {'dx': dx, 'dy': dy})
        if entity_node.root_slot is None:
            entity_node.root_slot = edge_id
        return component


# =============================
# Similarity Engine
# =============================
class SimilarityEngine:
    @staticmethod
    def _result_zeros(X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 4:
            return torch.zeros((X.shape[0], 1, X.shape[-2], X.shape[-1]), device=X.device, dtype=X.dtype)
        return torch.zeros((X.shape[0], 1), device=X.device, dtype=X.dtype)

    @staticmethod
    def _base_mask(mask: Optional[torch.Tensor], channels: int, device, dtype) -> torch.Tensor:
        if mask is None:
            return torch.ones((1, channels), device=device, dtype=dtype)
        out = mask.to(device=device, dtype=dtype).view(1, -1)
        if out.shape[1] < channels:
            pad = torch.zeros((1, channels - out.shape[1]), device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=1)
        return out[:, :channels]

    @staticmethod
    def _group_channels(channels_spec, channels: int) -> List[int]:
        if channels_spec == 'all':
            return list(range(channels))
        return [int(c) for c in channels_spec if int(c) < channels]

    @staticmethod
    def _group_mask(mask_X: torch.Tensor, mask_W: torch.Tensor,
                    group_channels: List[int], channels: int) -> torch.Tensor:
        group = torch.zeros((1, channels), device=mask_X.device, dtype=mask_X.dtype)
        for c in group_channels:
            if c < channels:
                group[0, c] = 1.0
        return mask_X * mask_W * group

    @staticmethod
    def _expand_mask(mask: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 4:
            return mask.view(1, -1, 1, 1)
        return mask

    @staticmethod
    def _parameter_tensor(value, channels: int, device, dtype) -> torch.Tensor:
        if isinstance(value, (list, tuple)):
            data = list(value)[:channels]
            if len(data) < channels:
                data.extend([data[-1] if data else 1.0] * (channels - len(data)))
            return torch.tensor(data, device=device, dtype=dtype).view(1, channels)
        return torch.full((1, channels), float(value), device=device, dtype=dtype)

    @staticmethod
    def sim_euclidean(X: torch.Tensor, W: torch.Tensor, mask_X: torch.Tensor, mask_W: torch.Tensor,
                      sigma=0.5, weights=1.0) -> torch.Tensor:
        """欧氏绝对坐标: 高斯RBF核"""
        M = mask_X * mask_W
        sigma_t = SimilarityEngine._parameter_tensor(sigma, X.shape[1], X.device, X.dtype).clamp(min=1e-6)
        weight_t = SimilarityEngine._parameter_tensor(weights, X.shape[1], X.device, X.dtype)
        if X.dim() == 4:
            M = M.view(1, -1, 1, 1)
            sigma_t = sigma_t.view(1, -1, 1, 1)
            weight_t = weight_t.view(1, -1, 1, 1)

        diff_sq = torch.sum((((X - W) / sigma_t) ** 2) * M * weight_t, dim=1, keepdim=True)
        valid_channels = torch.sum(M * weight_t, dim=1, keepdim=True).clamp(min=1e-6)
        dist_sq = diff_sq / valid_channels
        return torch.exp(-0.5 * dist_sq)

    @staticmethod
    def sim_vector(X: torch.Tensor, W: torch.Tensor, mask_X: torch.Tensor, mask_W: torch.Tensor,
                   gamma: float = 2.0, polarity: str = 'axis') -> torch.Tensor:
        """矢量占有度: 归一化向量相似度，可选择轴向或极性敏感匹配。"""
        M = mask_X * mask_W
        M_exp = SimilarityEngine._expand_mask(M, X)
        Xv = X * M_exp
        Wv = W * M_exp
        dot = torch.sum(Xv * Wv, dim=1, keepdim=True)
        norm_x = torch.sqrt(torch.sum(Xv * Xv, dim=1, keepdim=True) + 1e-6)
        norm_w = torch.sqrt(torch.sum(Wv * Wv, dim=1, keepdim=True) + 1e-6)
        cos_sim = dot / (norm_x * norm_w)
        if polarity == 'axis':
            cos_sim = torch.abs(cos_sim)
        else:
            cos_sim = torch.clamp(cos_sim, min=0.0)
        return torch.pow(torch.clamp(cos_sim, min=0.0, max=1.0), gamma)

    @staticmethod
    def sim_periodic(X: torch.Tensor, W: torch.Tensor, mask_X: torch.Tensor, mask_W: torch.Tensor,
                     periods: Dict[int, float], gamma: float = 2.0) -> torch.Tensor:
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
            
            S_raw = (1.0 + torch.cos(k * (Xc - Wc))) * 0.5
            S_prop = torch.pow(torch.clamp(S_raw, min=0.0, max=1.0), gamma)
            sum_sim = sum_sim + S_prop * Mc
            count = count + Mc
            
        count = count.clamp(min=1e-6)
        return sum_sim / count

    @staticmethod
    def calculate_similarity(X: torch.Tensor, W: torch.Tensor, cfg: 'MemoryConfig', mod_id: int, mask_X: Optional[torch.Tensor] = None, mask_W: Optional[torch.Tensor] = None) -> torch.Tensor:
        specs = getattr(cfg, 'feature_specs', {})
        spec = specs.get(mod_id)
        channels = min(X.shape[1], W.shape[1])
        if channels <= 0:
            return SimilarityEngine._result_zeros(X)

        X = X[:, :channels]
        W = W[:, :channels] if W.dim() != 4 else W[:, :channels, :, :]
        mask_X = SimilarityEngine._base_mask(mask_X, channels, X.device, X.dtype)
        mask_W = SimilarityEngine._base_mask(mask_W, channels, X.device, X.dtype)

        if spec and spec.get('metric_groups'):
            log_sum = None
            alpha_sum = 0.0
            for group in spec['metric_groups'].values():
                group_channels = SimilarityEngine._group_channels(group.get('channels', 'all'), channels)
                group_mask = SimilarityEngine._group_mask(mask_X, mask_W, group_channels, channels)
                valid_count = int(torch.sum(group_mask).item())
                min_valid = int(group.get('min_valid_channels', getattr(cfg, 'min_valid_sim_channels', 1)))
                if valid_count < min_valid:
                    continue

                metric = group.get('metric', 'Vector_Occupancy')
                if metric == 'Euclidean_Absolute':
                    S_g = SimilarityEngine.sim_euclidean(
                        X, W, group_mask, torch.ones_like(group_mask),
                        sigma=group.get('sigma', 0.05),
                        weights=group.get('weights', 1.0),
                    )
                elif metric == 'Periodic_Property':
                    k = group.get('k', group.get('period', 2.0))
                    periods = {c: k for c in group_channels}
                    S_g = SimilarityEngine.sim_periodic(
                        X, W, group_mask, torch.ones_like(group_mask),
                        periods,
                        gamma=group.get('gamma', 2.0),
                    )
                else:
                    S_g = SimilarityEngine.sim_vector(
                        X, W, group_mask, torch.ones_like(group_mask),
                        gamma=group.get('gamma', 2.0),
                        polarity=group.get('polarity', 'axis'),
                    )

                alpha = float(group.get('weight', 1.0))
                term = alpha * torch.log(S_g.clamp(min=1e-6))
                log_sum = term if log_sum is None else log_sum + term
                alpha_sum += alpha

            if log_sum is None or alpha_sum <= 0.0:
                return SimilarityEngine._result_zeros(X)
            return torch.exp(log_sum / alpha_sum)

        finfo = getattr(cfg, 'feature_types', {}).get(mod_id, {})
        metric = finfo.get('metric', 'Vector_Magnitude')

        if metric == 'Euclidean_Absolute':
            sigma = finfo.get('sigma', 0.05)
            return SimilarityEngine.sim_euclidean(X, W, mask_X, mask_W, sigma)
        elif metric in {'Vector_Magnitude', 'Vector_Occupancy'}:
            return SimilarityEngine.sim_vector(X, W, mask_X, mask_W)
        elif metric == 'Periodic_Property':
            periods = finfo.get('periods', {0: 1.0})
            gamma = finfo.get('gamma', 2.0)
            return SimilarityEngine.sim_periodic(X, W, mask_X, mask_W, periods, gamma)
        else:
            return SimilarityEngine.sim_vector(X, W, mask_X, mask_W)

    @staticmethod
    def _mean_filter(value: torch.Tensor, radius: int) -> torch.Tensor:
        kernel = max(1, 2 * int(radius) + 1)
        return F.avg_pool2d(value, kernel_size=kernel, stride=1, padding=kernel // 2)

    @staticmethod
    def _circular_coherence_map(theta: torch.Tensor, weight: torch.Tensor,
                                radius: int, k: float) -> torch.Tensor:
        sin_v = torch.sin(k * theta) * weight
        cos_v = torch.cos(k * theta) * weight
        sum_sin = SimilarityEngine._mean_filter(sin_v, radius)
        sum_cos = SimilarityEngine._mean_filter(cos_v, radius)
        sum_w = SimilarityEngine._mean_filter(weight, radius).clamp(min=1e-6)
        return torch.sqrt(sum_sin ** 2 + sum_cos ** 2) / sum_w

    @staticmethod
    def compute_gate_map(X: torch.Tensor, cfg: 'MemoryConfig', mod_id: int) -> torch.Tensor:
        B, C, H, W = X.shape
        if C <= 0:
            return torch.zeros((B, 1, H, W), device=X.device, dtype=X.dtype)

        if mod_id == 0:
            if C >= 4:
                strength = X[:, 0:1].abs()
                theta = X[:, 1:2]
            elif C >= 2:
                gx, gy = X[:, 0:1], X[:, 1:2]
                strength = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
                theta = torch.atan2(gy, gx)
            else:
                strength = X[:, 0:1].abs()
                theta = torch.zeros_like(strength)
            coh = SimilarityEngine._circular_coherence_map(
                theta,
                strength,
                getattr(cfg, 'grad_coh_radius', 1),
                getattr(cfg, 'grad_axis_k', 2),
            )
            return ((strength > cfg.tau_grad_str) & (coh > cfg.tau_grad_coh)).to(dtype=X.dtype)

        if mod_id == 1:
            radius = getattr(cfg, 'rgb_gate_radius', 1)
            mean = SimilarityEngine._mean_filter(X, radius)
            var = SimilarityEngine._mean_filter((X - mean) ** 2, radius).mean(dim=1, keepdim=True)
            return (var < cfg.tau_rgb_var).to(dtype=X.dtype)

        if mod_id == 2:
            return (X.abs() > cfg.tau_curv).any(dim=1, keepdim=True).to(dtype=X.dtype)

        if mod_id == 3:
            return (X.abs() > cfg.tau_aps).any(dim=1, keepdim=True).to(dtype=X.dtype)

        if mod_id == 4:
            weight = torch.ones_like(X[:, 0:1])
            gates = []
            for c in range(C):
                coh = SimilarityEngine._circular_coherence_map(
                    X[:, c:c+1],
                    weight,
                    getattr(cfg, 'ori_coh_radius', 1),
                    getattr(cfg, 'ori_axis_k', 2),
                )
                gates.append(coh > cfg.tau_ori_coh)
            return torch.cat(gates, dim=1).any(dim=1, keepdim=True).to(dtype=X.dtype)

        return torch.ones((B, 1, H, W), device=X.device, dtype=X.dtype)


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
            G_m = SimilarityEngine.compute_gate_map(X_m, self.cfg, mod_id)
            
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            S_maps[node.node_id] = S_m * G_m * weight
            
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
            G_m = SimilarityEngine.compute_gate_map(X_m, self.cfg, mod_id)
            
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            S_maps[node.node_id] = S_m * G_m * weight
            
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

    @torch.no_grad()
    def l2_structure_synthesis(self, S_maps, semantic_node, gmem_i=None, valid_mask=None):
        """Full fixed-pose response plus verified multi-peak hypotheses.

        Pass gmem_i to remove the known modality weights from legacy GposI maps.
        Returns the historical x/y/score keys as well as diagnostics and response.
        """
        if not S_maps:
            return None
        provider = _StoredMapProvider(S_maps, gmem_i.nodes if gmem_i else None,
                                      self.cfg if gmem_i else None, valid_mask)
        view = region_view(semantic_node, self.cfg)
        missing = {s['node_id'] for s in view.slots} - set(provider.nodes)
        if missing:
            from types import SimpleNamespace
            provider.nodes = dict(provider.nodes)
            provider.nodes.update({nid: SimpleNamespace(modality_id=nid) for nid in missing})
        h, w = provider.shape
        centers = [(x, y) for y in range(h) for x in range(w)]
        matcher = SpatialStructureMatcher(self.cfg)
        matches = matcher.evaluate(view, provider, centers)
        if not matches:
            return None
        peaks = matcher.nms(matches)
        best = peaks[0] if peaks else max(matches, key=lambda m: m.score)
        reference = next(iter(S_maps.values()))
        response = torch.tensor([m.score for m in matches], device=reference.device,
                                dtype=reference.dtype).reshape(1, 1, h, w)
        return {'x': best.point[0], 'y': best.point[1], 's_star': 1., 'theta_star': 0.,
                'score': best.score, 'accepted': best.accepted, 'response': response,
                'matches': peaks, 'diagnostics': best.summary()}

    @torch.no_grad()
    def l3_structure_synthesis(self, features, entity, gmem_i, gmem_ii,
                              valid_mask=None, centers=None):
        provider = FeatureResponseCache(features, gmem_i, self.cfg, valid_mask)
        if not provider.inputs:
            return []
        h, w = provider.shape
        if centers is None:
            centers = [(x, y) for y in range(h) for x in range(w)]
        matcher = SpatialStructureMatcher(self.cfg)
        return matcher.nms(matcher.evaluate(entity_view(entity, gmem_ii, self.cfg), provider, centers))


# =============================
# Deterministic, single-observation region builder
# Gmem, Gpos and SimilarityEngine above retain their original implementations.
# =============================

_NEIGHBORS = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if dx or dy)
_HALF_NEIGHBORS = ((1, 0), (-1, 1), (0, 1), (1, 1))


def _setting(value, modality, default):
    return value.get(modality, default) if isinstance(value, dict) else value


def _components(points, width):
    """Eight-connected components, ordered by their first flat pixel index."""
    unseen = set(map(int, points))
    result = []
    for start in sorted(unseen):
        if start not in unseen:
            continue
        unseen.remove(start)
        queue, component = deque([start]), []
        while queue:
            p = queue.popleft()
            component.append(p)
            y, x = divmod(p, width)
            for dx, dy in _NEIGHBORS:
                if not 0 <= x + dx < width or y + dy < 0:
                    continue
                q = (y + dy) * width + x + dx
                if q in unseen:
                    unseen.remove(q)
                    queue.append(q)
        result.append(sorted(component))
    return result


def _boundary(mask):
    padded = np.pad(mask, 1, constant_values=False)
    interior = np.ones_like(mask, dtype=bool)
    for dx, dy in _NEIGHBORS:
        interior &= padded[1 + dy:1 + dy + mask.shape[0],
                           1 + dx:1 + dx + mask.shape[1]]
    return mask & ~interior


def _distance_to_boundary(mask):
    """Grid distance, including image edges; used only to rank anchors/seeds."""
    h, w = mask.shape
    distance = np.full((h, w), np.inf)
    boundary = _boundary(mask)
    distance[boundary] = 0.0
    queue = deque(map(int, np.flatnonzero(boundary)))
    while queue:
        p = queue.popleft()
        y, x = divmod(p, w)
        for dx, dy in _NEIGHBORS:
            xx, yy = x + dx, y + dy
            if (0 <= xx < w and 0 <= yy < h and mask[yy, xx]
                    and distance[yy, xx] > distance[y, x] + 1):
                distance[yy, xx] = distance[y, x] + 1
                queue.append(yy * w + xx)
    distance[~mask] = 0
    return distance


@dataclass
class FeatureSupport:
    modality_id: int
    values: torch.Tensor
    writable: np.ndarray
    quality: np.ndarray
    support: np.ndarray
    tangent: np.ndarray
    directional: np.ndarray
    events: np.ndarray
    seeds: List[int]
    sim_mask: torch.Tensor
    threshold: float
    support_dim: int
    gate_context: Dict[str, np.ndarray] = field(default_factory=dict)
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    edge_scores: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class Region:
    region_id: int
    modality_id: int
    support_dim: int
    pixels: np.ndarray
    boundary: np.ndarray
    seeds: List[int]
    adjacency: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    anchor: Optional[int] = None
    samples: List[int] = field(default_factory=list)
    semantic_id: Optional[int] = None
    coverage_error: float = float('inf')
    curve_error: float = 0.0
    budget_truncated: bool = False
    view_truncated: bool = False
    closed: bool = False
    completed: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class RegionContact:
    region_a: int
    region_b: int
    source: int
    target: int
    distance: float
    semantic_id: Optional[int] = None
    kind: str = 'region_contact'
    written: bool = False


@dataclass
class BuildReport:
    build_id: int
    shape: Tuple[int, int]
    valid_mask: np.ndarray
    supports: Dict[int, FeatureSupport] = field(default_factory=dict)
    labels: Dict[int, np.ndarray] = field(default_factory=dict)
    regions: List[Region] = field(default_factory=list)
    contacts: List[RegionContact] = field(default_factory=list)
    sample_nodes: Dict[Tuple[int, int], int] = field(default_factory=dict)
    node_ids: List[int] = field(default_factory=list)
    region_semantic_ids: List[int] = field(default_factory=list)
    contact_semantic_ids: List[int] = field(default_factory=list)
    edge_observations: List[Dict[str, Any]] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)
    rejection_examples: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def reject(self, reason, count=1, example=None):
        self.rejected[reason] = self.rejected.get(reason, 0) + int(count)
        if example is not None and len(self.rejection_examples) < int(
                self.config.get('once_rejection_examples', 32)):
            self.rejection_examples.append({'reason': reason, **example})

    def summary(self):
        return {
            'build_id': self.build_id, 'shape': self.shape,
            'regions': len(self.regions), 'sample_instances': len(self.sample_nodes),
            'referenced_nodes': len(self.node_ids),
            'region_containers': len(self.region_semantic_ids),
            'contact_containers': len(self.contact_semantic_ids),
            'completed_regions': sum(r.completed for r in self.regions),
            'truncated_regions': sum(r.budget_truncated for r in self.regions),
            'edge_observations': len(self.edge_observations),
            'rejected': dict(self.rejected), 'timings': dict(self.timings),
            'warnings': list(self.warnings),
        }


def prepare_feature_subspaces(features, cfg):
    """Accept either canonical integer subspaces or named CNN outputs.

    Integer inputs must already obey the channel contract. Named orientation
    follows the existing Retina convention (normalised by pi/2).
    """
    if not features:
        return {}
    result = {}
    if all(isinstance(key, int) for key in features):
        entries = features.items()
    else:
        aliases = {0: ('grad',), 1: ('rgb', 'RGB', 'image', 'x', 'cropped', 'hue'),
                   2: ('curvature',), 3: ('aspect',), 4: ('orientation',)}
        entries = []
        for mid, names in aliases.items():
            value = next((features[n] for n in names if features.get(n) is not None), None)
            if value is None:
                continue
            if mid == 0:
                if value.ndim == 5 and value.shape[-1] == 2:
                    dx, dy = value[..., 0], value[..., 1]
                elif value.ndim == 4 and value.shape[1] == 2:
                    dx, dy = value[:, :1], value[:, 1:2]
                elif value.ndim == 4 and value.shape[-1] == 2 and value.shape[1] != 4:
                    dx, dy = value[..., 0].unsqueeze(1), value[..., 1].unsqueeze(1)
                else:
                    dx = dy = None
                if dx is not None:
                    value = torch.cat((torch.sqrt(dx.square() + dy.square() + 1e-6),
                                       torch.atan2(dy, dx), dx, dy), dim=1)
            if mid == 4 and value.numel() and value.detach().abs().max().item() <= 1.05:
                value = value * (math.pi / 2)
            entries.append((mid, value))
    shape = None
    for mid, value in entries:
        if value is None:
            continue
        if mid not in cfg.feature_specs:
            raise ValueError('Unknown modality requires feature_specs: %r' % mid)
        if not isinstance(value, torch.Tensor):
            raise TypeError('Feature inputs must be torch tensors.')
        if value.ndim == 2:
            value = value[None, None]
        elif value.ndim == 3:
            value = value[None]
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError('One observation requires [1,C,H,W]; process batches separately.')
        if value.shape[1] == 0:
            continue
        if min(value.shape[-2:]) < 1:
            raise ValueError('Spatial dimensions must be positive.')
        if shape is not None and shape != tuple(value.shape[-2:]):
            raise ValueError('All feature maps must use the same coordinates and shape.')
        shape = tuple(value.shape[-2:])
        result[mid] = value.detach().to(device=cfg.device, dtype=torch.float32)
    return dict(sorted(result.items()))


class FeatureSupportBuilder:
    """Cache existing gates once, then describe samples without state-machine calls."""
    def __init__(self, cfg):
        self.cfg = cfg

    def _channel_masks(self, mid, channels):
        spec = self.cfg.feature_specs[mid]
        sim = torch.zeros((1, channels), device=self.cfg.device)
        gate, context = torch.zeros_like(sim), torch.zeros_like(sim)
        for c in range(channels):
            channel = spec.get('channels', {}).get(c)
            if channel is None:
                if not any(g.get('channels', 'all') == 'all' or c in g['channels']
                           for g in spec.get('metric_groups', {}).values()):
                    continue
                channel = {'role': 'gate_and_sim'}
            role = channel.get('role', 'gate_and_sim')
            sim[0, c] = role in {'gate_and_sim', 'sim_only'}
            gate[0, c] = role in {'gate_and_sim', 'gate_only'}
            context[0, c] = role in {'gate_only', 'context_only'}
        thresholds = []
        for group in spec.get('metric_groups', {}).values():
            cs = SimilarityEngine._group_channels(group.get('channels', 'all'), channels)
            if cs and sim[0, cs].sum().item() >= int(group.get('min_valid_channels', 1)):
                thresholds.append(float(group.get('theta_match', 0.85)))
        return sim, gate, context, min(thresholds) if thresholds else None

    def build(self, inputs, valid, report):
        result = {}
        for mid, original in inputs.items():
            finite = torch.isfinite(original).all(dim=1, keepdim=True)
            value = torch.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
            gate = SimilarityEngine.compute_gate_map(value, self.cfg, mid)[0, 0] > 0.5
            sim, _, _, threshold = self._channel_masks(mid, value.shape[1])
            if threshold is None:
                report.reject('insufficient_metric_channels', example={'modality': mid})
                continue
            writable = gate.cpu().numpy() & finite[0, 0].cpu().numpy() & valid
            context = {}
            spec = self.cfg.feature_specs[mid]
            dim = 1 if spec.get('topology') in {'Boundary_Edge', 'Trace_1D'} else 2
            if mid == 0 and value.shape[1] >= 4:
                strength = value[0, 0].abs()
                theta = value[:, 1:2]
                coherence = SimilarityEngine._circular_coherence_map(
                    theta, value[:, :1].abs(), self.cfg.grad_coh_radius, self.cfg.grad_axis_k)[0, 0]
                context = {'strength': strength.cpu().numpy(),
                           'theta': theta[0, 0].cpu().numpy(), 'coherence': coherence.cpu().numpy()}
                norm = torch.sqrt(value[0, 2].square() + value[0, 3].square())
                tx, ty = -value[0, 3] / norm.clamp_min(1e-8), value[0, 2] / norm.clamp_min(1e-8)
                directional = (norm > 1e-8) & (coherence >= self.cfg.once_direction_coherence)
                scale = max(float(strength[gate].max().item()) if gate.any() else 1., 1e-8)
                quality = (strength / scale).clamp(0, 1) * coherence.clamp(0, 1)
            elif dim == 1:
                if mid == 4:
                    # ori is an unoriented axis; coherence is computed in doubled angle.
                    c = torch.cos(2 * value).mean(dim=1, keepdim=True)
                    s = torch.sin(2 * value).mean(dim=1, keepdim=True)
                    axis = 0.5 * torch.atan2(s, c)
                    coherence = torch.sqrt(c.square() + s.square())[0, 0]
                    tx, ty = torch.cos(axis)[0, 0], torch.sin(axis)[0, 0]
                else:
                    # Principal axis of local support coordinates for scalar trace banks.
                    strength = value.abs().amax(dim=1, keepdim=True)
                    coords = torch.arange(-2, 3, device=value.device, dtype=value.dtype)
                    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
                    def moment(kernel):
                        return F.conv2d(strength, kernel[None, None], padding=2)[0, 0]
                    mass = moment(torch.ones_like(xx)).clamp_min(1e-8)
                    mx, my = moment(xx) / mass, moment(yy) / mass
                    cxx, cyy = moment(xx * xx) / mass - mx * mx, moment(yy * yy) / mass - my * my
                    cxy = moment(xx * yy) / mass - mx * my
                    axis = 0.5 * torch.atan2(2 * cxy, cxx - cyy)
                    coherence = torch.sqrt((cxx - cyy).square() + 4 * cxy.square()) / (cxx + cyy).clamp_min(1e-8)
                    tx, ty = torch.cos(axis), torch.sin(axis)
                quality = coherence.clamp(0, 1)
                directional = coherence >= self.cfg.once_direction_coherence
            else:
                mean = SimilarityEngine._mean_filter(value, getattr(self.cfg, 'rgb_gate_radius', 1))
                variance = SimilarityEngine._mean_filter((value - mean).square(), 1).mean(dim=1)[0]
                quality = torch.exp(-variance / max(float(self.cfg.tau_rgb_var), 1e-8))
                tx, ty = torch.ones_like(quality), torch.zeros_like(quality)
                directional = torch.zeros_like(quality, dtype=torch.bool)
                context['local_variance'] = variance.cpu().numpy()
            q = quality.cpu().numpy()
            active = writable & (q >= _setting(self.cfg.once_support_threshold, mid, 0.05))
            if mid == 0 and self.cfg.once_thin_gradient_ridges:
                # Nonmaximum suppression across the normal, with deterministic
                # asymmetric ties to reduce flat two-pixel ridges to one side.
                h, w = active.shape
                if h > 1 and w > 1:
                    yy, xx = torch.meshgrid(torch.arange(h, device=value.device, dtype=value.dtype),
                                            torch.arange(w, device=value.device, dtype=value.dtype), indexing='ij')
                    nx, ny = -ty, tx
                    flip = (nx < 0) | ((nx.abs() < 1e-8) & (ny < 0))
                    nx, ny = torch.where(flip, -nx, nx), torch.where(flip, -ny, ny)
                    def normal_sample(sign):
                        grid = torch.stack(((xx + sign * nx) * 2 / (w - 1) - 1,
                                            (yy + sign * ny) * 2 / (h - 1) - 1), dim=-1)[None]
                        return F.grid_sample(strength[None, None], grid, align_corners=True)[0, 0]
                    ridge = (strength >= normal_sample(-1)) & (strength > normal_sample(1))
                    active &= ridge.cpu().numpy()
            tangent = torch.stack((tx, ty), dim=-1).cpu().numpy()
            directions = directional.cpu().numpy() & active
            # Reject low-confidence directional connections; retain these pixels as events.
            events = active & ~directions if dim == 1 else np.zeros_like(active)
            maxima = F.max_pool2d(quality[None, None], 2 * self.cfg.once_seed_nms_radius + 1,
                                  stride=1, padding=self.cfg.once_seed_nms_radius)[0, 0].cpu().numpy()
            peaks = active & (q >= self.cfg.once_seed_threshold) & (q == maxima)
            depth = _distance_to_boundary(active) if dim == 2 else np.zeros_like(q)
            seeds = [min(group, key=lambda p: (-depth.flat[p], -q.flat[p], p))
                     for group in _components(np.flatnonzero(peaks), active.shape[1])]
            result[mid] = FeatureSupport(mid, value, writable, q, active, tangent,
                                         directions, events, seeds, sim, threshold, dim, context)
            report.reject('nonfinite_pixels', int((~finite[0, 0].cpu().numpy() & valid).sum()))
        return result

    def describe(self, support, pixel):
        h, w = support.writable.shape
        y, x = divmod(int(pixel), w)
        if not (0 <= y < h and support.writable[y, x]):
            raise ValueError('Sample is not writable: %r' % ((support.modality_id, x, y),))
        sim, gate, context, threshold = self._channel_masks(support.modality_id, support.values.shape[1])
        values = support.values[0, :, y, x].unsqueeze(0)
        return FeatureDescriptor(
            support.modality_id, values, sim, gate, context,
            {'gate': 1.0, **{key: float(v[y, x]) for key, v in support.gate_context.items()}},
            {'topology': self.cfg.feature_specs[support.modality_id].get('topology', 'Surface_2D')},
            threshold,
        )


class LocalAffinityBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, support, barrier, report):
        h, w = support.support.shape
        pairs, scores = [], []
        for dx, dy in _HALF_NEIGHBORS:
            x0, x1 = max(0, -dx), min(w, w - dx)
            y0, y1 = 0, h - dy
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            p, q = (yy * w + xx).ravel(), ((yy + dy) * w + xx + dx).ravel()
            use = support.support.flat[p] & support.support.flat[q]
            p, q = p[use], q[use]
            if not len(p):
                continue
            values = support.values[0].reshape(support.values.shape[1], -1).T
            ip = torch.as_tensor(p, device=values.device)
            iq = torch.as_tensor(q, device=values.device)
            similarity = SimilarityEngine.calculate_similarity(
                values[ip], values[iq], self.cfg, support.modality_id,
                support.sim_mask, support.sim_mask).flatten().cpu().numpy()
            similarity = np.clip(similarity, 0, 1)
            good = similarity >= _setting(self.cfg.once_similarity_min, support.modality_id, 0.8)
            report.reject('local_similarity', int((~good).sum()))
            length = math.hypot(dx, dy)
            if support.support_dim == 1:
                t = support.tangent.reshape(-1, 2)
                align_p, align_q = np.abs(t[p] @ np.array([dx, dy]) / length), np.abs(t[q] @ np.array([dx, dy]) / length)
                direction_ok = (support.directional.flat[p] & support.directional.flat[q]
                                & (np.abs((t[p] * t[q]).sum(axis=1)) >= self.cfg.once_tangent_min)
                                & (align_p >= self.cfg.once_alignment_min)
                                & (align_q >= self.cfg.once_alignment_min))
                report.reject('direction_conflict', int((good & ~direction_ok).sum()))
                good &= direction_ok
                parallel, perpendicular = self.cfg.once_ell_parallel, self.cfg.once_ell_perp
                exponent = length * length * ((align_p ** 2 + align_q ** 2) / parallel ** 2
                           + (2 - align_p ** 2 - align_q ** 2) / perpendicular ** 2) / 4
                spatial = np.exp(-exponent)
            else:
                blocked = np.maximum(barrier.flat[p], barrier.flat[q]) > self.cfg.once_surface_barrier
                report.reject('surface_barrier', int((good & blocked).sum()))
                good &= ~blocked
                spatial = math.exp(-length * length / (2 * self.cfg.once_surface_sigma ** 2))
            affinity = similarity * spatial * np.sqrt(support.quality.flat[p] * support.quality.flat[q])
            good &= affinity >= self.cfg.once_affinity_min
            pairs.append(np.column_stack((p[good], q[good])))
            scores.append(affinity[good])
        if not pairs:
            return
        support.edges = np.concatenate(pairs)
        support.edge_scores = np.concatenate(scores)
        if support.support_dim == 1:
            # Count connected exits in the eight-pixel ring, not raw pixel degree.
            ring = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
            padded = np.pad(support.support, 1)
            neighbors = np.stack([padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                                  for dx, dy in ring])
            exits = ((~neighbors) & np.roll(neighbors, -1, axis=0)).sum(axis=0)
            support.events |= support.support & (exits > 2)
            normal = ~(support.events.flat[support.edges[:, 0]] | support.events.flat[support.edges[:, 1]])
            report.reject('junction_connections', int((~normal).sum()))
            support.edges, support.edge_scores = support.edges[normal], support.edge_scores[normal]


class ContinuousRegionAssembler:
    def __init__(self, cfg):
        self.cfg = cfg

    def assemble(self, support, valid, first_id, report, region_quota=None):
        h, w = support.support.shape
        active = np.flatnonzero(support.support)
        labels = np.full((h, w), -1, dtype=np.int32)
        if not len(active):
            return [], labels
        parent = np.arange(h * w)
        size = np.ones(h * w, dtype=np.int64)
        def root(p):
            p = int(p)
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = int(parent[p])
            return p
        # Only absolute-coordinate groups enter the global range constraint.
        range_limit = self.cfg.once_range_max.get(support.modality_id)
        normalized = []
        if range_limit is not None and support.support_dim == 2:
            for group in self.cfg.feature_specs[support.modality_id].get('metric_groups', {}).values():
                if group.get('metric') != 'Euclidean_Absolute':
                    continue
                cs = SimilarityEngine._group_channels(group.get('channels', 'all'), support.values.shape[1])
                sigma = SimilarityEngine._parameter_tensor(group.get('sigma', 0.1), support.values.shape[1],
                                                           support.values.device, support.values.dtype)
                for c in cs:
                    if support.sim_mask[0, c] > 0:
                        normalized.append((support.values[0, c] / sigma[0, c].clamp_min(1e-6)).cpu().numpy().ravel())
        low = np.stack(normalized, axis=1) if normalized else None
        high = low.copy() if low is not None else None
        if len(support.edges):
            order = np.lexsort((support.edges[:, 1], support.edges[:, 0], -support.edge_scores))
            for edge_index in order:
                p, q = support.edges[edge_index]
                a, b = root(p), root(q)
                if a == b:
                    continue
                if low is not None:
                    lo, hi = np.minimum(low[a], low[b]), np.maximum(high[a], high[b])
                    if np.sqrt(np.mean((hi - lo) ** 2)) > range_limit:
                        report.reject('region_range', example={'source': int(p), 'target': int(q)})
                        continue
                if size[a] < size[b] or (size[a] == size[b] and a > b):
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                if low is not None:
                    low[a], high[a] = lo, hi
        groups = defaultdict(list)
        for p in active:
            groups[root(p)].append(int(p))
        seeds = set(support.seeds)
        regions = []
        valid_boundary = _boundary(valid).ravel()
        # Preserve the strongest supported regions under a resource cap.
        ranked_groups = sorted(groups.values(), key=lambda g: (-float(support.quality.flat[g].sum()), g[0]))
        for pixels in ranked_groups:
            event = all(support.events.flat[p] for p in pixels)
            minimum = 1 if event else _setting(self.cfg.once_min_support, support.modality_id, 3)
            if len(pixels) < minimum:
                report.reject('small_region', example={'modality': support.modality_id, 'size': len(pixels)})
                continue
            if (first_id + len(regions) >= self.cfg.once_region_budget
                    or (region_quota is not None and len(regions) >= region_quota)):
                report.reject('region_budget', example={'modality': support.modality_id, 'size': len(pixels)})
                continue
            rid = first_id + len(regions)
            points = np.asarray(pixels, dtype=np.int64)
            labels.flat[points] = rid
            mask = labels == rid
            region_seeds = sorted(seeds.intersection(pixels))
            if not region_seeds:
                region_seeds = [min(pixels, key=lambda p: (-support.quality.flat[p], p))]
            regions.append(Region(rid, support.modality_id, 0 if event else support.support_dim,
                                  points, np.flatnonzero(_boundary(mask)), region_seeds,
                                  view_truncated=bool(valid_boundary[points].any())))
        by_id = {r.region_id: r for r in regions}
        for p, q in support.edges:
            rid = int(labels.flat[p])
            if rid >= 0 and rid == labels.flat[q]:
                length = math.hypot(int(p % w - q % w), int(p // w - q // w))
                by_id[rid].adjacency.setdefault(int(p), []).append((int(q), length))
                by_id[rid].adjacency.setdefault(int(q), []).append((int(p), length))
        for region in regions:
            for neighbors in region.adjacency.values():
                neighbors.sort()
        return regions, labels


class RegionContactBuilder:
    """Find contacts by local label shifts, not all-pairs region distances."""
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, regions, labels, barrier, report):
        h, w = report.shape
        by_id = {r.region_id: r for r in regions}
        boundary_maps = {mid: np.zeros((h, w), dtype=bool) for mid in labels}
        for r in regions:
            boundary_maps[r.modality_id].flat[r.boundary] = True
        best = defaultdict(dict)
        modalities = sorted(labels)
        radius = int(self.cfg.once_contact_radius)
        offsets = sorted(((dx, dy) for dy in range(-radius, radius + 1)
                          for dx in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius),
                         key=lambda d: (d[0] * d[0] + d[1] * d[1], d[1], d[0]))
        for i, ma in enumerate(modalities):
            for mb in modalities[i:]:
                for dx, dy in offsets:
                    x0, x1 = max(0, -dx), min(w, w - dx)
                    y0, y1 = max(0, -dy), min(h, h - dy)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    la = labels[ma][y0:y1, x0:x1]
                    lb = labels[mb][y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                    use = (la >= 0) & (lb >= 0) & (la != lb)
                    if ma == mb:
                        use &= la < lb
                        use &= (boundary_maps[ma][y0:y1, x0:x1]
                                & boundary_maps[mb][y0 + dy:y1 + dy, x0 + dx:x1 + dx])
                    # Cross-modal contact also permits interior overlap.
                    yy, xx = np.nonzero(use)
                    length = math.hypot(dx, dy)
                    for y, x in zip(yy, xx):
                        a, b = int(la[y, x]), int(lb[y, x])
                        p = (int(y) + y0) * w + int(x) + x0
                        q = p + dy * w + dx
                        if ma == mb and length:
                            steps = max(abs(dx), abs(dy)) + 1
                            ys = np.rint(np.linspace(p // w, q // w, steps)).astype(int)
                            xs = np.rint(np.linspace(p % w, q % w, steps)).astype(int)
                            # A high-gradient line is not its own barrier.
                            if (by_id[a].support_dim == 2 and by_id[b].support_dim == 2
                                    and barrier[ys, xs].max() > self.cfg.once_surface_barrier):
                                continue
                        candidate = (length, q)
                        old = best[(a, b)].get(p)
                        if old is None or candidate < old:
                            best[(a, b)][p] = candidate
        contacts = []
        for (a, b), candidates in sorted(best.items()):
            first_contacts, extra_contacts = [], []
            for component in _components(candidates, w):
                order = sorted(component, key=lambda p: (candidates[p][0],
                               -min(report.supports[by_id[a].modality_id].quality.flat[p],
                                    report.supports[by_id[b].modality_id].quality.flat[candidates[p][1]]), p))
                component_selected = []
                for p in order:
                    if all(math.hypot(p % w - s % w, p // w - s // w)
                           >= self.cfg.once_contact_spacing for s in component_selected):
                        component_selected.append(p)
                if component_selected:
                    first_contacts.append(component_selected[0])
                    extra_contacts.extend(component_selected[1:])
            selected = first_contacts + extra_contacts
            # Select at least one from each component before densifying a contact band.
            if len(selected) > self.cfg.once_contacts_per_pair:
                report.reject('contact_pair_budget', len(selected) - self.cfg.once_contacts_per_pair)
                selected = selected[:self.cfg.once_contacts_per_pair]
            for p in selected:
                if len(contacts) >= self.cfg.once_contact_budget:
                    report.reject('contact_budget')
                    continue
                length, q = candidates[p]
                contacts.append(RegionContact(a, b, int(p), int(q), float(length)))
        return contacts


class RegionSampler:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _distances(region, sources):
        distance = {int(p): float('inf') for p in region.pixels}
        queue = []
        for p in sources:
            distance[int(p)] = 0.
            heapq.heappush(queue, (0., int(p)))
        while queue:
            d, p = heapq.heappop(queue)
            if d > distance[p]:
                continue
            for q, step in region.adjacency.get(p, []):
                nd = d + step
                if nd < distance[q]:
                    distance[q] = nd
                    heapq.heappush(queue, (nd, q))
        return distance

    @staticmethod
    def _ordered_curve(region):
        pixels = set(map(int, region.pixels))
        if len(pixels) == 1:
            return sorted(pixels), False
        degrees = {p: len(region.adjacency.get(p, [])) for p in pixels}
        endpoints = sorted(p for p, degree in degrees.items() if degree == 1)
        closed = all(degree == 2 for degree in degrees.values())
        if not (closed or (len(endpoints) == 2 and max(degrees.values()) <= 2)):
            return [], False
        start = endpoints[0] if endpoints else min(pixels)
        path, seen, p = [], set(), start
        while p not in seen:
            path.append(p)
            seen.add(p)
            remaining = sorted(q for q, _ in region.adjacency.get(p, []) if q not in seen)
            if not remaining:
                break
            p = remaining[0]
        return (path, closed) if len(path) == len(pixels) else ([], False)

    def _curve_keypoints(self, path, closed, width):
        if len(path) < 2:
            return list(path)
        ordered = path + [path[0]] if closed else path
        xy = np.array([(p % width, p // width) for p in ordered], dtype=float)
        required = {0, len(ordered) - 1}
        stack = [(0, len(ordered) - 1)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            vector = xy[b] - xy[a]
            norm2 = float(vector @ vector)
            t = np.clip(((xy[a:b + 1] - xy[a]) @ vector) / max(norm2, 1e-12), 0, 1)
            error = np.linalg.norm(xy[a:b + 1] - (xy[a] + t[:, None] * vector), axis=1)
            local = int(np.argmax(error))
            if error[local] > self.cfg.once_curve_error:
                middle = a + local
                required.add(middle)
                stack.extend(((a, middle), (middle, b)))
        return list(dict.fromkeys(ordered[i] for i in sorted(required)))

    @staticmethod
    def _polyline_error(path, samples, closed, width):
        chosen = [p for p in path if p in samples]
        if not chosen:
            return float('inf')
        if closed and len(chosen) > 1:
            chosen.append(chosen[0])
        xy = np.asarray([(p % width, p // width) for p in path], dtype=float)
        points = np.asarray([(p % width, p // width) for p in chosen], dtype=float)
        nearest = np.linalg.norm(xy - points[0], axis=1)
        for a, b in zip(points, points[1:]):
            v = b - a
            t = np.clip(((xy - a) @ v) / max(float(v @ v), 1e-12), 0, 1)
            nearest = np.minimum(nearest, np.linalg.norm(xy - a - t[:, None] * v, axis=1))
        return float(nearest.max(initial=0.))

    def sample(self, regions, contacts, supports, report):
        h, w = report.shape
        required_contacts = defaultdict(list)
        for c in contacts:
            required_contacts[c.region_a].append(c.source)
            required_contacts[c.region_b].append(c.target)
        remaining = {mid: int(_setting(self.cfg.once_sample_budget, mid, 512)) for mid in supports}
        # Reserve one anchor per region before spending any budget on dense regions.
        paths = {}
        for r in regions:
            support = supports[r.modality_id]
            if r.support_dim == 1:
                path, closed = self._ordered_curve(r)
                paths[r.region_id] = path
                r.closed = closed and not r.view_truncated
                if path:
                    r.anchor = path[len(path) // 2] if not closed else min(path, key=lambda p: (-support.quality.flat[p], p))
                else:
                    r.anchor = min(map(int, r.pixels), key=lambda p: (-support.quality.flat[p], p))
                    r.reasons.append('curve_order_unresolved')
            elif r.support_dim == 2:
                mask = report.labels[r.modality_id] == r.region_id
                depth = _distance_to_boundary(mask)
                cy, cx = np.mean(r.pixels // w), np.mean(r.pixels % w)
                r.anchor = min(map(int, r.pixels), key=lambda p: (-depth.flat[p], -support.quality.flat[p],
                                    (p // w - cy) ** 2 + (p % w - cx) ** 2, p))
            else:
                r.anchor = min(map(int, r.pixels), key=lambda p: (-support.quality.flat[p], p))
            if remaining[r.modality_id] > 0:
                r.samples = [r.anchor]
                remaining[r.modality_id] -= 1
            else:
                r.budget_truncated = True
                r.reasons.append('anchor_budget')
                report.reject('anchor_budget', example={'region': r.region_id})
        def add(r, p):
            if p in r.samples:
                return True
            if len(r.samples) >= self.cfg.once_samples_per_region or remaining[r.modality_id] <= 0:
                r.budget_truncated = True
                return False
            r.samples.append(int(p))
            remaining[r.modality_id] -= 1
            return True
        # Reserve all contact endpoints before optional coverage points.
        for r in regions:
            if r.samples:
                for p in sorted(set(required_contacts[r.region_id])):
                    if not add(r, p):
                        report.reject('contact_sample_budget', example={'region': r.region_id, 'pixel': p})
        for r in regions:
            if not r.samples:
                continue
            path = paths.get(r.region_id, [])
            if path:
                for p in self._curve_keypoints(path, r.closed, w):
                    if not add(r, p):
                        report.reject('curve_keypoint_budget', example={'region': r.region_id})
                        r.curve_error = float('inf')
            distance = self._distances(r, r.samples)
            # Cover boundaries (including holes) before interior points, using region paths.
            if r.support_dim == 2:
                while len(r.boundary):
                    p = max(map(int, r.boundary), key=lambda q: (distance[q], -q))
                    if distance[p] <= self.cfg.once_boundary_spacing or not add(r, p):
                        break
                    fresh = self._distances(r, [p])
                    distance = {q: min(d, fresh[q]) for q, d in distance.items()}
            radius = self.cfg.once_curve_spacing / 2 if r.support_dim == 1 else self.cfg.once_surface_cover_radius
            if r.support_dim == 0:
                radius = 0.
            while distance:
                p = max(distance, key=lambda q: (distance[q], -q))
                if distance[p] <= radius or not add(r, p):
                    break
                fresh = self._distances(r, [p])
                distance = {q: min(d, fresh[q]) for q, d in distance.items()}
            r.coverage_error = max(distance.values(), default=0.)
            if r.support_dim == 1:
                r.curve_error = self._polyline_error(path, r.samples, r.closed, w) if path else float('inf')
            boundary_ok = (r.support_dim != 2 or
                           max((distance[int(p)] for p in r.boundary), default=0.) <= self.cfg.once_boundary_spacing)
            r.completed = (r.coverage_error <= radius and boundary_ok and not r.budget_truncated
                           and 'curve_order_unresolved' not in r.reasons
                           and r.curve_error <= self.cfg.once_curve_error)
            if not r.completed:
                r.reasons.append('sampling_target_unmet')
            if self.cfg.once_write_local_contacts and path:
                ordered = [p for p in path if p in r.samples]
                links = list(zip(ordered, ordered[1:]))
                if r.closed and len(ordered) > 2:
                    links.append((ordered[-1], ordered[0]))
                for p, q in links:
                    if len(contacts) >= self.cfg.once_contact_budget:
                        report.reject('local_contact_budget')
                        continue
                    contacts.append(RegionContact(r.region_id, r.region_id, p, q,
                                    math.hypot(p % w - q % w, p // w - q // w), kind='local_curve'))


class GraphWriteAdapter:
    """Stage a whole observation before touching the supplied memory pools."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.describer = FeatureSupportBuilder(cfg)

    def write(self, report, gmem_i, gmem_ii):
        mode = self.cfg.once_node_reuse_mode
        if mode not in {'sample_instance', 'prototype'}:
            raise ValueError('once_node_reuse_mode must be sample_instance or prototype.')
        staged_i, staged_ii = GmemoryI(), GmemoryII(self.cfg)
        staged_i.next_node_id = gmem_i.next_node_id
        staged_ii.next_node_id = gmem_ii.next_node_id
        staged_ii.next_relation_edge_id = gmem_ii.next_relation_edge_id
        # Existing prototype nodes are copied lazily before any update.
        if mode == 'prototype':
            staged_i.nodes = dict(gmem_i.nodes)
        copied = set()
        sample_keys = sorted({(r.modality_id, int(p)) for r in report.regions for p in r.samples})
        descriptors = [(key, self.describer.describe(report.supports[key[0]], key[1])) for key in sample_keys]
        for key, descriptor in descriptors:
            node = None
            if mode == 'prototype':
                score = -1.
                for candidate in sorted(staged_i.nodes.values(), key=lambda n: n.node_id):
                    if candidate.modality_id != descriptor.modality_id:
                        continue
                    similarity = float(SimilarityEngine.calculate_similarity(
                        descriptor.values, candidate.prototype.unsqueeze(0), self.cfg, candidate.modality_id,
                        descriptor.sim_mask, candidate.mask.unsqueeze(0)).item())
                    if similarity > score:
                        node, score = candidate, similarity
                if score < descriptor.match_threshold:
                    node = None
                if node is not None:
                    if node.node_id in gmem_i.nodes and node.node_id not in copied:
                        node = copy.deepcopy(node)
                        staged_i.nodes[node.node_id] = node
                        copied.add(node.node_id)
                    node.update_from_descriptor(descriptor)
                    node.count += 1
            if node is None:
                node = staged_i.add_node(descriptor.modality_id, descriptor.values.squeeze(0).clone(),
                        mask=descriptor.sim_mask.squeeze(0).clone(),
                        gate_context=dict(descriptor.gate_context),
                        topology_context=dict(descriptor.topology_context), match_threshold=descriptor.match_threshold)
            report.sample_nodes[key] = node.node_id
        by_id = {r.region_id: r for r in report.regions}
        h, w = report.shape
        def write_edge(sem, ma, p, mb, q, kind):
            source, target = report.sample_nodes[(ma, p)], report.sample_nodes[(mb, q)]
            dx, dy = int(q % w - p % w), int(q // w - p // w)
            span = math.hypot(dx, dy)
            rho = math.log(max(span, self.cfg.once_geometry_epsilon))
            theta = math.atan2(dy, dx) if span else 0.
            edge = staged_ii.add_relation_edge(sem, source, target, rho, theta,
                                               role='peripheral', span=span)
            report.edge_observations.append({
                'semantic_id': sem.node_id, 'edge_id': edge.edge_id, 'kind': kind,
                'source_id': source, 'target_id': target,
                'source_xy': (int(p % w), int(p // w)), 'target_xy': (int(q % w), int(q // w)),
                'rho': rho, 'theta': theta, 'coincident': span == 0,
                'collapsed_instances': source == target and p != q,
            })
        for r in report.regions:
            if not r.samples:
                continue
            sem = staged_ii.add_semantic_node(report.sample_nodes[(r.modality_id, r.anchor)])
            r.semantic_id = sem.node_id
            sem.is_completed = r.completed
            sem.is_closed_contour = r.closed
            sem.max_span = max((math.hypot(p % w - r.anchor % w, p // w - r.anchor // w)
                               for p in r.samples), default=0.)
            for p in r.samples:
                if p != r.anchor:
                    write_edge(sem, r.modality_id, r.anchor, r.modality_id, p, 'region_star')
            report.region_semantic_ids.append(sem.node_id)
        for c in report.contacts:
            a, b = by_id[c.region_a], by_id[c.region_b]
            ka, kb = (a.modality_id, c.source), (b.modality_id, c.target)
            if ka not in report.sample_nodes or kb not in report.sample_nodes:
                report.reject('unwritten_contact', example={'regions': (c.region_a, c.region_b)})
                continue
            if ka == kb:
                c.written = True  # Already joined by one shared instance.
                continue
            sem = staged_ii.add_semantic_node(report.sample_nodes[ka])
            sem.kind = 'contact'
            write_edge(sem, ka[0], ka[1], kb[0], kb[1], c.kind)
            sem.is_completed = True
            sem.max_span = c.distance
            c.semantic_id, c.written = sem.node_id, True
            report.contact_semantic_ids.append(sem.node_id)
        # Every geometry computation and descriptor validation has succeeded.
        gmem_i.nodes.update(staged_i.nodes)
        gmem_i.next_node_id = staged_i.next_node_id
        gmem_ii.semantic_nodes.update(staged_ii.semantic_nodes)
        gmem_ii.next_node_id = staged_ii.next_node_id
        gmem_ii.next_relation_edge_id = staged_ii.next_relation_edge_id
        report.node_ids = sorted(set(report.sample_nodes.values()))
        if mode == 'prototype':
            report.warnings.append('Prototype reuse can collapse instances; inspect self edges and geometric error.')


class OnceGraphBuilder:
    """Finite region construction; no fixation loop and no memory feedback."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.support_builder = FeatureSupportBuilder(cfg)
        self.affinity_builder = LocalAffinityBuilder(cfg)
        self.region_builder = ContinuousRegionAssembler(cfg)
        self.contact_builder = RegionContactBuilder(cfg)
        self.sampler = RegionSampler(cfg)
        self.writer = GraphWriteAdapter(cfg)
        self.last_report = None
        self._validate_config()

    def _validate_config(self):
        for name in ('once_ell_parallel', 'once_ell_perp', 'once_surface_sigma', 'once_geometry_epsilon',
                     'once_curve_spacing', 'once_curve_error', 'once_surface_cover_radius',
                     'once_boundary_spacing', 'once_contact_spacing'):
            if not math.isfinite(float(getattr(self.cfg, name))) or float(getattr(self.cfg, name)) <= 0:
                raise ValueError(name + ' must be positive.')
        for name in ('once_seed_nms_radius', 'once_region_budget', 'once_samples_per_region',
                     'once_contact_radius', 'once_contacts_per_pair', 'once_contact_budget'):
            value = getattr(self.cfg, name)
            if int(value) != value or value < 0:
                raise ValueError(name + ' must be a nonnegative integer.')
        if self.cfg.once_samples_per_region < 1:
            raise ValueError('At least one sample slot per region is required.')
        if self.cfg.once_node_reuse_mode not in {'sample_instance', 'prototype'}:
            raise ValueError('Unknown once_node_reuse_mode.')
        for name in ('once_min_support', 'once_sample_budget'):
            values = getattr(self.cfg, name)
            for value in values.values() if isinstance(values, dict) else [values]:
                if not math.isfinite(float(value)) or int(value) != value or value < 0:
                    raise ValueError(name + ' must contain nonnegative integers.')

    @torch.no_grad()
    def build_once(self, features, gmem_i, gmem_ii, valid_mask=None,
                   H_orig=None, W_orig=None, build_id=1):
        self._validate_config()
        start = time.perf_counter()
        inputs = prepare_feature_subspaces(features, self.cfg)
        h, w = next(iter(inputs.values())).shape[-2:] if inputs else (0, 0)
        if valid_mask is not None:
            raw_mask = valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.asarray(valid_mask)
            if raw_mask.shape == (1, 1, h, w):
                raw_mask = raw_mask[0, 0]
            elif raw_mask.shape == (1, h, w):
                raw_mask = raw_mask[0]
            if raw_mask.shape != (h, w):
                raise ValueError('valid_mask must have spatial shape [H,W].')
            valid = np.isfinite(raw_mask) & (raw_mask > 0)
        else:
            valid = np.ones((h, w), dtype=bool)
            if H_orig is not None or W_orig is not None:
                if H_orig is None or W_orig is None or H_orig <= 0 or W_orig <= 0:
                    raise ValueError('Both original dimensions must be positive.')
                # Current Retina crops at floor(original/2), without resizing.
                y0 = max(0, h // 2 - int(H_orig) // 2)
                x0 = max(0, w // 2 - int(W_orig) // 2)
                valid[:] = False
                valid[y0:min(h, y0 + int(H_orig)), x0:min(w, x0 + int(W_orig))] = True
        config = {name: copy.deepcopy(getattr(self.cfg, name)) for name in dir(self.cfg)
                  if name.startswith('once_')}
        config['feature_specs'] = copy.deepcopy(self.cfg.feature_specs)
        config['modality_weights'] = copy.deepcopy(self.cfg.modality_weights)
        report = BuildReport(build_id, (h, w), valid, config=config)
        if not inputs:
            report.warnings.append('Empty feature input; no memory was written.')
            report.timings['total'] = time.perf_counter() - start
            self.last_report = report
            return report
        t = time.perf_counter()
        report.supports = self.support_builder.build(inputs, valid, report)
        report.timings['supports'] = time.perf_counter() - t
        barrier = np.zeros((h, w), dtype=np.float32)
        if 0 in report.supports:
            edge = report.supports[0]
            barrier = np.where(valid, edge.values[0, 0].abs().cpu().numpy(), 0)
        t = time.perf_counter()
        support_items = list(report.supports.items())
        for index, (mid, support) in enumerate(support_items):
            self.affinity_builder.build(support, barrier, report)
            # Reserve a share for remaining modalities so dense edge noise does
            # not consume every region slot before the RGB pass.
            slots = max(0, self.cfg.once_region_budget - len(report.regions))
            quota = math.ceil(slots / (len(support_items) - index))
            regions, labels = self.region_builder.assemble(support, valid, len(report.regions), report, quota)
            report.regions.extend(regions)
            report.labels[mid] = labels
        report.timings['regions'] = time.perf_counter() - t
        t = time.perf_counter()
        report.contacts = self.contact_builder.build(report.regions, report.labels, barrier, report)
        report.timings['contacts'] = time.perf_counter() - t
        t = time.perf_counter()
        self.sampler.sample(report.regions, report.contacts, report.supports, report)
        report.timings['sampling'] = time.perf_counter() - t
        t = time.perf_counter()
        self.writer.write(report, gmem_i, gmem_ii)
        report.timings['writing'] = time.perf_counter() - t
        report.timings['total'] = time.perf_counter() - start
        self.last_report = report
        return report


# Compatibility names: the new optimizer is a finite graph builder.
class InterestOptimizer(OnceGraphBuilder):
    pass


class Controller:
    """run_step() is one entire observation, not a saccade."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.optimizer = InterestOptimizer(cfg)
        self.current_step = 0
        self.last_report = None
        self.debug_optimizer = {}
        self.current_interest_map = None
        self.matrix1 = self.matrix2 = self.matrix3 = self.matrix4 = None

    @torch.no_grad()
    def run_step(self, X_subspaces, gmem_i, gmem_ii, gmem_iii=None, gpos=None,
                 valid_mask=None, H_orig=None, W_orig=None, source_id=None, episode_id=None,
                 learn=True):
        """One observation. With GmemIII, query first and consolidate long-term memory.

        learn=False retains the raw BuildReport interface for segmentation diagnostics.
        LearningResult keeps observation pools separate from persistent template ids.
        """
        if learn and gmem_iii is not None:
            result = GraphConsolidationOptimizer(self.cfg).learn(
                X_subspaces, gmem_i, gmem_ii, gmem_iii, valid_mask, source_id, episode_id,
                H_orig, W_orig)
            report = result.observation
        else:
            result = self.optimizer.build_once(X_subspaces, gmem_i, gmem_ii, valid_mask,
                                               H_orig, W_orig, self.current_step + 1)
            report = result
        self.current_step += 1
        self.last_report = result
        self.debug_optimizer = result.summary()
        self.matrix1 = {m: s.quality for m, s in report.supports.items()}
        self.matrix2 = report.labels
        self.matrix3 = {r.region_id: r.samples for r in report.regions}
        self.matrix4 = report.edge_observations
        self.current_interest_map = self.matrix1
        return result


class MultilevelCoordinator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gmem_i, self.gmem_ii, self.gmem_iii = GmemoryI(), GmemoryII(cfg), GmemoryIII(cfg)
        self.gpos, self.controller = Gposition(cfg), Controller(cfg)
        self.last_report = None

    def handle_new_view(self, features, H_orig=None, W_orig=None, valid_mask=None,
                        source_id=None, episode_id=None, learn=True):
        self.last_report = self.controller.run_step(
            features, self.gmem_i, self.gmem_ii, self.gmem_iii, self.gpos,
            valid_mask=valid_mask, H_orig=H_orig, W_orig=W_orig,
            source_id=source_id, episode_id=episode_id, learn=learn)
        return self.last_report

    def learn_view(self, features, source_id=None, episode_id=None, valid_mask=None):
        return self.handle_new_view(features, source_id=source_id, episode_id=episode_id,
                                    valid_mask=valid_mask, learn=True)

    def create_entity(self, components):
        """Explicit retrieval fixture: [(semantic_id, dx, dy), ...], not learned evidence."""
        values = [(int(sid), float(dx), float(dy)) for sid, dx, dy in components]
        if len(values) < 2 or not all(math.isfinite(x) and math.isfinite(y) for _, x, y in values):
            raise ValueError('An entity fixture needs at least two finite component placements.')
        for sid, _, _ in values:
            if sid not in self.gmem_ii.semantic_nodes or self.gmem_ii.semantic_nodes[sid].kind != 'region':
                raise ValueError('Entity members must reference existing region hypernodes.')
        entity = self.gmem_iii.add_entity_node()
        _, ox, oy = values[0]
        for sid, x, y in values:
            member = self.gmem_iii.add_component(entity, sid, x - ox, y - oy)
            self.gmem_ii.parent_entities[sid].add(entity.node_id)
            if member['slot_id'] != entity.root_slot:
                entity.relation_edges[(entity.root_slot, member['slot_id'])] = {
                    'delta': (x - ox, y - oy), 'evidence': EvidenceStats()}
        return entity

    @torch.no_grad()
    def query_hierarchy(self, features, valid_mask=None, exact=False):
        return HierarchyRetriever(self.cfg).query(features, self.gmem_i, self.gmem_ii,
                                                  self.gmem_iii, valid_mask, exact)

    def state_dict(self):
        """Versioned pools include incidence, role geometry and evidence provenance."""
        return copy.deepcopy({'schema_version': 1, 'gmem_i': self.gmem_i,
                              'gmem_ii': self.gmem_ii, 'gmem_iii': self.gmem_iii,
                              'feature_contract': self.cfg.feature_specs,
                              'geometry_mode': 'translation'})

    def load_state_dict(self, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported memory schema; old raw pools need explicit migration.')
        if state.get('feature_contract') != self.cfg.feature_specs:
            raise ValueError('Stored feature contract differs from the current configuration.')
        staged = copy.deepcopy(state)
        self.gmem_i, self.gmem_ii, self.gmem_iii = staged['gmem_i'], staged['gmem_ii'], staged['gmem_iii']
        self.last_report = None

    @torch.no_grad()
    def query_l1(self, features, node_ids=None, valid_mask=None):
        """Read-only projection. Select node ids to keep diagnostic memory bounded."""
        inputs = prepare_feature_subspaces(features, self.cfg)
        finite = {mid: torch.isfinite(value).all(dim=1, keepdim=True) for mid, value in inputs.items()}
        inputs = {mid: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.) for mid, value in inputs.items()}
        ids = sorted(self.gmem_i.nodes) if node_ids is None else sorted(set(node_ids))
        maps = self.gpos.l1_footprint_projection(inputs, [self.gmem_i.nodes[n] for n in ids])
        for nid in maps:
            maps[nid] = maps[nid] * finite[self.gmem_i.nodes[nid].modality_id]
        if valid_mask is not None:
            for nid, value in maps.items():
                mask = torch.as_tensor(valid_mask, device=value.device, dtype=value.dtype)
                h, w = value.shape[-2:]
                if tuple(mask.shape) not in {(h, w), (1, h, w), (1, 1, h, w)}:
                    raise ValueError('Query mask shape does not match the feature maps.')
                mask = torch.isfinite(mask) & (mask > 0)
                maps[nid] = value * mask.reshape(1, 1, h, w)
        return maps


def check_written_geometry(report, gmem_ii):
    """Diagnostic against write-time coordinates; never use this as retrieval."""
    rows = []
    for obs in report.edge_observations:
        edge = gmem_ii.semantic_nodes[obs['semantic_id']].relation_edges[obs['edge_id']]
        dx = obs['target_xy'][0] - obs['source_xy'][0]
        dy = obs['target_xy'][1] - obs['source_xy'][1]
        decoded = (math.exp(edge.rho_offset) * math.cos(edge.theta_offset),
                   math.exp(edge.rho_offset) * math.sin(edge.theta_offset))
        rows.append({**obs, 'decoded_delta': decoded,
                     'position_error': math.hypot(decoded[0] - dx, decoded[1] - dy),
                     'merged_count': edge.count})
    return rows


@torch.no_grad()
def reference_star_response(S_maps, semantic_node, tolerance=1):
    """Read-only fixed-pose baseline, NOT GposII. Uses only maps and stored edges.

    Does not enforce one-to-one assignment of repeated instances. Out-of-view
    samples are zero. No report masks or ground-truth coordinates are accepted.
    """
    anchor = S_maps.get(semantic_node.anchor_id)
    if anchor is None:
        return None
    if int(tolerance) != tolerance or tolerance < 0:
        raise ValueError('tolerance must be a nonnegative integer.')
    h, w = anchor.shape[-2:]
    if min(h, w) < 2:
        raise ValueError('Reference sampling requires spatial dimensions >= 2.')
    yy, xx = torch.meshgrid(torch.arange(h, device=anchor.device, dtype=anchor.dtype),
                            torch.arange(w, device=anchor.device, dtype=anchor.dtype), indexing='ij')
    logs = []
    for edge in semantic_node.iter_relation_edges():
        if edge.src_id != semantic_node.anchor_id:
            raise ValueError('Reference baseline only accepts pure stars.')
        value = S_maps.get(edge.dst_id)
        if value is None:
            value = torch.zeros_like(anchor)
        if tolerance:
            value = F.max_pool2d(value, 2 * tolerance + 1, stride=1, padding=tolerance)
        radius = math.exp(edge.rho_offset)
        gx = (xx + radius * math.cos(edge.theta_offset)) * 2 / (w - 1) - 1
        gy = (yy + radius * math.sin(edge.theta_offset)) * 2 / (h - 1) - 1
        grid = torch.stack((gx, gy), dim=-1)[None]
        sampled = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        logs.append(sampled.clamp_min(1e-8).log())
    return anchor if not logs else anchor * torch.exp(torch.stack(logs).mean(dim=0))


# ---------------------------------------------------------------------------
# Persistent evidence and query-only hypergraph views.
# A slot identifies an occurrence; equal feature vectors do not merge slots.
# ---------------------------------------------------------------------------
@dataclass
class EvidenceStats:
    raw_hits: int = 0
    support: float = 0.0
    opportunities: float = 0.0
    episodes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mean: Optional[Tuple[float, float]] = None
    m2: Tuple[float, float] = (0.0, 0.0)
    geometry_weight: float = 0.0

    def observe(self, episode, success, opportunity=True, delta=None):
        if not opportunity:
            return False
        success = float(np.clip(success, 0., 1.))
        self.raw_hits += int(success > 0)
        key = str(episode)
        old_op, old_hit = self.episodes.get(key, (0., 0.))
        # Correlated views of one episode have at most one opportunity/support.
        new_hit = max(old_hit, success)
        self.opportunities += 1. - old_op
        self.support += new_hit - old_hit
        self.episodes[key] = (1., new_hit)
        added = new_hit - old_hit
        if delta is not None and added > 0:
            value = np.asarray(delta, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError('Geometry evidence must be a finite 2D displacement.')
            previous = value if self.mean is None else np.asarray(self.mean)
            total = self.geometry_weight + added
            updated = previous + added / total * (value - previous)
            self.m2 = tuple(np.asarray(self.m2) + added * (value - previous) * (value - updated))
            self.mean = tuple(updated)
            self.geometry_weight = total
        return added > 0

    def reliability(self, prior=1.):
        return (self.support + prior) / (self.opportunities + 2 * prior)

    def weight(self, cfg):
        mature = 1. - math.exp(-self.support / cfg.graph_maturity_support)
        return max(.05, self.reliability(cfg.graph_evidence_prior) * mature)

    def variance(self):
        return tuple(v / max(self.geometry_weight, 1.) for v in self.m2)

    def summary(self):
        return {'raw_hits': self.raw_hits, 'effective_support': self.support,
                'opportunities': self.opportunities, 'reliability': self.reliability(),
                'mean': self.mean, 'variance': self.variance()}

    def merge_equivalent(self, other):
        """Union provenance for geometry-identical records; never sum episode credit."""
        self.raw_hits += other.raw_hits
        for episode, (op, hit) in other.episodes.items():
            a, b = self.episodes.get(episode, (0., 0.))
            self.episodes[episode] = (max(a, op), max(b, hit))
        self.opportunities = sum(v[0] for v in self.episodes.values())
        self.support = sum(v[1] for v in self.episodes.values())
        # No per-episode geometry samples are retained: use the more conservative
        # variance envelope instead of pretending duplicate moments are independent.
        variance = np.maximum(self.variance(), other.variance())
        self.geometry_weight = max(self.geometry_weight, other.geometry_weight)
        self.m2 = tuple(variance * self.geometry_weight)
        if self.mean is None:
            self.mean = other.mean


@dataclass
class StructureView:
    template_id: int
    level: int
    slots: List[dict]
    constraints: List[dict]
    version: int = 0


@dataclass
class StructureMatch:
    template_id: int
    level: int
    point: Tuple[float, float]
    score: float
    evaluable_coverage: float
    matched_coverage: float
    geometry_error: float
    assignments: Dict[Any, dict]
    member_positions: Dict[Any, Tuple[float, float]]
    missing_evidence: List[Any]
    version: int
    accepted: bool
    role_bits: int = 0
    evaluated_bits: int = 0

    def summary(self):
        return {'template_id': self.template_id, 'level': self.level,
                'point': self.point, 'score': self.score,
                'evaluable_coverage': self.evaluable_coverage,
                'matched_coverage': self.matched_coverage,
                'geometry_error': self.geometry_error, 'accepted': self.accepted,
                'matched_roles': self.role_bits.bit_count(),
                'evaluated_roles': self.evaluated_bits.bit_count(),
                'role_bits_hex': hex(self.role_bits), 'evaluated_bits_hex': hex(self.evaluated_bits),
                'version': self.version}


@dataclass
class HierarchyQuery:
    regions: List[StructureMatch] = field(default_factory=list)
    entities: List[StructureMatch] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    activation: dict = field(default_factory=dict)

    def summary(self):
        return {'region_instances': len(self.regions), 'entity_instances': len(self.entities),
                **self.diagnostics}


@dataclass
class LearningResult:
    observation: BuildReport
    observation_i: Any
    observation_ii: Any
    prior_query: HierarchyQuery
    region_mapping: dict = field(default_factory=dict)
    entity_ids: List[int] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        return {**self.observation.summary(), **self.diagnostics,
                'matched_regions': len(self.region_mapping),
                'learned_entities': len(self.entity_ids)}


def _edge_delta(edge):
    return np.array([math.exp(edge.rho_offset) * math.cos(edge.theta_offset),
                     math.exp(edge.rho_offset) * math.sin(edge.theta_offset)])


def region_view(sem, cfg):
    """Resolve a connected graph in a fixed gauge; retain all cycle factors."""
    adjacency = defaultdict(list)
    for e in sem.relation_edges.values():
        d = _edge_delta(e)
        if e.src_id == e.dst_id:
            if np.linalg.norm(d) > cfg.once_geometry_epsilon * 2:
                raise ValueError('Collapsed occurrences require explicit slots; use sample_instance mode.')
            continue
        adjacency[e.src_id].append((e.dst_id, d))
        adjacency[e.dst_id].append((e.src_id, -d))
    positions = {sem.anchor_id: np.zeros(2)}
    queue = deque([sem.anchor_id])
    while queue:
        source = queue.popleft()
        for target, delta in adjacency[source]:
            predicted = positions[source] + delta
            if target not in positions:
                positions[target] = predicted
                queue.append(target)
            elif np.linalg.norm(positions[target] - predicted) > cfg.graph_cycle_tolerance:
                raise ValueError('Inconsistent geometry or alternative layout: split relation versions first.')
    if set(sem.related_node_ids()) - set(positions):
        raise ValueError('Disconnected structure has no common coordinate frame.')
    weights = defaultdict(list)
    constraints = []
    for e in sem.relation_edges.values():
        weights[e.dst_id].append(e.evidence.weight(cfg))
        constraints.append({'source': e.src_id, 'target': e.dst_id,
                            'delta': tuple(_edge_delta(e)), 'edge_id': e.edge_id})
    slots = [{'key': nid, 'node_id': nid, 'xy': tuple(xy), 'group': 0,
              'weight': max(weights.get(nid, [1.])), 'group_weight': 1.,
              'anchor': nid == sem.anchor_id}
             for nid, xy in sorted(positions.items())]
    return StructureView(sem.node_id, 2, slots, constraints, sem.version)


def entity_view(entity, gmem_ii, cfg):
    """Query-only flattening: roles remain unique, component weights normalised."""
    slots, constraints = [], []
    for sid, member in sorted(entity.component_edges.items()):
        if member.get('scale', 1.) != 1. or member.get('angle', 0.) != 0.:
            raise ValueError('This implementation supports translation only, including appearance.')
        sem = gmem_ii.semantic_nodes[member['sem_id']]
        sub = region_view(sem, cfg)
        offset = np.array([member['dx'], member['dy']])
        for item in sub.slots:
            slots.append({**item, 'key': (sid, item['key']), 'group': sid,
                          'xy': tuple(offset + item['xy']),
                          'member_xy': tuple(offset),
                          'group_weight': member['evidence'].weight(cfg)})
        for e in sub.constraints:
            constraints.append({**e, 'source': (sid, e['source']), 'target': (sid, e['target'])})
    for (a, b), relation in entity.relation_edges.items():
        ma, mb = entity.component_edges[a], entity.component_edges[b]
        constraints.append({'source': (a, gmem_ii.semantic_nodes[ma['sem_id']].anchor_id),
                            'target': (b, gmem_ii.semantic_nodes[mb['sem_id']].anchor_id),
                            'delta': relation['delta'], 'edge_id': (a, b)})
    return StructureView(entity.node_id, 3, slots, constraints, entity.version)


class FeatureResponseCache:
    """Exact original metric; lazy LRU maps, shared equal descriptors, no peak renormalisation."""
    def __init__(self, inputs, gmem_i, cfg, valid_mask=None):
        self.cfg, self.nodes = cfg, gmem_i.nodes
        self.inputs = prepare_feature_subspaces(inputs, cfg)
        self.shape = next(iter(self.inputs.values())).shape[-2:] if self.inputs else (0, 0)
        h, w = self.shape
        device = next(iter(self.inputs.values())).device if self.inputs else cfg.device
        if valid_mask is None:
            self.valid = torch.ones((1, 1, h, w), dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(valid_mask, device=device)
            if tuple(mask.shape) not in {(h, w), (1, h, w), (1, 1, h, w)}:
                raise ValueError('Query valid_mask must match the input spatial shape.')
            self.valid = (torch.isfinite(mask) & (mask > 0)).reshape(1, 1, h, w)
        self.finite = {m: torch.isfinite(x).all(1, keepdim=True) & self.valid
                       for m, x in self.inputs.items()}
        self.inputs = {m: torch.nan_to_num(x, nan=0., posinf=0., neginf=0.) for m, x in self.inputs.items()}
        self.gates = {m: SimilarityEngine.compute_gate_map(x, cfg, m) for m, x in self.inputs.items()}
        self.cache = OrderedDict()
        self.map_computations = 0
        self.signatures = {}

    def signature(self, nid):
        if nid not in self.signatures:
            n = self.nodes[nid]
            self.signatures[nid] = (n.modality_id, tuple(n.prototype.shape),
                n.prototype.detach().cpu().contiguous().numpy().tobytes(),
                n.mask.detach().cpu().contiguous().numpy().tobytes())
        return self.signatures[nid]

    def evaluable(self, nid):
        node = self.nodes[nid]
        x = self.inputs.get(node.modality_id)
        if x is None or self.cfg.modality_weights.get(node.modality_id, 1.) <= 0:
            return None
        # Missing required metric channels are unknown, not a zero-valued match.
        spec = self.cfg.feature_specs.get(node.modality_id, {})
        common = min(x.shape[1], node.prototype.shape[0])
        groups = spec.get('metric_groups', {})
        if groups and not any(sum(float(node.mask[c].item()) > 0 for c in
                              SimilarityEngine._group_channels(g.get('channels', 'all'), common))
                              >= int(g.get('min_valid_channels', 1)) for g in groups.values()):
            return None
        return self.finite[node.modality_id]

    def get(self, nid):
        if self.evaluable(nid) is None:
            return None
        key = self.signature(nid)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        n = self.nodes[nid]
        x = self.inputs[n.modality_id]
        response = SimilarityEngine.calculate_similarity(x, n.prototype.to(x.device).view(1, -1, 1, 1),
                    self.cfg, n.modality_id, mask_W=n.mask.to(x.device).view(1, -1))
        response = response * self.gates[n.modality_id] * self.finite[n.modality_id]
        self.cache[key] = response.clamp(0., 1.)
        self.map_computations += 1
        while len(self.cache) > self.cfg.graph_response_cache:
            self.cache.popitem(last=False)
        return response


class SpatialStructureMatcher:
    """Translation search with common assignment, coverage and cycle checks."""
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _sample(value, points):
        h, w = value.shape[-2:]
        p = torch.as_tensor(points, device=value.device, dtype=value.dtype).reshape(1, -1, 1, 2)
        p = p.clone()
        inside = (torch.isfinite(p).all(-1) & (p[..., 0] >= 0) & (p[..., 0] <= w - 1)
                  & (p[..., 1] >= 0) & (p[..., 1] <= h - 1)).reshape(-1)
        p[..., 0] = 2 * p[..., 0] / max(w - 1, 1) - 1 if w > 1 else 0
        p[..., 1] = 2 * p[..., 1] / max(h - 1, 1) - 1 if h > 1 else 0
        return F.grid_sample(value.float(), p.float(), align_corners=True,
                             padding_mode='zeros').reshape(-1) * inside

    def evaluate(self, view, provider, centers):
        if not view.slots or not len(centers):
            return []
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        radius = self.cfg.graph_geometry_radius
        shifts = np.array([(x, y) for y in range(-radius, radius + 1)
                           for x in range(-radius, radius + 1)], dtype=float)
        penalty = np.exp(-np.sum(shifts ** 2, axis=1) / (2 * self.cfg.graph_geometry_sigma ** 2))
        groups = defaultdict(list)
        for item in view.slots:
            groups[item['group']].append(item)
        group_weights = {g: items[0]['group_weight'] for g, items in groups.items()}
        total_group = sum(group_weights.values())
        slot_weights = {s['key']: group_weights[g] / total_group * s['weight'] /
                       sum(i['weight'] for i in items) for g, items in groups.items() for s in items}
        shared = Counter((s['node_id'], tuple(round(v, 6) for v in s['xy'])) for s in view.slots)
        for item in view.slots:
            slot_weights[item['key']] /= shared[(item['node_id'], tuple(round(v, 6) for v in item['xy']))]
        norm = sum(slot_weights.values())
        slot_weights = {key: value / norm for key, value in slot_weights.items()}
        all_results = []
        for start in range(0, len(centers), self.cfg.graph_query_chunk):
            batch = centers[start:start + self.cfg.graph_query_chunk]
            data = {}
            for item in view.slots:
                nid = item['node_id']
                valid = provider.evaluable(nid)
                response = provider.get(nid) if valid is not None else None
                expected = batch + item['xy']
                if response is None:
                    data[item['key']] = (np.zeros(len(batch)), np.zeros(len(batch), bool), expected)
                    continue
                h, w = response.shape[-2:]
                # Evaluable at the predicted point; response absence is negative evidence.
                known = ((expected[:, 0] >= 0) & (expected[:, 0] <= w - 1) &
                         (expected[:, 1] >= 0) & (expected[:, 1] <= h - 1))
                known &= self._sample(valid.float(), expected).cpu().numpy() > .5
                points = expected[:, None, :] + shifts[None, :, :]
                values = self._sample(response, points.reshape(-1, 2)).reshape(len(batch), -1)
                weighted = values * torch.as_tensor(penalty, device=values.device, dtype=values.dtype)
                best, indices = weighted.max(dim=1)
                chosen = expected + shifts[indices.cpu().numpy()]
                data[item['key']] = (best.cpu().numpy(), known, chosen)
            for k, center in enumerate(batch):
                assignments, missing, member_positions = {}, [], {}
                group_votes = defaultdict(list)
                log_score = evaluable = matched = 0.
                bits = evaluated_bits = 0
                for bit, item in enumerate(view.slots):
                    key = item['key']
                    scores, known, chosen = data[key]
                    weight = slot_weights[key]
                    if not known[k]:
                        missing.append(key)
                        continue
                    evaluated_bits |= 1 << bit
                    evaluable += weight
                    value = float(scores[k])
                    log_score += weight * math.log(max(value, 1e-8))
                    hit = value >= self.cfg.graph_recall_threshold
                    if hit:
                        matched += weight
                        bits |= 1 << bit
                        inferred = chosen[k] - item['xy'] + np.asarray(item.get('member_xy', (0., 0.)))
                        group_votes[item['group']].append((value, inferred))
                    assignments[key] = {'point': tuple(chosen[k]), 'score': value,
                                        'supported': hit, 'node_id': item['node_id']}
                    if item['anchor'] and hit:
                        member_positions[item['group']] = tuple(chosen[k])
                for group, votes in group_votes.items():
                    if group not in member_positions:
                        weights = np.asarray([v[0] for v in votes])
                        member_positions[group] = tuple(np.average([v[1] for v in votes], axis=0, weights=weights))
                geometry = []
                for e in view.constraints:
                    a, b = assignments.get(e['source']), assignments.get(e['target'])
                    if a and b and a['supported'] and b['supported']:
                        residual = np.asarray(b['point']) - a['point'] - e['delta']
                        geometry.append(float(np.linalg.norm(residual)))
                error = max(geometry, default=0.)
                # Different predicted roles may not collapse to exactly the same evidence point.
                occupied, conflict = {}, False
                for item in view.slots:
                    a = assignments.get(item['key'])
                    if not a or not a['supported']:
                        continue
                    evidence_key = (provider.nodes[item['node_id']].modality_id,
                                    tuple(round(x, 4) for x in a['point']))
                    if evidence_key in occupied and math.dist(occupied[evidence_key], item['xy']) > .5:
                        conflict = True
                    occupied[evidence_key] = item['xy']
                score = math.exp(log_score / evaluable) if evaluable > 0 else 0.
                coverage = matched / evaluable if evaluable else 0.
                accepted = (score >= self.cfg.graph_verify_threshold and
                            evaluable >= self.cfg.graph_evaluable_min and
                            coverage >= self.cfg.graph_coverage_min and not conflict and
                            error <= 2 * radius + self.cfg.graph_cycle_tolerance)
                all_results.append(StructureMatch(view.template_id, view.level, tuple(center), score,
                    evaluable, coverage, error, assignments, member_positions, missing, view.version,
                    accepted, bits, evaluated_bits))
        return all_results

    def nms(self, matches, accepted_only=True, limit=None):
        result = []
        for match in sorted(matches, key=lambda m: (-m.score, -m.matched_coverage, m.point)):
            if accepted_only and not match.accepted:
                continue
            if any(math.dist(match.point, old.point) < self.cfg.graph_nms_radius for old in result):
                continue
            result.append(match)
            if limit and len(result) >= limit:
                break
        return result


class HypergraphIndex:
    """Shared exact descriptor classes and reverse incidence; never replaces geometry."""
    def __init__(self, gi, gii, giii, cfg, provider):
        self.classes, self.node_class, self.parents = {}, {}, defaultdict(list)
        self.region_bits, self.entity_bits = {}, {}
        self.entity_parents = defaultdict(list)
        self.views, self.invalid = {}, {}
        signature_ids = {}
        for nid in sorted(gi.nodes):
            signature = provider.signature(nid)
            if signature not in signature_ids:
                signature_ids[signature] = len(signature_ids)
                self.classes[signature_ids[signature]] = nid
            self.node_class[nid] = signature_ids[signature]
        for sid, sem in sorted(gii.semantic_nodes.items()):
            if sem.kind != 'region':
                continue
            try:
                view = region_view(sem, cfg)
            except ValueError as exc:
                self.invalid[sid] = str(exc)
                continue
            self.views[sid] = view
            bits = 0
            for slot in view.slots:
                cls = self.node_class[slot['node_id']]
                bits |= 1 << cls
                self.parents[cls].append((sid, slot['key'], slot['xy']))
            self.region_bits[sid] = bits
        for eid, entity in sorted(giii.entity_nodes.items()):
            bits = 0
            for role, member in entity.component_edges.items():
                sid = member['sem_id']
                bits |= 1 << sid
                self.entity_parents[sid].append((eid, role, (member['dx'], member['dy'])))
            self.entity_bits[eid] = bits

    def feature_events(self, provider, cfg):
        events, evaluated, truncated = [], 0, 0
        stride = cfg.graph_query_stride
        for cls, nid in self.classes.items():
            valid = provider.evaluable(nid)
            if valid is None:
                continue
            evaluated |= 1 << cls
            node = provider.nodes[nid]
            x = provider.inputs[node.modality_id][..., ::stride, ::stride]
            # Same metric at sampled points, not an unrelated ANN distance.
            values = SimilarityEngine.calculate_similarity(x,
                node.prototype.to(x.device).view(1, -1, 1, 1), cfg, node.modality_id,
                mask_W=node.mask.to(x.device).view(1, -1))
            values *= provider.gates[node.modality_id][..., ::stride, ::stride]
            values *= valid[..., ::stride, ::stride]
            ys, xs = torch.where(values[0, 0] >= cfg.graph_recall_threshold)
            candidates = sorted([(float(values[0, 0, y, x]), int(x) * stride, int(y) * stride)
                                 for y, x in zip(ys, xs)], key=lambda t: (-t[0], t[2], t[1]))
            if cfg.graph_events_per_class and len(candidates) > cfg.graph_events_per_class:
                truncated += len(candidates) - cfg.graph_events_per_class
                candidates = candidates[:cfg.graph_events_per_class]
            events.extend((cls, (x, y), score) for score, x, y in candidates)
        # 'evaluated' means modality/descriptor evaluated at sampled sites, not proven absent globally.
        return events, evaluated, truncated

    @staticmethod
    def _votes(events, parents, cfg, requirements=None):
        buckets, visits = {}, 0
        for key, point, score in events:
            for parent, role, offset in parents.get(key, []):
                visits += 1
                center = np.asarray(point) - offset
                bucket = (parent, round(center[0] / cfg.graph_pose_bin), round(center[1] / cfg.graph_pose_bin))
                entry = buckets.setdefault(bucket, {'roles': {}, 'center': tuple(center), 'hits': 0})
                entry['roles'][role] = max(score, entry['roles'].get(role, 0.))
                entry['hits'] |= 1 << key
        by_parent = defaultdict(list)
        for (parent, _, _), entry in buckets.items():
            required = requirements.get(parent, 0) if requirements is not None else entry['hits']
            coverage = (required & entry['hits']).bit_count() / max(1, required.bit_count())
            if coverage < cfg.graph_bit_coverage_min:
                continue
            rank = (coverage, len(entry['roles']), sum(entry['roles'].values()))
            by_parent[parent].append((rank, entry['center']))
        return by_parent, visits

    def region_candidates(self, events, cfg):
        ranked, visits = self._votes(events, self.parents, cfg, self.region_bits)
        return self._limit(ranked, cfg), visits

    @staticmethod
    def _limit(ranked, cfg):
        rows = []
        for parent, values in ranked.items():
            for rank, center in sorted(values, reverse=True)[:cfg.graph_poses_per_node]:
                rows.append((rank, parent, center))
        rows.sort(reverse=True)
        truncated = max(0, len(rows) - cfg.graph_candidate_budget) if cfg.graph_candidate_budget else 0
        if cfg.graph_candidate_budget:
            rows = rows[:cfg.graph_candidate_budget]
        result = defaultdict(list)
        for _, parent, center in rows:
            result[parent].append(center)
        return result, truncated


class HierarchyRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.matcher = SpatialStructureMatcher(cfg)
        self.validate()

    def validate(self):
        for name in ('graph_geometry_sigma', 'graph_maturity_support', 'graph_pose_bin',
                     'graph_member_radius', 'graph_nms_radius', 'graph_evidence_prior',
                     'graph_geometry_update_limit', 'graph_stable_support'):
            value = float(getattr(self.cfg, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive.')
        for name in ('graph_query_stride', 'graph_poses_per_node', 'graph_response_cache',
                     'graph_query_chunk', 'graph_max_members', 'graph_min_members'):
            value = getattr(self.cfg, name)
            if int(value) != value or value < 1:
                raise ValueError(name + ' must be a positive integer.')
        for name in ('graph_geometry_radius', 'graph_candidate_budget', 'graph_events_per_class',
                     'graph_proposal_budget', 'graph_candidate_expiry', 'graph_candidate_memory_budget'):
            value = getattr(self.cfg, name)
            if int(value) != value or value < 0:
                raise ValueError(name + ' must be a nonnegative integer.')
        for name in ('graph_recall_threshold', 'graph_bit_coverage_min', 'graph_verify_threshold', 'graph_learn_threshold',
                     'graph_evaluable_min', 'graph_coverage_min', 'graph_match_margin',
                     'graph_member_remove_reliability'):
            if not 0 <= float(getattr(self.cfg, name)) <= 1:
                raise ValueError(name + ' must be in [0,1].')
        if not self.cfg.graph_recall_threshold <= self.cfg.graph_verify_threshold <= self.cfg.graph_learn_threshold:
            raise ValueError('Require recall <= verification <= learning thresholds.')
        if self.cfg.graph_min_members > self.cfg.graph_max_members:
            raise ValueError('Minimum members must not exceed maximum members.')

    @torch.no_grad()
    def query(self, features, gi, gii, giii, valid_mask=None, exact=False):
        start = time.perf_counter()
        provider = FeatureResponseCache(features, gi, self.cfg, valid_mask)
        result = HierarchyQuery()
        if not provider.inputs or not gi.nodes:
            result.diagnostics = {'search_complete': True, 'mode': 'empty', 'elapsed': time.perf_counter() - start}
            return result
        index = HypergraphIndex(gi, gii, giii, self.cfg, provider)
        h, w = provider.shape
        events, evaluated, event_drop = ([], 0, 0) if exact else index.feature_events(provider, self.cfg)
        (candidates, dropped), visits = index.region_candidates(events, self.cfg)
        dense = [(x, y) for y in range(h) for x in range(w)] if exact else None
        if exact:
            candidates = {sid: dense for sid in index.views}
        weak_regions, refinements = [], 0
        for sid, centers in candidates.items():
            matches = self.matcher.evaluate(index.views[sid], provider, centers)
            refinements += len(matches)
            weak = self.matcher.nms([m for m in matches if m.score >= self.cfg.graph_recall_threshold],
                                    accepted_only=False, limit=None if exact else self.cfg.graph_poses_per_node)
            weak_regions.extend(weak)
            result.regions.extend(m for m in weak if m.accepted)
        parent_events = [(m.template_id, m.point, m.score) for m in weak_regions]
        entity_ranked, entity_visits = index._votes(parent_events, index.entity_parents, self.cfg, index.entity_bits)
        entity_candidates, entity_drop = index._limit(entity_ranked, self.cfg)
        if exact:
            entity_candidates = {eid: dense for eid in giii.entity_nodes}
        invalid_entities = {}
        for eid, centers in entity_candidates.items():
            entity = giii.entity_nodes[eid]
            try:
                view = entity_view(entity, gii, self.cfg)
            except (ValueError, KeyError) as exc:
                invalid_entities[eid] = str(exc)
                continue
            matches = self.matcher.evaluate(view, provider, centers)
            refinements += len(matches)
            # Leaf evidence is decisive, while root roles must remain spatially distinct.
            for match in matches:
                positions = list(match.member_positions.values())
                if len(positions) < min(self.cfg.graph_min_members, len(entity.component_edges)):
                    match.accepted = False
                if len(set(positions)) != len(positions):
                    # Co-located DIFFERENT modalities are valid; repeated semantic roles are not.
                    seen = set()
                    for role, point in match.member_positions.items():
                        key = (entity.component_edges[role]['sem_id'], point)
                        if key in seen:
                            match.accepted = False
                        seen.add(key)
            result.entities.extend(self.matcher.nms(matches, limit=None if exact else self.cfg.graph_poses_per_node))
        hit_bits = 0
        for cls, _, _ in events:
            hit_bits |= 1 << cls
        result.activation = {eid: max(m.score for m in result.entities if m.template_id == eid)
                             for eid in {m.template_id for m in result.entities}}
        result.diagnostics = {'mode': 'dense_translation' if exact else 'sparse_translation',
            'search_complete': bool(exact and not index.invalid and not invalid_entities),
            'assignment_search_complete': False,
            'classes': len(index.classes), 'feature_events': len(events),
            'event_budget_dropped': event_drop, 'candidate_budget_dropped': dropped + entity_drop,
            'posting_visits': visits + entity_visits, 'geometry_evaluations': refinements,
            'response_maps_computed': provider.map_computations,
            'static_region_bits': index.region_bits, 'static_entity_bits': index.entity_bits,
            'query_hit_bits': hit_bits, 'sampled_evaluated_bits': evaluated,
            'invalid_regions': index.invalid, 'invalid_entities': invalid_entities,
            'elapsed': time.perf_counter() - start,
            'limitations': ['Translation only; sparse search is approximate.',
                           'Deterministic local assignment rejects conflicts; no combinatorial backtracking.']}
        return result


class _StoredMapProvider:
    """Compatibility adapter for callers already holding GposI response maps."""
    def __init__(self, maps, nodes=None, cfg=None, valid_mask=None):
        from types import SimpleNamespace
        self.maps = maps
        self.nodes = nodes if nodes is not None else {nid: SimpleNamespace(modality_id=nid) for nid in maps}
        self.cfg = cfg
        self.shape = next(iter(maps.values())).shape[-2:] if maps else (0, 0)
        self.valid = valid_mask

    def get(self, nid):
        value = self.maps.get(nid)
        if value is None:
            return None
        weight = self.cfg.modality_weights.get(self.nodes[nid].modality_id, 1.) if self.cfg else 1.
        return (value / weight).clamp(0, 1) if weight > 0 else None

    def evaluable(self, nid):
        value = self.get(nid)
        if value is None:
            return None
        if self.valid is None:
            return torch.ones_like(value, dtype=torch.bool)
        mask = torch.as_tensor(self.valid, device=value.device)
        return (torch.isfinite(mask) & (mask > 0)).reshape_as(value)


class GraphConsolidationOptimizer:
    """Bounded overlapping compositions, persistent roles and episode-aware evidence.

    Translation-only, conservative: ambiguous matches do not update or create a
    competing memory. Incompatible layouts remain separate candidate entities.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = HierarchyRetriever(cfg)
        self.builder = OnceGraphBuilder(cfg)
        self.work_counts = Counter()

    @staticmethod
    def source_digest(inputs):
        digest = hashlib.sha256()
        for mid, value in sorted(inputs.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((mid, tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _region_match(self, region, report, gi, gii, provider, prior):
        if not gi.nodes:
            return None, False
        # Exhaustive local translation fallback: a sparse global miss is not novelty.
        h, w = report.shape
        mask = np.zeros((h, w), bool)
        mask.flat[region.pixels] = True
        local = FeatureResponseCache(provider.inputs, gi, self.cfg, mask & report.valid_mask)
        centers = [(int(p % w), int(p // w)) for p in region.pixels]
        choices = []
        unresolved = False
        sparse_centers = defaultdict(list)
        for match in prior.regions:
            x, y = map(lambda v: int(round(v)), match.point)
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                sparse_centers[match.template_id].append(match.point)
        for sid, sem in sorted(gii.semantic_nodes.items()):
            if sem.kind != 'region' or gi.nodes[sem.anchor_id].modality_id != region.modality_id:
                continue
            try:
                view = region_view(sem, self.cfg)
            except ValueError:
                unresolved = True
                continue
            self.work_counts['region_template_pairs'] += 1
            self.work_counts['sparse_centers'] += len(sparse_centers.get(sid, []))
            candidates = self.retriever.matcher.evaluate(view, local, sparse_centers.get(sid, []))
            if not any(m.accepted and m.score >= self.cfg.graph_learn_threshold for m in candidates):
                self.work_counts['dense_fallback_templates'] += 1
                self.work_counts['dense_fallback_centers'] += len(centers)
                self.work_counts['dense_slot_centers'] += len(centers) * len(view.slots)
                candidates = self.retriever.matcher.evaluate(view, local, centers)
            accepted = [m for m in candidates if m.accepted and m.score >= self.cfg.graph_learn_threshold]
            if accepted:
                best = max(accepted, key=lambda m: (m.score, m.matched_coverage, -m.geometry_error))
                # Reverse extent check prevents a tiny same-colour fragment claiming a large region.
                observed_span = max((math.dist((p % w, p // w), best.point) for p in region.samples), default=0.)
                template_span = max((math.hypot(*s['xy']) for s in view.slots), default=0.)
                if abs(observed_span - template_span) <= max(self.cfg.graph_geometry_update_limit,
                                                             .2 * max(observed_span, template_span)):
                    choices.append(best)
        self.work_counts['local_response_maps_computed'] += local.map_computations
        choices.sort(key=lambda m: (-m.score, m.template_id))
        if len(choices) > 1 and choices[0].score - choices[1].score < self.cfg.graph_match_margin:
            return None, True
        return (choices[0], False) if choices else (None, unresolved)

    @staticmethod
    def _copy_region(sem, source_i, gi, gii, node_mapping):
        copied = gii.add_semantic_node(-1)
        copied.kind = 'region'
        for nid in sorted(sem.related_node_ids()):
            if nid not in node_mapping:
                n = copy.deepcopy(source_i.nodes[nid])
                n.node_id = gi.next_node_id
                gi.next_node_id += 1
                gi.nodes[n.node_id] = n
                node_mapping[nid] = n.node_id
        copied.anchor_id = node_mapping[sem.anchor_id]
        copied.child_node_ids = {node_mapping[n] for n in sem.related_node_ids()}
        copied.is_completed = sem.is_completed
        copied.is_closed_contour = sem.is_closed_contour
        copied.max_span = sem.max_span
        for e in sem.relation_edges.values():
            gii.add_relation_edge(copied, node_mapping[e.src_id], node_mapping[e.dst_id],
                e.rho_offset, e.theta_offset, e.lambda_rho, e.gamma_theta, e.role, e.span, e.area)
        return copied

    def _update_region(self, sem, match, episode):
        changed = sem.evidence.observe(episode, True)
        for e in sem.relation_edges.values():
            a = match.assignments.get(e.src_id) if match else None
            b = match.assignments.get(e.dst_id) if match else None
            if match is None:
                delta, success, opportunity = _edge_delta(e), True, True
            else:
                opportunity = a is not None and b is not None
                success = opportunity and a['supported'] and b['supported']
                delta = np.asarray(b['point']) - a['point'] if success else None
                if success and np.linalg.norm(delta - _edge_delta(e)) > self.cfg.graph_geometry_update_limit:
                    success, delta = False, None
            if e.evidence.observe(episode, success, opportunity, delta):
                mean = e.evidence.mean
                if mean is not None:
                    e.rho_offset = math.log(max(math.hypot(*mean), self.cfg.once_geometry_epsilon))
                    e.theta_offset = math.atan2(mean[1], mean[0])
                changed = True
            e.count = e.evidence.raw_hits
            e.confidence = e.evidence.reliability(self.cfg.graph_evidence_prior)
        if changed:
            sem.version += 1
            # Rebuild legacy representative links after updates.
            sem.peripheral_links = {}
            for e in sem.relation_edges.values():
                sem.peripheral_links.setdefault(e.dst_id, e.as_legacy_link())
        return changed

    def _merge_identical_regions(self, gi, gii, giii):
        """Only merge losslessly equivalent attributed hypergraphs, never raw id sets."""
        signatures, remap, slot_remap = {}, {}, {}
        for sid, sem in sorted(gii.semantic_nodes.items()):
            if sem.kind != 'region':
                continue
            try:
                view = region_view(sem, self.cfg)
            except ValueError:
                continue
            descriptors = {}
            for slot in view.slots:
                n = gi.nodes[slot['node_id']]
                descriptors[slot['key']] = (n.modality_id, tuple(n.prototype.shape),
                    n.prototype.detach().cpu().contiguous().numpy().tobytes(),
                    n.mask.detach().cpu().contiguous().numpy().tobytes(), tuple(slot['xy']))
            edge_keys = {(descriptors[e.src_id], descriptors[e.dst_id], e.role,
                          e.lambda_rho, e.gamma_theta): e for e in sem.relation_edges.values()}
            if len(edge_keys) != len(sem.relation_edges):
                continue  # Different edge identities cannot silently collapse.
            signature = (descriptors[sem.anchor_id], tuple(sorted(descriptors.values())),
                         tuple(sorted(edge_keys)))
            if signature not in signatures:
                signatures[signature] = (sid, edge_keys, descriptors)
                continue
            canonical, old_edges, old_descriptors = signatures[signature]
            inverse = {description: nid for nid, description in old_descriptors.items()}
            for nid, description in descriptors.items():
                slot_remap[(sid, nid)] = (canonical, inverse[description])
            target = gii.semantic_nodes[canonical]
            target.evidence.merge_equivalent(sem.evidence)
            for key, edge in edge_keys.items():
                old_edges[key].evidence.merge_equivalent(edge.evidence)
                old_edges[key].count = old_edges[key].evidence.raw_hits
                old_edges[key].confidence = old_edges[key].evidence.reliability()
            target.version += 1
            remap[sid] = canonical
        for entity in giii.entity_nodes.values():
            entity.component_edges_by_semantic = defaultdict(list)
            entity.components = {}
            for role, member in entity.component_edges.items():
                member['sem_id'] = remap.get(member['sem_id'], member['sem_id'])
                entity.component_edges_by_semantic[member['sem_id']].append(role)
                entity.components.setdefault(member['sem_id'], {'dx': member['dx'], 'dy': member['dy']})
            for pending in entity.pending_members.values():
                pending['sem_id'] = remap.get(pending['sem_id'], pending['sem_id'])
        for relation in gii.region_relations.values():
            relation['source'] = remap.get(relation['source'], relation['source'])
            relation['target'] = remap.get(relation['target'], relation['target'])
            if relation['source'] > relation['target']:
                relation['source'], relation['target'] = relation['target'], relation['source']
                relation['delta'] = tuple(-v for v in relation['delta'])
                if relation['evidence'].mean is not None:
                    relation['evidence'].mean = tuple(-v for v in relation['evidence'].mean)
            witnesses = {}
            for witness in relation.get('contact_witnesses', {}).values():
                witness['source'] = slot_remap.get(witness['source'], witness['source'])
                witness['target'] = slot_remap.get(witness['target'], witness['target'])
                if witness['source'][0] > witness['target'][0]:
                    witness['source'], witness['target'] = witness['target'], witness['source']
                    witness['source_local'], witness['target_local'] = witness['target_local'], witness['source_local']
                key = (witness['source'], witness['target'])
                if key in witnesses:
                    witnesses[key]['evidence'].merge_equivalent(witness['evidence'])
                else:
                    witnesses[key] = witness
            relation['contact_witnesses'] = witnesses
        for sid in remap:
            del gii.semantic_nodes[sid]
        gii.parent_entities = defaultdict(set)
        for eid, entity in giii.entity_nodes.items():
            for m in entity.component_edges.values():
                gii.parent_entities[m['sem_id']].add(eid)
        return remap

    def _merge_identical_entities(self, gii, giii):
        signatures, remap = {}, {}
        for eid, entity in sorted(giii.entity_nodes.items()):
            if entity.pending_members:
                continue
            members = sorted(((m['sem_id'], m['dx'], m['dy']), role)
                             for role, m in entity.component_edges.items())
            root = entity.component_edges[entity.root_slot]
            signature = ((root['sem_id'], root['dx'], root['dy']), tuple(key for key, _ in members))
            if signature not in signatures:
                signatures[signature] = (eid, members)
                continue
            old_id, old_members = signatures[signature]
            target = giii.entity_nodes[old_id]
            role_map = {role: old_role for (_, role), (_, old_role) in zip(members, old_members)}
            if {(role_map[a], role_map[b]) for a, b in entity.relation_edges} != set(target.relation_edges):
                continue
            if any(entity.relation_edges[p]['delta'] != target.relation_edges[(role_map[p[0]], role_map[p[1]])]['delta']
                   for p in entity.relation_edges):
                continue
            target.evidence.merge_equivalent(entity.evidence)
            for role, old_role in role_map.items():
                target.component_edges[old_role]['evidence'].merge_equivalent(entity.component_edges[role]['evidence'])
            for (a, b), rel in entity.relation_edges.items():
                target.relation_edges[(role_map[a], role_map[b])]['evidence'].merge_equivalent(rel['evidence'])
            target.version += 1
            target.last_observed = max(target.last_observed, entity.last_observed)
            if entity.status == 'stable':
                target.status = 'stable'
            remap[eid] = old_id
        for eid in remap:
            del giii.entity_nodes[eid]
        for parents in gii.parent_entities.values():
            for eid in list(parents):
                if eid in remap:
                    parents.remove(eid)
                    parents.add(remap[eid])
        return remap

    def _relations_and_proposals(self, report, mapped, positions, gii, episode):
        adjacency = defaultdict(set)
        observed_layouts = defaultdict(list)
        contacts = defaultdict(list)
        for c in report.contacts:
            if c.written:
                contacts[tuple(sorted((c.region_a, c.region_b)))].append(c)
        supported_pairs = {tuple(sorted((c.region_a, c.region_b))) for c in report.contacts if c.written}
        ids = sorted(mapped)
        for ai, a in enumerate(ids):
            for b in ids[ai + 1:]:
                delta = np.asarray(positions[b]) - positions[a]
                distance = np.linalg.norm(delta)
                contact = (a, b) in supported_pairs
                if not contact and distance > self.cfg.graph_member_radius:
                    continue
                sa, sb = mapped[a], mapped[b]
                # Canonical template orientation, preserving the sign of relative position.
                if sa > sb:
                    sa, sb, delta = sb, sa, -delta
                observed_layouts[(sa, sb)].append(tuple(delta))
                variants = [(key, rel) for key, rel in gii.region_relations.items()
                            if rel['source'] == sa and rel['target'] == sb]
                chosen = next((rel for _, rel in variants if math.dist(rel['delta'], delta)
                               <= self.cfg.graph_geometry_update_limit), None)
                if chosen is None:
                    key = max(gii.region_relations, default=-1) + 1
                    chosen = {'source': sa, 'target': sb, 'delta': tuple(delta),
                              'evidence': EvidenceStats(), 'contact': contact}
                    gii.region_relations[key] = chosen
                chosen['contact'] |= contact
                witnesses = chosen.setdefault('contact_witnesses', {})
                for c in contacts.get((a, b), [])[:self.cfg.once_contacts_per_pair]:
                    # Resolve contact endpoints to persistent feature slots. Coordinates
                    # below are relative to member anchors, never query-time ground truth.
                    ra, rb = c.region_a, c.region_b
                    points = [(c.source % report.shape[1], c.source // report.shape[1]),
                              (c.target % report.shape[1], c.target // report.shape[1])]
                    resolved = []
                    for rid, point in zip((ra, rb), points):
                        view = region_view(gii.semantic_nodes[mapped[rid]], self.cfg)
                        local = np.asarray(point) - positions[rid]
                        slot = min(view.slots, key=lambda s: math.dist(s['xy'], local))
                        if math.dist(slot['xy'], local) > self.cfg.graph_geometry_update_limit:
                            break
                        resolved.append((mapped[rid], slot['node_id'], tuple(local)))
                    if len(resolved) != 2:
                        continue
                    if resolved[0][0] > resolved[1][0]:
                        resolved.reverse()
                    key = tuple((item[0], item[1]) for item in resolved)
                    witness = witnesses.setdefault(key, {'source': resolved[0][:2],
                        'target': resolved[1][:2], 'source_local': resolved[0][2],
                        'target_local': resolved[1][2], 'evidence': EvidenceStats()})
                    witness['evidence'].observe(episode, True)
                adjacency[a].add(b)
                adjacency[b].add(a)
        # Check alternatives once per observation, not once per sampled contact pair.
        for relation in gii.region_relations.values():
            layouts = observed_layouts.get((relation['source'], relation['target']), [])
            if not layouts:
                continue  # No jointly observed pair: unknown, not a failed relation.
            best = min(layouts, key=lambda d: math.dist(d, relation['delta']))
            supported = math.dist(best, relation['delta']) <= self.cfg.graph_geometry_update_limit
            relation['evidence'].observe(episode, supported, delta=best if supported else None)
            if relation['evidence'].mean is not None:
                relation['delta'] = relation['evidence'].mean
        proposals, seen = [], set()
        for root in ids:
            group, frontier = [root], set(adjacency[root])
            while frontier and len(group) < self.cfg.graph_max_members:
                nxt = min(frontier, key=lambda rid: (math.dist(positions[root], positions[rid]), rid))
                frontier.remove(nxt)
                if math.dist(positions[root], positions[nxt]) > self.cfg.graph_member_radius:
                    continue
                group.append(nxt)
                frontier.update(adjacency[nxt] - set(group))
                key = tuple(sorted(group))
                if len(key) >= self.cfg.graph_min_members and key not in seen:
                    proposals.append(key)
                    seen.add(key)
                    if self.cfg.graph_proposal_budget and len(proposals) >= self.cfg.graph_proposal_budget:
                        return proposals, True
        return proposals, False

    def _align_entity(self, entity, proposal, mapped, positions):
        members = entity.component_edges
        if not members:
            return None
        choices = []
        centers = {tuple(np.asarray(positions[rid]) - [m['dx'], m['dy']])
                   for rid in proposal for m in members.values() if mapped[rid] == m['sem_id']}
        for candidate in sorted(centers):
            center = np.asarray(candidate)
            assignment, used, errors = {}, set(), []
            # Exact semantic identity after GmemII alignment; occurrences stay distinct.
            for role, member in sorted(members.items()):
                target = center + [member['dx'], member['dy']]
                options = sorted((math.dist(positions[r], target), r) for r in proposal
                                 if r not in used and mapped[r] == member['sem_id'])
                if options and options[0][0] <= self.cfg.graph_geometry_update_limit:
                    error, region = options[0]
                    assignment[role] = region
                    used.add(region)
                    errors.append(error)
            coverage = len(assignment) / len(members)
            reverse = len(assignment) / len(proposal)
            if coverage >= self.cfg.graph_coverage_min and reverse >= self.cfg.graph_coverage_min:
                score = min(coverage, reverse) * math.exp(-max(errors, default=0.) ** 2 /
                                                        (2 * self.cfg.graph_geometry_sigma ** 2))
                choices.append((score, tuple(center), assignment))
        return max(choices, key=lambda x: x[0]) if choices else None

    def _update_entity(self, entity, alignment, proposal, mapped, positions, provider, episode, step):
        score, center, assignment = alignment
        entity.evidence.observe(episode, True)
        entity.last_observed = step
        for role, member in entity.component_edges.items():
            rid = assignment.get(role)
            point = tuple(np.asarray(center) + [member['dx'], member['dy']])
            h, w = provider.shape
            opportunity = (0 <= point[0] < w and 0 <= point[1] < h and
                           bool(provider.valid[0, 0, int(point[1]), int(point[0])]))
            anchor_id = self._working_ii.semantic_nodes[member['sem_id']].anchor_id
            opportunity &= provider.evaluable(anchor_id) is not None
            delta = tuple(np.asarray(positions[rid]) - center) if rid is not None else None
            # Other matched roles are necessary before treating a missing member as a failure.
            opportunity &= len(assignment) >= self.cfg.graph_min_members
            changed = member['evidence'].observe(episode, rid is not None, opportunity, delta)
            if changed and role != entity.root_slot and member['evidence'].mean is not None:
                member['dx'], member['dy'] = member['evidence'].mean
            member['count'] = member['evidence'].raw_hits
        for (a, b), rel in entity.relation_edges.items():
            ra, rb = assignment.get(a), assignment.get(b)
            delta = tuple(np.asarray(positions[rb]) - positions[ra]) if ra is not None and rb is not None else None
            rel['evidence'].observe(episode, delta is not None,
                                    opportunity=ra is not None and rb is not None, delta=delta)
            if rel['evidence'].mean is not None:
                rel['delta'] = rel['evidence'].mean
        # Repeated unexplained members are probationary, not immediate structural edits.
        used = set(assignment.values())
        observed_pending = set()
        for rid in proposal:
            if rid in used:
                continue
            delta = tuple(np.asarray(positions[rid]) - center)
            pending_key = next((key for key, item in entity.pending_members.items()
                                if item['sem_id'] == mapped[rid] and
                                math.dist(item['delta'], delta) <= self.cfg.graph_geometry_update_limit), None)
            if pending_key is None:
                pending_key = max(entity.pending_members, default=-1) + 1
                entity.pending_members[pending_key] = {'sem_id': mapped[rid], 'delta': delta,
                                                       'evidence': EvidenceStats()}
            item = entity.pending_members[pending_key]
            item['evidence'].observe(episode, True, delta=delta)
            item['delta'] = item['evidence'].mean or delta
            observed_pending.add(pending_key)
        for key, item in list(entity.pending_members.items()):
            if key not in observed_pending:
                p = np.asarray(center) + item['delta']
                h, w = provider.shape
                nid = self._working_ii.semantic_nodes[item['sem_id']].anchor_id
                known = (0 <= p[0] < w and 0 <= p[1] < h and
                         bool(provider.valid[0, 0, int(p[1]), int(p[0])]) and
                         provider.evaluable(nid) is not None)
                item['evidence'].observe(episode, False, known)
            if (item['evidence'].support >= self.cfg.graph_stable_support and
                    item['evidence'].reliability() >= self.cfg.graph_coverage_min and
                    len(entity.component_edges) < self.cfg.graph_max_members):
                member = self._working_iii.add_component(entity, item['sem_id'], *item['delta'])
                member['evidence'] = copy.deepcopy(item['evidence'])
                member['count'] = member['evidence'].raw_hits
                self._working_ii.parent_entities[item['sem_id']].add(entity.node_id)
                entity.relation_edges[(entity.root_slot, member['slot_id'])] = {
                    'delta': item['delta'], 'evidence': copy.deepcopy(item['evidence'])}
                del entity.pending_members[key]
        removable = [role for role, member in entity.component_edges.items()
                     if role != entity.root_slot and
                     member['evidence'].opportunities >= self.cfg.graph_stable_support and
                     member['evidence'].reliability() < self.cfg.graph_member_remove_reliability]
        for role in removable:
            if len(entity.component_edges) <= self.cfg.graph_min_members:
                break
            member = entity.component_edges.pop(role)
            entity.component_edges_by_semantic[member['sem_id']].remove(role)
            if not entity.component_edges_by_semantic[member['sem_id']]:
                self._working_ii.parent_entities[member['sem_id']].discard(entity.node_id)
            entity.relation_edges = {pair: rel for pair, rel in entity.relation_edges.items() if role not in pair}
        # The representative compatibility view must follow role changes.
        entity.components = {}
        for member in entity.component_edges.values():
            entity.components.setdefault(member['sem_id'], {'dx': member['dx'], 'dy': member['dy']})
        entity.version += 1
        variances = [max(m['evidence'].variance()) for m in entity.component_edges.values()]
        if entity.evidence.support >= self.cfg.graph_stable_support and max(variances, default=0.) <= self.cfg.graph_stable_variance:
            entity.status = 'stable'

    def _new_entity(self, proposal, mapped, positions, gii, giii, episode, step):
        entity = giii.add_entity_node()
        root = proposal[0]
        root_xy = np.asarray(positions[root])
        for rid in proposal:
            delta = tuple(np.asarray(positions[rid]) - root_xy)
            member = giii.add_component(entity, mapped[rid], *delta)
            member['evidence'].observe(episode, True, delta=delta)
            gii.parent_entities[mapped[rid]].add(entity.node_id)
        roles = sorted(entity.component_edges)
        for role in roles[1:]:
            member = entity.component_edges[role]
            stat = EvidenceStats()
            stat.observe(episode, True, delta=(member['dx'], member['dy']))
            entity.relation_edges[(entity.root_slot, role)] = {
                'delta': (member['dx'], member['dy']), 'evidence': stat}
        entity.evidence.observe(episode, True)
        entity.last_observed = step
        return entity

    def _prune_candidates(self, gii, giii):
        candidates = [e for e in giii.entity_nodes.values() if e.status == 'candidate']
        candidates.sort(key=lambda e: (e.evidence.support, e.last_observed, e.node_id))
        excess = max(0, len(candidates) - self.cfg.graph_candidate_memory_budget)
        remove = {e.node_id for e in candidates[:excess]}
        remove.update(e.node_id for e in candidates
                      if giii.observations - e.last_observed > self.cfg.graph_candidate_expiry)
        for eid in remove:
            entity = giii.entity_nodes.pop(eid)
            for member in entity.component_edges.values():
                gii.parent_entities[member['sem_id']].discard(eid)
        return sorted(remove)

    @torch.no_grad()
    def learn(self, features, gi, gii, giii, valid_mask=None, source_id=None, episode_id=None,
              H_orig=None, W_orig=None):
        # Stage timings include completed CUDA work, not only asynchronous launches.
        stage_seconds = {}
        self.work_counts = Counter()
        device = torch.device(self.cfg.device)
        def clock():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            return time.perf_counter()
        last_stage = clock()
        def mark(name):
            nonlocal last_stage
            now = clock()
            stage_seconds[name] = now - last_stage
            last_stage = now

        inputs = prepare_feature_subspaces(features, self.cfg)
        fingerprint = self.source_digest(inputs)
        source = str(source_id) if source_id is not None else fingerprint
        episode = str(episode_id) if episode_id is not None else fingerprint
        if valid_mask is None and inputs and H_orig is not None and W_orig is not None:
            h, w = next(iter(inputs.values())).shape[-2:]
            valid_mask = np.zeros((h, w), bool)
            y, x = max(0, h // 2 - int(H_orig) // 2), max(0, w // 2 - int(W_orig) // 2)
            valid_mask[y:min(h, y + int(H_orig)), x:min(w, x + int(W_orig))] = True
        mark('prepare_and_digest')
        prior = self.retriever.query(inputs, gi, gii, giii, valid_mask)
        mark('prior_query')
        oi, oii = GmemoryI(), GmemoryII(self.cfg)
        # Geometry-preserving occurrence mode is compulsory for long-term learning.
        if self.cfg.once_node_reuse_mode != 'sample_instance':
            raise ValueError('Hierarchy learning requires sample_instance; prototype mode remains a raw-build comparison.')
        report = self.builder.build_once(inputs, oi, oii, valid_mask, H_orig, W_orig, giii.observations + 1)
        mark('observation_build')
        result = LearningResult(report, oi, oii, prior)
        if not inputs:
            result.diagnostics = {'source_id': source, 'episode_id': episode, 'empty_input': True,
                                  'stage_seconds': stage_seconds, 'work_counts': dict(self.work_counts)}
            return result
        original_provider = FeatureResponseCache(inputs, gi, self.cfg, report.valid_mask)
        decisions, ambiguous = {}, []
        # Every match sees the same frozen original pools, even within this observation.
        for region in report.regions:
            if region.semantic_id is None:
                continue
            match, uncertain = self._region_match(region, report, gi, gii, original_provider, prior)
            if uncertain:
                ambiguous.append(region.region_id)
            else:
                decisions[region.region_id] = match
        mark('region_matching')
        # Stage updates, including indexes and evidence, before touching persistent pools.
        wi, wii, wiii = copy.deepcopy(gi), copy.deepcopy(gii), copy.deepcopy(giii)
        mark('pool_deepcopy')
        self._working_ii = wii
        self._working_iii = wiii
        wiii.observations += 1
        mapped, positions, node_mapping = {}, {}, {}
        by_region = {r.region_id: r for r in report.regions}
        new_regions = updated_regions = 0
        for rid, match in decisions.items():
            region = by_region[rid]
            if match is None:
                sem = self._copy_region(oii.semantic_nodes[region.semantic_id], oi, wi, wii, node_mapping)
                position = (region.anchor % report.shape[1], region.anchor // report.shape[1])
                new_regions += 1
            else:
                sem = wii.semantic_nodes[match.template_id]
                position = match.point
                updated_regions += 1
            self._update_region(sem, match, episode)
            mapped[rid], positions[rid] = sem.node_id, position
        region_remap = self._merge_identical_regions(wi, wii, wiii)
        mapped = {rid: region_remap.get(sid, sid) for rid, sid in mapped.items()}
        mark('region_update_and_merge')
        proposals, truncated = self._relations_and_proposals(report, mapped, positions, wii, episode)
        mark('relations_and_proposals')
        provider = FeatureResponseCache(inputs, wi, self.cfg, report.valid_mask)
        new_entities, updated_entities, unresolved_entities = 0, 0, 0
        entity_instances = []
        frozen_entities = copy.deepcopy(wiii.entity_nodes)
        mark('entity_snapshot')
        # New candidates do not become evidence for themselves in this observation.
        updated_occurrences = set()
        for proposal in proposals:
            choices = []
            for eid, entity in sorted(frozen_entities.items()):
                aligned = self._align_entity(entity, proposal, mapped, positions)
                if aligned and aligned[0] >= self.cfg.graph_learn_threshold:
                    choices.append((aligned[0], eid, aligned))
            choices.sort(key=lambda x: (-x[0], x[1]))
            if len(choices) > 1 and choices[0][0] - choices[1][0] < self.cfg.graph_match_margin:
                unresolved_entities += 1
                continue
            if choices:
                _, eid, aligned = choices[0]
                entity = wiii.entity_nodes[eid]
                occurrence = (eid, tuple(round(v, 3) for v in aligned[1]))
                if occurrence in updated_occurrences:
                    continue
                updated_occurrences.add(occurrence)
                self._update_entity(entity, aligned, proposal, mapped, positions, provider, episode, wiii.observations)
                updated_entities += 1
                instance_point = aligned[1]
            else:
                entity = self._new_entity(proposal, mapped, positions, wii, wiii, episode, wiii.observations)
                new_entities += 1
                instance_point = positions[proposal[0]]
            result.entity_ids.append(entity.node_id)
            entity_instances.append({'entity_id': entity.node_id, 'point': tuple(instance_point)})
        mark('entity_alignment_and_update')
        entity_remap = self._merge_identical_entities(wii, wiii)
        removed = self._prune_candidates(wii, wiii)
        # One commit; observations and their truth coordinates remain separate.
        gi.__dict__.update(wi.__dict__)
        gii.__dict__.update(wii.__dict__)
        giii.__dict__.update(wiii.__dict__)
        result.region_mapping = {rid: {'semantic_id': sid, 'point': positions[rid]} for rid, sid in mapped.items()}
        result.entity_ids = sorted({entity_remap.get(eid, eid) for eid in result.entity_ids} - set(removed))
        mark('entity_merge_prune_commit')
        result.diagnostics = {'source_id': source, 'episode_id': episode,
            'stage_seconds': stage_seconds, 'work_counts': dict(self.work_counts),
            'new_regions': new_regions, 'updated_regions': updated_regions,
            'ambiguous_regions': ambiguous, 'new_entities': new_entities,
            'updated_entities': updated_entities, 'ambiguous_entity_proposals': unresolved_entities,
            'proposal_budget_truncated': truncated, 'pruned_candidates': removed,
            'merged_region_ids': region_remap, 'merged_entity_ids': entity_remap,
            'entity_instances': [{'entity_id': entity_remap.get(item['entity_id'], item['entity_id']),
                                  'point': item['point']} for item in entity_instances
                                 if entity_remap.get(item['entity_id'], item['entity_id']) not in removed],
            'memory_counts': (len(gi.nodes), len(gii.semantic_nodes), len(giii.entity_nodes)),
            'learning_match_mode': 'dense_local_translation',
            'limitations': ['Bounded spatial/contact proposals; no semantic object labels.',
                           'Conservative ambiguity handling; no automatic shared-template reanchoring.',
                           'Distinct incompatible layouts are retained instead of averaging their geometry.']}
        return result

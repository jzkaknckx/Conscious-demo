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

    # View-window vector optimizer params
    window_r_min = 12.0
    window_r_max = 96.0
    window_candidate_count = 5
    window_seed_radius = 5
    window_grow_steps = 64
    theta_window_work = 0.08
    theta_window_low = 0.03
    theta_barrier = 0.65
    attention_budget = 512.0
    window_radius_smooth = 0.35
    window_log_radius_step_max = 0.45
    kappa_step_min = 0.18
    kappa_step_max = 0.85
    kappa_edge_step = 0.38
    kappa_surface_delta = 0.30

    # Dynamic state competition params
    state_temperature = 1.0
    state_switch_margin = 0.20
    state_min_age = 2
    theta_review_preempt = 0.45
    review_preempt_margin = 0.15
    review_recent_decay = 0.90
    resume_value_threshold = 0.25
    fail_decay = 0.85
    fail_increment = 1.0
    theta_vector_norm = 1.5
    theta_complete = 0.72
    theta_frontier_close = 0.05
    theta_novel_close = 0.05
    theta_gap_close = 0.08

    state_bias_macro = 0.05
    state_bias_micro = 0.10
    state_bias_review = 0.00
    state_bias_review = -10.00
    state_bias_resume = -0.05
    state_inertia_macro = 0.18
    state_inertia_micro = 0.25
    state_inertia_review = 0.12
    state_inertia_resume = 0.10
    z_micro_novel = 1.1
    z_micro_frontier = 1.2
    z_micro_gap = 0.8
    z_micro_quality = 0.6
    z_micro_complete = 0.9
    z_micro_fail = 0.9
    z_macro_fail = 1.1
    z_macro_noise = 0.8
    z_macro_complete = 0.8
    z_macro_global = 0.6
    z_macro_recall = 0.8
    z_review_out = 1.4
    z_review_in = 0.5
    z_review_recent = 0.8
    z_resume_value = 1.2

    # Local potential and keypoint sampling params
    dynamic_decay = 0.82
    dynamic_w_edge = 0.9
    dynamic_w_surface = 1.0
    dynamic_w_event = 0.55
    dynamic_w_gap = 0.65
    dynamic_w_return = 0.75
    theta_edge_low = 0.10
    theta_edge_high = 0.65
    theta_surf_low = 0.10
    theta_surf_high = 0.72
    theta_event = 0.45
    theta_novel = 0.18
    vector_w_potential = 1.0
    vector_w_low_gpos = 0.55
    vector_w_gap = 0.55
    vector_w_novel = 0.45
    vector_w_inertia = 0.30
    vector_w_return = 0.50
    keypoint_nms_radius = 6
    max_keypoints_dim0 = 2
    max_keypoints_dim1 = 3
    max_keypoints_dim2 = 3
    ell_base_dim0 = 1.0
    ell_base_dim1 = 3.0
    ell_base_dim2 = 5.0
    ell_max_dim0 = 8.0
    ell_max_dim1 = 28.0
    ell_max_dim2 = 48.0
    ell_eta_dim0 = 0.45
    ell_eta_dim1 = 0.55
    ell_eta_dim2 = 0.65
    theta_surface_rho_expand = 4.0
    surface_theta_bins = 36
    self_extent_alpha = 0.20


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


class OptimizerState(Enum):
    MACRO_SEARCH = 1
    MICRO_BUILD = 2
    REVIEW_VERIFY = 3
    SUSPEND_RESUME = 4


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


@dataclass
class WindowState:
    mask: torch.Tensor
    work_response: torch.Tensor
    barrier: torch.Tensor
    frontier: torch.Tensor
    r_eff: float
    area: float
    beta: float
    scores: Dict[str, float]


@dataclass
class StateObservation:
    novelty: float = 0.0
    frontier: float = 0.0
    quality: float = 0.0
    completion: float = 0.0
    recall_out: float = 0.0
    recall_in: float = 0.0
    gap: float = 0.0
    fail: float = 0.0
    global_novelty: float = 0.0
    review_target: Optional[Tuple[int, int]] = None
    resume_value: float = 0.0


@dataclass
class StateScores:
    macro: float
    micro: float
    review: float
    resume: float

    def as_dict(self) -> Dict[str, float]:
        return {
            'MACRO_SEARCH': self.macro,
            'MICRO_BUILD': self.micro,
            'REVIEW_VERIFY': self.review,
            'SUSPEND_RESUME': self.resume,
        }


@dataclass
class KeypointCandidate:
    x: int
    y: int
    kind: str
    support_dim: int
    modality_id: int
    score: float
    node_id: Optional[int] = None


@dataclass
class VectorDecision:
    vector: Tuple[float, float]
    target: Tuple[int, int]
    norm: float
    potential: torch.Tensor


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
        self.surface_extents: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)
        self.texture_density = 0.0
        self.detail_entropy = 0.0
        self.detail_pressure = 0.0
        self.frontier_strength = 1.0
        self.gap_strength = 0.0

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

    def valid_view_bounds(self, H: int, W: int) -> Tuple[int, int, int, int]:
        if self.H_orig is None or self.W_orig is None:
            return 0, H, 0, W
        margin = int(self.r or 0)
        valid_h = min(H, max(1, int(self.H_orig) - margin))
        valid_w = min(W, max(1, int(self.W_orig) - margin))
        y1 = max(0, (H - valid_h) // 2)
        x1 = max(0, (W - valid_w) // 2)
        y2 = min(H, y1 + valid_h)
        x2 = min(W, x1 + valid_w)
        return y1, y2, x1, x2

    def valid_view_mask(self, ref: torch.Tensor) -> torch.Tensor:
        H, W = ref.shape[-2], ref.shape[-1]
        mask = torch.zeros_like(ref)
        y1, y2, x1, x2 = self.valid_view_bounds(H, W)
        mask[:, :, y1:y2, x1:x2] = 1.0
        return mask

    def suppress_invalid_view(self, drive: torch.Tensor, invalid_value: float = 0.0) -> torch.Tensor:
        if self.H_orig is None or self.W_orig is None:
            return drive
        mask = self.valid_view_mask(drive)
        fill = torch.full_like(drive, invalid_value)
        return torch.where(mask > 0.0, drive, fill)
        
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
        self.base_interest = self.suppress_invalid_view(self.base_interest, invalid_value=0.0)

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
        
        I_map = self.suppress_invalid_view(I_map, invalid_value=0.0)
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
        
        self.state = MainPhase.LEARN_MACRO
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
        self.pending_boundary_target: Optional[Tuple[int, int]] = None
        self.boundary_rejected: Optional[torch.Tensor] = None
        self.boundary_suppression: Optional[torch.Tensor] = None
        self.trace_age = 0
        self.boundary_hit_confidence = 0.0
        self.boundary_forward_support = 0.0
        self.boundary_role_confidence = 0.0

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

        # Dynamic view-window vector optimizer state.
        self.behavior_state = OptimizerState.MACRO_SEARCH
        self.state_age = 0
        self.dynamic_matrix: Optional[torch.Tensor] = None
        self.work_window: Optional[torch.Tensor] = None
        self.work_response: Optional[torch.Tensor] = None
        self.work_barrier: Optional[torch.Tensor] = None
        self.work_frontier: Optional[torch.Tensor] = None
        self.work_radius_eff = float(cfg.window_r_min)
        self.prev_local_vector = (0.0, 0.0)
        self.fail_energy = 0.0
        self.review_recent_fail = 0.0
        self.review_target: Optional[Tuple[int, int]] = None
        self.current_window_state: Optional[WindowState] = None
        self.current_state_observation = StateObservation()
        self.current_state_scores = StateScores(0.0, 0.0, 0.0, 0.0)
        self.last_keypoints: List[KeypointCandidate] = []
        self.prev_keypoints: List[KeypointCandidate] = []
        self.debug_optimizer: Dict[str, Any] = {}

    # ---- Dynamic view-window primitives ----
    def _circle_mask(self, pt: Tuple[int, int], radius: float, H: int, W: int) -> torch.Tensor:
        cx, cy = pt
        x_grid, y_grid = self._xy_grids(H, W)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        return (dist_sq <= float(radius) ** 2).float().view(1, 1, H, W)

    def _valid_view_mask_like(self, ref: torch.Tensor) -> torch.Tensor:
        return self.optimizer.valid_view_mask(ref)

    def _morph_dilate(self, mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
        k = max(1, 2 * int(radius) + 1)
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

    def _morph_erode(self, mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
        return 1.0 - self._morph_dilate(1.0 - mask, radius)

    def _mask_boundary(self, mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
        return (self._morph_dilate(mask, radius) - self._morph_erode(mask, radius)).clamp(0.0, 1.0)

    def _mask_area(self, mask: torch.Tensor) -> float:
        return float(torch.sum(mask).item())

    def _support_dim_for_topology(self, topology: str) -> int:
        if topology in {'Surface_2D', 'Texture_Surface', 'Entity_Shape'}:
            return 2
        if topology in {'Boundary_Edge', 'Trace_1D'}:
            return 1
        return 0

    def _support_dim_for_node(self, node: ModalityNode) -> int:
        return self._support_dim_for_topology(self._node_topology(node))

    def _max_s_maps(self, S_maps: Dict[int, torch.Tensor], node_ids: Optional[Set[int]] = None,
                    ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        maps = []
        for nid, value in S_maps.items():
            if node_ids is None or nid in node_ids:
                maps.append(value)
        if maps:
            return torch.stack(maps, dim=0).max(dim=0).values
        if ref is not None:
            return torch.zeros_like(ref)
        if self.optimizer.base_interest is not None:
            return torch.zeros_like(self.optimizer.base_interest)
        return torch.zeros((1, 1, 1, 1), device=self.device)

    def _raw_response_map(self, X_subspaces: Dict[int, torch.Tensor]) -> torch.Tensor:
        ref = self.optimizer.base_interest
        if ref is None:
            sample = next(iter(X_subspaces.values()))
            ref = torch.zeros((sample.shape[0], 1, sample.shape[-2], sample.shape[-1]), device=self.device)
        raw = torch.zeros_like(ref)
        total_w = 0.0
        for mod_id, X_m in X_subspaces.items():
            gate = SimilarityEngine.compute_gate_map(X_m, self.cfg, mod_id)
            weight = float(self.cfg.modality_weights.get(mod_id, 1.0))
            raw += weight * self._normalize_map(gate, ref)
            total_w += weight
        if total_w > 1e-6:
            raw = raw / total_w
        return self._suppress_invalid_view(self._normalize_map(raw, ref), invalid_value=0.0)

    def _explained_response_map(self, S_maps: Dict[int, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        return self._normalize_map(self._max_s_maps(S_maps, ref=ref), ref)

    def _semantic_related_ids(self, gmem_ii: GmemoryII) -> Optional[Set[int]]:
        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is None:
            return None
        return sem_node.related_node_ids()

    def _build_work_response(self, X_subspaces: Dict[int, torch.Tensor], S_maps: Dict[int, torch.Tensor],
                             gmem_ii: GmemoryII) -> torch.Tensor:
        raw = self._raw_response_map(X_subspaces)
        explained = self._explained_response_map(S_maps, raw)
        related_ids = self._semantic_related_ids(gmem_ii)
        if related_ids:
            related = self._normalize_map(self._max_s_maps(S_maps, related_ids, raw), raw)
        else:
            related = explained
        novelty = (raw * (1.0 - explained)).clamp(0.0, 1.0)
        frontier = self.work_frontier if self.work_frontier is not None else torch.zeros_like(raw)
        work = 0.55 * related + 0.25 * raw + 0.15 * novelty + 0.20 * self._normalize_map(frontier, raw)
        return self._suppress_invalid_view(self._normalize_map(work, raw), invalid_value=0.0)

    def _build_barrier_map(self, X_subspaces: Dict[int, torch.Tensor]) -> torch.Tensor:
        ref = self.optimizer.base_interest
        if ref is None:
            sample = next(iter(X_subspaces.values()))
            ref = torch.zeros((sample.shape[0], 1, sample.shape[-2], sample.shape[-1]), device=self.device)
        barrier = torch.zeros_like(ref)
        if self.optimizer.I_str is not None:
            barrier = torch.max(barrier, self._normalize_map(self.optimizer.I_str, ref))
        surface = X_subspaces.get(1)
        if surface is not None and surface.shape[1] > 0:
            dx = surface[:, :, :, 1:] - surface[:, :, :, :-1]
            dy = surface[:, :, 1:, :] - surface[:, :, :-1, :]
            dx = F.pad(dx, (0, 1, 0, 0))
            dy = F.pad(dy, (0, 0, 0, 1))
            transition = torch.sqrt((dx ** 2 + dy ** 2).mean(dim=1, keepdim=True) + 1e-6)
            barrier = torch.max(barrier, self._normalize_map(transition, ref))
        invalid = 1.0 - self._valid_view_mask_like(ref)
        barrier = torch.max(barrier, invalid)
        return barrier.clamp(0.0, 1.0)

    def _candidate_area_budgets(self) -> List[float]:
        count = max(1, int(self.cfg.window_candidate_count))
        amin = math.pi * float(self.cfg.window_r_min) ** 2
        amax = math.pi * float(self.cfg.window_r_max) ** 2
        if count == 1:
            return [amin]
        return [
            amin * ((amax / amin) ** (idx / (count - 1)))
            for idx in range(count)
        ]

    def _seeded_grow(self, response: torch.Tensor, barrier: torch.Tensor, budget: float,
                     seed: Tuple[int, int]) -> torch.Tensor:
        H, W = response.shape[-2], response.shape[-1]
        valid = self._valid_view_mask_like(response)
        cap_radius = math.sqrt(max(float(budget), 1.0) / math.pi)
        cap = self._circle_mask(seed, min(cap_radius, self.cfg.window_r_max), H, W).to(response.device)
        seed_mask = self._circle_mask(seed, self.cfg.window_seed_radius, H, W).to(response.device)
        low = max(float(self.cfg.theta_window_low), min(float(self.cfg.theta_window_work), float(response.max().item()) * 0.35))
        allowed = (response > low).float() * (barrier < self.cfg.theta_barrier).float() * valid * cap
        W_mask = seed_mask * valid * cap
        if torch.sum(W_mask * allowed).item() <= 0.0:
            W_mask = seed_mask * valid
        else:
            W_mask = W_mask * allowed
        for _ in range(int(self.cfg.window_grow_steps)):
            if self._mask_area(W_mask) >= budget:
                break
            grown = self._morph_dilate(W_mask, 1) * allowed
            if torch.equal((grown > 0).float(), (W_mask > 0).float()):
                break
            W_mask = grown
        if self._mask_area(W_mask) < 1.0:
            W_mask = seed_mask * valid
        return W_mask.clamp(0.0, 1.0)

    def _score_window_candidate(self, mask: torch.Tensor, response: torch.Tensor, barrier: torch.Tensor,
                                budget: float) -> Tuple[float, Dict[str, float], torch.Tensor]:
        valid = self._valid_view_mask_like(mask)
        area = max(self._mask_area(mask), 1.0)
        response_norm = self._normalize_map(response, mask)
        union = torch.max(mask, response_norm)
        domain = float(torch.sum(mask * response_norm).item() / (torch.sum(union).item() + 1e-6))
        valid_ratio = float(torch.sum(mask * valid).item() / area)
        boundary = self._mask_boundary(mask, 1)
        frontier = boundary * response_norm * (1.0 - self.m_aversion.clamp(0.0, 1.0))
        frontier_score = float(torch.sum(frontier).item() / (torch.sum(boundary).item() + 1e-6))
        coarse = float(torch.sum(mask * response_norm).item() / area)
        fine = float(torch.sum(mask * response_norm * barrier).item() / area)
        mix = float(torch.sum(mask * (barrier > self.cfg.theta_barrier).float()).item() / area)
        r_eff = math.sqrt(area / math.pi)
        scale_penalty = abs(math.log(max(r_eff, 1.0)) - math.log(max(self.work_radius_eff, 1.0)))
        score = (
            1.2 * domain
            + 0.7 * frontier_score
            + 0.5 * coarse
            + 0.6 * valid_ratio
            - 0.7 * fine
            - 0.8 * mix
            - 0.25 * scale_penalty
        )
        detail = {
            'domain': domain,
            'frontier': frontier_score,
            'coarse': coarse,
            'valid': valid_ratio,
            'fine': fine,
            'mix': mix,
            'r_eff': r_eff,
            'budget': budget,
        }
        return score, detail, frontier

    def _select_dynamic_window(self, X_subspaces: Dict[int, torch.Tensor], S_maps: Dict[int, torch.Tensor],
                               gmem_ii: GmemoryII) -> WindowState:
        response = self._build_work_response(X_subspaces, S_maps, gmem_ii)
        barrier = self._build_barrier_map(X_subspaces)
        H, W = response.shape[-2], response.shape[-1]
        seed = self.anchor_position if self.active_semantic_id is not None and self.anchor_position else self.fixation_point

        best_mask = None
        best_score = -float('inf')
        best_detail = {}
        best_frontier = torch.zeros_like(response)
        for budget in self._candidate_area_budgets():
            candidate = self._seeded_grow(response, barrier, budget, seed)
            score, detail, frontier = self._score_window_candidate(candidate, response, barrier, budget)
            if score > best_score:
                best_score = score
                best_mask = candidate
                best_detail = detail
                best_frontier = frontier

        if best_mask is None:
            best_mask = self._circle_mask(self.fixation_point, self.cfg.window_r_min, H, W)
            best_frontier = self._mask_boundary(best_mask, 1) * response
            best_detail = {'r_eff': self.cfg.window_r_min, 'domain': 0.0, 'frontier': 0.0, 'valid': 1.0}

        observed_radius = max(float(best_detail.get('r_eff', self.cfg.window_r_min)), 1.0)
        prev = max(float(self.work_radius_eff), 1.0)
        target_log = math.log(observed_radius)
        prev_log = math.log(prev)
        delta = max(-self.cfg.window_log_radius_step_max, min(self.cfg.window_log_radius_step_max, target_log - prev_log))
        smooth_log = prev_log + self.cfg.window_radius_smooth * delta
        self.work_radius_eff = float(math.exp(smooth_log))
        area = max(self._mask_area(best_mask), 1.0)
        beta = float(self.cfg.attention_budget / (area + 1e-6))
        self.work_window = best_mask
        self.work_response = response
        self.work_barrier = barrier
        self.work_frontier = best_frontier
        state = WindowState(
            mask=best_mask,
            work_response=response,
            barrier=barrier,
            frontier=best_frontier,
            r_eff=self.work_radius_eff,
            area=area,
            beta=beta,
            scores={'selected': best_score, **best_detail},
        )
        self.current_window_state = state
        self.matrix2 = best_mask
        self.matrix3 = best_frontier
        return state

    # ---- Dynamic state competition ----
    def _state_to_main_phase(self, state: OptimizerState) -> MainPhase:
        if state == OptimizerState.MACRO_SEARCH:
            return MainPhase.LEARN_MACRO
        if state == OptimizerState.REVIEW_VERIFY:
            return MainPhase.REVIEW
        return MainPhase.LEARN_MICRO

    def _semantic_recall_map(self, gmem_ii: GmemoryII, S_maps: Dict[int, torch.Tensor],
                             ref: torch.Tensor) -> torch.Tensor:
        recall = torch.zeros_like(ref)
        for sem_node in gmem_ii.semantic_nodes.values():
            maps = []
            for nid in sem_node.related_node_ids():
                if nid in S_maps:
                    maps.append(S_maps[nid])
            if not maps:
                continue
            sem_resp = torch.stack(maps, dim=0).mean(dim=0)
            confidence = 1.0 if sem_node.is_completed else 0.5
            recall = torch.max(recall, sem_resp * confidence)
        return self._suppress_invalid_view(self._normalize_map(recall, ref), invalid_value=0.0)

    def _peak_outside_window(self, value: torch.Tensor, window: torch.Tensor) -> Tuple[float, Optional[Tuple[int, int]]]:
        outside = (1.0 - window).clamp(0.0, 1.0) * self._valid_view_mask_like(value)
        masked = value * outside
        if masked.max() <= 1e-6:
            return 0.0, None
        idx = torch.argmax(masked).item()
        H, W = masked.shape[-2], masked.shape[-1]
        return float(masked.max().item()), (int(idx % W), int(idx // W))

    def _peak_inside_window(self, value: torch.Tensor, window: torch.Tensor) -> float:
        masked = value * window
        if masked.max() <= 1e-6:
            return 0.0
        return float(masked.max().item())

    def _estimate_completion(self, sem_node: Optional[SemanticNode],
                             window_state: WindowState, novelty: float) -> float:
        if sem_node is None:
            return 0.0
        relation_score = min(1.0, len(sem_node.relation_edge_ids) / 8.0)
        child_score = min(1.0, len(sem_node.child_node_ids) / 6.0)
        frontier_penalty = min(1.0, sem_node.frontier_strength)
        score = (
            0.25 * float(sem_node.is_completed)
            + 0.25 * sem_node.boundary_coverage
            + 0.20 * relation_score
            + 0.15 * child_score
            + 0.15 * window_state.scores.get('coarse', 0.0)
            - 0.20 * novelty
            - 0.15 * frontier_penalty
        )
        return float(max(0.0, min(1.0, score)))

    def _resume_value(self) -> float:
        if not self.suspend_stack:
            return 0.0
        ctx = self.suspend_stack[-1]
        age = max(0, self.current_step - int(ctx.get('step', self.current_step)))
        frontier = float(ctx.get('frontier', 0.0))
        gap = float(ctx.get('gap', 0.0))
        completion = float(ctx.get('completion', 0.0))
        return max(0.0, 0.5 * (1.0 - completion) + 0.35 * frontier + 0.25 * gap - 0.02 * age)

    def _compute_state_observation(self, window_state: WindowState, X_subspaces: Dict[int, torch.Tensor],
                                   S_maps: Dict[int, torch.Tensor], gmem_ii: GmemoryII) -> StateObservation:
        ref = window_state.mask
        raw = self._raw_response_map(X_subspaces)
        explained = self._explained_response_map(S_maps, ref)
        novelty_map = (raw * (1.0 - explained)).clamp(0.0, 1.0)
        area = max(window_state.area, 1.0)
        novelty = float(torch.sum(novelty_map * window_state.mask).item() / area)
        frontier = float(torch.sum(window_state.frontier).item() / area)
        quality = float(torch.sum(window_state.mask * self._valid_view_mask_like(ref)).item() / area)
        recall = self._semantic_recall_map(gmem_ii, S_maps, ref)
        recall_out, target = self._peak_outside_window(recall, window_state.mask)
        recall_in = self._peak_inside_window(recall, window_state.mask)
        sem_node = self._current_semantic_node(gmem_ii)
        completion = self._estimate_completion(sem_node, window_state, novelty)
        gap = float(getattr(sem_node, 'gap_strength', 0.0)) if sem_node is not None else 0.0
        if sem_node is not None and not sem_node.is_completed:
            gap = max(gap, (1.0 - completion) * frontier)
        global_novelty = float(torch.max(novelty_map * (1.0 - window_state.mask)).item())
        observation = StateObservation(
            novelty=novelty,
            frontier=frontier,
            quality=quality,
            completion=completion,
            recall_out=recall_out,
            recall_in=recall_in,
            gap=gap,
            fail=float(self.fail_energy),
            global_novelty=global_novelty,
            review_target=target,
            resume_value=self._resume_value(),
        )
        self.current_state_observation = observation
        return observation

    def _state_inertia(self, state: OptimizerState) -> float:
        if self.behavior_state == state:
            if state == OptimizerState.MACRO_SEARCH:
                return self.cfg.state_inertia_macro
            if state == OptimizerState.MICRO_BUILD:
                return self.cfg.state_inertia_micro
            if state == OptimizerState.REVIEW_VERIFY:
                return self.cfg.state_inertia_review
            return self.cfg.state_inertia_resume
        return 0.0

    def _score_optimizer_states(self, obs: StateObservation) -> StateScores:
        micro = (
            self.cfg.state_bias_micro
            + self.cfg.z_micro_novel * obs.novelty
            + self.cfg.z_micro_frontier * obs.frontier
            + self.cfg.z_micro_gap * obs.gap
            + self.cfg.z_micro_quality * obs.quality
            - self.cfg.z_micro_complete * obs.completion
            - self.cfg.z_micro_fail * obs.fail
            + self._state_inertia(OptimizerState.MICRO_BUILD)
        )
        macro = (
            self.cfg.state_bias_macro
            + self.cfg.z_macro_fail * obs.fail
            + self.cfg.z_macro_noise * (1.0 - obs.quality)
            + self.cfg.z_macro_complete * obs.completion
            + self.cfg.z_macro_global * obs.global_novelty
            - self.cfg.z_macro_recall * obs.recall_out
            + self._state_inertia(OptimizerState.MACRO_SEARCH)
        )
        review = (
            self.cfg.state_bias_review
            + self.cfg.z_review_out * obs.recall_out
            + self.cfg.z_review_in * obs.recall_in
            - self.cfg.z_review_recent * self.review_recent_fail
            + self._state_inertia(OptimizerState.REVIEW_VERIFY)
        )
        resume = (
            self.cfg.state_bias_resume
            + self.cfg.z_resume_value * obs.resume_value
            + self._state_inertia(OptimizerState.SUSPEND_RESUME)
        )
        scores = StateScores(macro=macro, micro=micro, review=review, resume=resume)
        self.current_state_scores = scores
        return scores

    def _suspend_current_work(self, obs: StateObservation):
        if self.active_semantic_id is None:
            return
        if self.suspend_stack and self.suspend_stack[-1].get('sid') == self.active_semantic_id:
            return
        self.suspend_stack.append({
            'sid': self.active_semantic_id,
            'anchor': self.anchor_position,
            'mask': self.semantic_mask,
            'explore_time': self.micro_explore_time,
            'max_span': self.micro_max_span,
            'behavior_state': self.behavior_state,
            'frontier': obs.frontier,
            'gap': obs.gap,
            'completion': obs.completion,
            'step': self.current_step,
        })
        self.active_semantic_id = None
        self.anchor_position = None
        self.semantic_mask = None
        self.prev_keypoints = []

    def _restore_suspended_work(self) -> bool:
        if not self.suspend_stack:
            return False
        ctx = self.suspend_stack.pop()
        self.active_semantic_id = ctx.get('sid')
        self.anchor_position = ctx.get('anchor')
        self.semantic_mask = ctx.get('mask')
        self.micro_explore_time = float(ctx.get('explore_time', 0.0))
        self.micro_max_span = float(ctx.get('max_span', 0.0))
        return self.active_semantic_id is not None

    def _set_behavior_state(self, next_state: OptimizerState):
        if next_state != self.behavior_state:
            self.state_age = 0
        else:
            self.state_age += 1
        self.behavior_state = next_state
        self.state = self._state_to_main_phase(next_state)

    def _select_behavior_state(self, obs: StateObservation, scores: StateScores) -> OptimizerState:
        current_score = scores.as_dict()[self.behavior_state.name]
        if (
            obs.review_target is not None
            and obs.recall_out > self.cfg.theta_review_preempt
            and scores.review > max(scores.micro, scores.macro) + self.cfg.review_preempt_margin
            and self.behavior_state != OptimizerState.REVIEW_VERIFY
        ):
            self._suspend_current_work(obs)
            self.review_target = obs.review_target
            self.debug_optimizer['selection_reason'] = 'review_preempt'
            return OptimizerState.REVIEW_VERIFY

        candidates = {
            OptimizerState.MACRO_SEARCH: scores.macro,
            OptimizerState.MICRO_BUILD: scores.micro,
            OptimizerState.REVIEW_VERIFY: scores.review,
            OptimizerState.SUSPEND_RESUME: scores.resume,
        }
        winner = max(candidates, key=candidates.get)
        if winner == OptimizerState.SUSPEND_RESUME and obs.resume_value <= self.cfg.resume_value_threshold:
            candidates[winner] = -float('inf')
            winner = max(candidates, key=candidates.get)
        if winner != self.behavior_state:
            if self.state_age < self.cfg.state_min_age:
                self.debug_optimizer['selection_reason'] = 'minimum_age'
                return self.behavior_state
            if candidates[winner] <= current_score + self.cfg.state_switch_margin:
                self.debug_optimizer['selection_reason'] = 'switch_margin'
                return self.behavior_state
        self.debug_optimizer['selection_reason'] = 'score_winner'
        return winner

    def _feature_spec(self, mod_id: int) -> Dict[str, Any]:
        return getattr(self.cfg, 'feature_specs', {}).get(mod_id, {})

    def _sample_feature_scalar(self, X_m: torch.Tensor, channel: int, x: int, y: int) -> float:
        if channel >= X_m.shape[1]:
            return 0.0
        return float(X_m[0, channel, y, x].item())

    def _local_variance_value(self, X_m: torch.Tensor, x: int, y: int, radius: int) -> float:
        H, W = X_m.shape[-2], X_m.shape[-1]
        r = max(0, int(radius))
        x1, x2 = max(0, x - r), min(W, x + r + 1)
        y1, y2 = max(0, y - r), min(H, y + r + 1)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        window = X_m[:, :, y1:y2, x1:x2]
        mean = window.mean(dim=(-2, -1), keepdim=True)
        return float(((window - mean) ** 2).mean().item())

    def _evaluate_gates(self, mod_id: int, X_m: torch.Tensor, x: int, y: int) -> Tuple[bool, Dict[str, bool], Dict[str, float]]:
        gate_map = SimilarityEngine.compute_gate_map(X_m, self.cfg, mod_id)
        gate_passed = bool(gate_map[0, 0, y, x].item() > 0.5)
        gates = {'modality': gate_passed}
        context = {'gate': float(gate_map[0, 0, y, x].item())}
        C_m = X_m.shape[1]

        if mod_id == 0:
            if C_m >= 4:
                strength = abs(self._sample_feature_scalar(X_m, 0, x, y))
                theta = self._sample_feature_scalar(X_m, 1, x, y)
                theta_map = X_m[:, 1:2]
                strength_map = X_m[:, 0:1].abs()
            elif C_m >= 2:
                gx = X_m[:, 0:1]
                gy = X_m[:, 1:2]
                strength_map = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
                theta_map = torch.atan2(gy, gx)
                strength = float(strength_map[0, 0, y, x].item())
                theta = float(theta_map[0, 0, y, x].item())
            else:
                strength_map = X_m[:, 0:1].abs()
                theta_map = torch.zeros_like(strength_map)
                strength = float(strength_map[0, 0, y, x].item())
                theta = 0.0
            coh_map = SimilarityEngine._circular_coherence_map(
                theta_map,
                strength_map,
                self.cfg.grad_coh_radius,
                self.cfg.grad_axis_k,
            )
            coherence = float(coh_map[0, 0, y, x].item())
            gates['grad_strength'] = strength > self.cfg.tau_grad_str
            gates['grad_coherence'] = coherence > self.cfg.tau_grad_coh
            context.update({'strength': strength, 'theta': theta, 'coherence': coherence})

        elif mod_id == 1:
            variance = self._local_variance_value(X_m, x, y, self.cfg.rgb_gate_radius)
            gates['rgb_surface'] = gate_passed
            context['local_variance'] = variance

        elif mod_id == 2:
            value = float(X_m[0, :, y, x].abs().max().item())
            gates['curv_parent'] = gate_passed
            context['max_abs_curv'] = value

        elif mod_id == 3:
            value = float(X_m[0, :, y, x].abs().max().item())
            gates['aps_support'] = gate_passed
            context['max_abs_aps'] = value

        elif mod_id == 4:
            gates['ori_coherence'] = gate_passed
            context['orientation'] = float(X_m[0, 0, y, x].item())

        return gate_passed, gates, context

    def _descriptor_channel_spec(self, spec: Dict[str, Any], channel: int, mod_id: int) -> Dict[str, Any]:
        channel_specs = spec.get('channels', {})
        if channel in channel_specs:
            return channel_specs[channel]
        for name, group in spec.get('metric_groups', {}).items():
            channels = group.get('channels', 'all')
            if channels == 'all' or channel in channels:
                parents = []
                if mod_id == 2:
                    parents = ['curv_parent']
                elif mod_id == 3:
                    parents = ['aps_support']
                elif mod_id == 4:
                    parents = ['ori_coherence']
                return {'role': 'gate_and_sim', 'metric_group': name, 'parents': parents}
        return {'role': 'gate_and_sim'}

    def _parents_pass(self, parents: List[str], gates: Dict[str, bool]) -> bool:
        return all(gates.get(parent, gates.get('modality', False)) for parent in parents)

    def _build_descriptor(self, mod_id: int, X_m: torch.Tensor, x: int, y: int) -> Optional[FeatureDescriptor]:
        C_m = X_m.shape[1]
        if C_m <= 0:
            return None
        spec = self._feature_spec(mod_id)
        gate_passed, gates, gate_context = self._evaluate_gates(mod_id, X_m, x, y)
        if not gate_passed:
            return None

        sim_mask = torch.zeros((1, C_m), device=self.device)
        gate_mask = torch.zeros((1, C_m), device=self.device)
        context_mask = torch.zeros((1, C_m), device=self.device)

        for c in range(C_m):
            ch_spec = self._descriptor_channel_spec(spec, c, mod_id)
            role = ch_spec.get('role', 'gate_and_sim')
            parents = ch_spec.get('parents', [])
            if role in {'gate_only', 'gate_and_sim'}:
                gate_mask[0, c] = 1.0
            if role in {'context_only', 'gate_only'}:
                context_mask[0, c] = 1.0
            if role in {'sim_only', 'gate_and_sim'} and self._parents_pass(parents, gates):
                sim_mask[0, c] = 1.0

        threshold = 0.85
        valid_group = False
        metric_groups = spec.get('metric_groups', {})
        if metric_groups:
            thresholds = []
            for group in metric_groups.values():
                channels = SimilarityEngine._group_channels(group.get('channels', 'all'), C_m)
                if not channels:
                    continue
                count = int(sim_mask[:, channels].sum().item())
                required = int(group.get('min_valid_channels', self.cfg.min_valid_sim_channels))
                required = min(required, len(channels))
                if count >= required:
                    valid_group = True
                    thresholds.append(float(group.get('theta_match', threshold)))
            if not valid_group:
                return None
            threshold = min(thresholds) if thresholds else threshold
        elif int(sim_mask.sum().item()) < self.cfg.min_valid_sim_channels:
            return None

        values = X_m[0, :, y, x].unsqueeze(0)
        topology_context = {'topology': spec.get('topology', self._modality_topology(mod_id))}
        return FeatureDescriptor(
            modality_id=mod_id,
            values=values,
            sim_mask=sim_mask,
            gate_mask=gate_mask,
            context_mask=context_mask,
            gate_context=gate_context,
            topology_context=topology_context,
            match_threshold=threshold,
        )

    def _match_or_create_descriptor(self, descriptor: FeatureDescriptor, gmem_i: GmemoryI,
                                    similarity_threshold: Optional[float] = None) -> ModalityNode:
        matched_node = None
        best_sim = -1.0
        for node in gmem_i.nodes.values():
            if node.modality_id != descriptor.modality_id:
                continue
            w_node = node.prototype.unsqueeze(0)
            sim = SimilarityEngine.calculate_similarity(
                descriptor.values,
                w_node,
                self.cfg,
                descriptor.modality_id,
                mask_X=descriptor.sim_mask,
                mask_W=node.mask.unsqueeze(0),
            ).item()
            if sim > best_sim:
                best_sim = sim
                matched_node = node

        threshold = similarity_threshold if similarity_threshold is not None else descriptor.match_threshold
        if matched_node is not None and best_sim >= threshold:
            matched_node.update_from_descriptor(descriptor)
            matched_node.count += 1
            matched_node.last_seen = time.time()
            return matched_node

        return gmem_i.add_node(
            descriptor.modality_id,
            descriptor.values.squeeze(0).detach().clone(),
            mask=descriptor.sim_mask.squeeze(0).detach().clone(),
            gate_context=dict(descriptor.gate_context),
            topology_context=dict(descriptor.topology_context),
            match_threshold=descriptor.match_threshold,
        )

    def _extract_and_match_local_features(self, X_subspaces: Dict[int, torch.Tensor],
                                          gmem_i: GmemoryI, x: int, y: int,
                                          similarity_threshold: Optional[float] = None) -> List[ModalityNode]:
        active_nodes = []
        for mod_id, X_m in X_subspaces.items():
            descriptor = self._build_descriptor(mod_id, X_m, x, y)
            if descriptor is None:
                continue
            active_nodes.append(self._match_or_create_descriptor(descriptor, gmem_i, similarity_threshold))
                
        return active_nodes

    # ---- Window keypoint sampling and immediate GmemII filling ----
    def _local_response_scale_min(self, support_dim: int, window_state: WindowState) -> float:
        r_ratio = max(window_state.r_eff / max(float(self.cfg.window_r_min), 1.0), 1.0)
        if support_dim == 2:
            base, max_v, eta = self.cfg.ell_base_dim2, self.cfg.ell_max_dim2, self.cfg.ell_eta_dim2
        elif support_dim == 1:
            base, max_v, eta = self.cfg.ell_base_dim1, self.cfg.ell_max_dim1, self.cfg.ell_eta_dim1
        else:
            base, max_v, eta = self.cfg.ell_base_dim0, self.cfg.ell_max_dim0, self.cfg.ell_eta_dim0
        return float(max(base, min(max_v, base * (r_ratio ** eta))))

    def _keypoint_quota(self, support_dim: int) -> int:
        if support_dim == 2:
            return int(self.cfg.max_keypoints_dim2)
        if support_dim == 1:
            return int(self.cfg.max_keypoints_dim1)
        return int(self.cfg.max_keypoints_dim0)

    def _select_nms_points(self, candidate_map: torch.Tensor, threshold: float,
                           max_count: int, min_distance: int) -> List[Tuple[int, int, float]]:
        if max_count <= 0:
            return []
        k = max(1, 2 * int(min_distance) + 1)
        pooled = F.max_pool2d(candidate_map, kernel_size=k, stride=1, padding=k // 2)
        peaks = (candidate_map >= pooled) & (candidate_map > threshold)
        indices = torch.nonzero(peaks[0, 0], as_tuple=False)
        if indices.numel() == 0:
            return []
        scores = candidate_map[0, 0, indices[:, 0], indices[:, 1]]
        order = torch.argsort(scores, descending=True)
        selected: List[Tuple[int, int, float]] = []
        for idx in order:
            y = int(indices[idx, 0].item())
            x = int(indices[idx, 1].item())
            score = float(scores[idx].item())
            too_close = any((x - px) ** 2 + (y - py) ** 2 < min_distance ** 2 for px, py, _ in selected)
            if too_close:
                continue
            selected.append((x, y, score))
            if len(selected) >= max_count:
                break
        return selected

    def _node_thresholds(self, node: ModalityNode) -> Tuple[float, float]:
        topology = self._node_topology(node)
        if topology in {'Surface_2D', 'Texture_Surface'}:
            return self.cfg.theta_surf_low, self.cfg.theta_surf_high
        if topology in {'Boundary_Edge', 'Trace_1D'}:
            return self.cfg.theta_edge_low, self.cfg.theta_edge_high
        return self.cfg.theta_window_low, self.cfg.theta_event

    def _sample_window_keypoints(self, X_subspaces: Dict[int, torch.Tensor], S_maps: Dict[int, torch.Tensor],
                                 gmem_i: GmemoryI, window_state: WindowState) -> List[KeypointCandidate]:
        candidates: List[KeypointCandidate] = []
        window = window_state.mask
        min_dist = int(self.cfg.keypoint_nms_radius)
        explained = self._explained_response_map(S_maps, window)

        for nid, S in S_maps.items():
            node = gmem_i.nodes.get(nid)
            if node is None:
                continue
            support_dim = self._support_dim_for_node(node)
            low, high = self._node_thresholds(node)
            quota = self._keypoint_quota(support_dim)
            confirm_map = S * window * (S > high).float()
            extend_map = S * window * (S > low).float() * (S <= high).float() * (1.0 - self.m_aversion.clamp(0.0, 1.0))
            for x, y, score in self._select_nms_points(confirm_map, high, quota, min_dist):
                candidates.append(KeypointCandidate(x, y, 'known_confirm', support_dim, node.modality_id, score, node_id=nid))
            for x, y, score in self._select_nms_points(extend_map, low, quota, min_dist):
                candidates.append(KeypointCandidate(x, y, 'known_extend', support_dim, node.modality_id, score, node_id=nid))

        for mod_id, X_m in X_subspaces.items():
            gate = SimilarityEngine.compute_gate_map(X_m, self.cfg, mod_id)
            novelty = gate * window * (1.0 - explained)
            support_dim = self._support_dim_for_topology(self._feature_spec(mod_id).get('topology', self._modality_topology(mod_id)))
            quota = self._keypoint_quota(support_dim)
            for x, y, score in self._select_nms_points(novelty, self.cfg.theta_novel, quota, min_dist):
                candidates.append(KeypointCandidate(x, y, 'novel_seed', support_dim, mod_id, score))

        for mod_id in (2, 3, 4):
            X_m = X_subspaces.get(mod_id)
            if X_m is None:
                continue
            event_map = self._normalize_map(X_m.abs().max(dim=1, keepdim=True).values, window) * window
            for x, y, score in self._select_nms_points(event_map, self.cfg.theta_event, self.cfg.max_keypoints_dim0, min_dist):
                candidates.append(KeypointCandidate(x, y, 'event_peak', 0, mod_id, score))

        filtered: List[KeypointCandidate] = []
        for kp in sorted(candidates, key=lambda item: item.score, reverse=True):
            ell_min = self._local_response_scale_min(kp.support_dim, window_state)
            if min_dist < ell_min and kp.kind not in {'event_peak', 'novel_seed'}:
                continue
            if any((kp.x - old.x) ** 2 + (kp.y - old.y) ** 2 < min_dist ** 2 for old in filtered):
                continue
            filtered.append(kp)
            if len(filtered) >= sum(self._keypoint_quota(d) for d in (0, 1, 2)):
                break
        self.last_keypoints = filtered
        return filtered

    def _choose_anchor_from_keypoints(self, matched: List[Tuple[KeypointCandidate, ModalityNode]]) -> Optional[Tuple[KeypointCandidate, ModalityNode]]:
        if not matched:
            return None
        surface = [item for item in matched if item[0].support_dim == 2]
        pool = surface if surface else matched
        cx, cy = self.fixation_point
        return max(pool, key=lambda item: item[0].score - 0.01 * math.sqrt((item[0].x - cx) ** 2 + (item[0].y - cy) ** 2))

    def _ensure_work_semantic_from_keypoints(self, matched: List[Tuple[KeypointCandidate, ModalityNode]],
                                             gmem_ii: GmemoryII) -> Optional[SemanticNode]:
        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is not None:
            return sem_node
        anchor = self._choose_anchor_from_keypoints(matched)
        if anchor is None:
            return None
        kp, node = anchor
        sem_node = gmem_ii.add_semantic_node(node.node_id)
        self.active_semantic_id = sem_node.node_id
        self.anchor_position = (kp.x, kp.y)
        self.semantic_mask = self.work_window.clone() if self.work_window is not None else None
        sem_node.is_completed = False
        return sem_node

    def _relation_role_for_keypoint(self, kp: KeypointCandidate, node: ModalityNode) -> str:
        if kp.kind == 'event_peak':
            return 'event'
        if kp.kind == 'known_extend':
            return 'extend'
        return self._relation_role_for_node(node)

    def _update_surface_extent(self, sem_node: SemanticNode, node: ModalityNode,
                               kp: KeypointCandidate, score: float, gmem_ii: GmemoryII):
        if self.anchor_position is None or self._support_dim_for_node(node) != 2:
            return
        ax, ay = self.anchor_position
        dx, dy = kp.x - ax, kp.y - ay
        rho = math.sqrt(dx * dx + dy * dy)
        if rho < 1e-6:
            return
        theta = math.atan2(dy, dx)
        bin_count = max(1, int(self.cfg.surface_theta_bins))
        theta_norm = (theta + math.pi) / (2 * math.pi)
        bin_idx = int(theta_norm * bin_count) % bin_count
        extent = sem_node.surface_extents[node.node_id].get(bin_idx)
        if extent is None:
            sem_node.surface_extents[node.node_id][bin_idx] = {
                'rho_max': rho,
                'confidence': score,
                'count': 1,
            }
            gmem_ii.add_relation_edge(
                sem_node,
                node.node_id,
                node.node_id,
                math.log(rho + 1e-5),
                theta,
                role='self_extent',
                span=rho,
                area=max(1.0, rho),
            )
            return
        old_rho = float(extent.get('rho_max', 0.0))
        alpha = float(self.cfg.self_extent_alpha)
        extent['confidence'] = (1.0 - alpha) * float(extent.get('confidence', 0.0)) + alpha * score
        extent['count'] = int(extent.get('count', 0)) + 1
        if rho - old_rho > self.cfg.theta_surface_rho_expand:
            extent['rho_max'] = rho
            gmem_ii.add_relation_edge(
                sem_node,
                node.node_id,
                node.node_id,
                math.log(rho + 1e-5),
                theta,
                role='self_extent',
                span=rho,
                area=max(1.0, rho),
            )

    def _write_keypoint_relation(self, sem_node: SemanticNode, kp: KeypointCandidate,
                                 node: ModalityNode, gmem_ii: GmemoryII):
        sem_node.child_node_ids.add(node.node_id)
        if self.anchor_position is None:
            self.anchor_position = (kp.x, kp.y)
        ax, ay = self.anchor_position
        dx, dy = kp.x - ax, kp.y - ay
        r = math.sqrt(dx * dx + dy * dy) + 1e-5
        rho = math.log(r)
        theta = math.atan2(dy, dx)
        role = self._relation_role_for_keypoint(kp, node)
        if node.node_id != sem_node.anchor_id:
            gmem_ii.add_relation_edge(
                sem_node,
                sem_node.anchor_id,
                node.node_id,
                rho,
                theta,
                role=role,
                span=r,
                area=max(1.0, kp.score),
            )
        self._update_surface_extent(sem_node, node, kp, kp.score, gmem_ii)

    def _write_window_keypoints(self, keypoints: List[KeypointCandidate], X_subspaces: Dict[int, torch.Tensor],
                                gmem_i: GmemoryI, gmem_ii: GmemoryII) -> List[ModalityNode]:
        matched: List[Tuple[KeypointCandidate, ModalityNode]] = []
        for kp in keypoints:
            X_m = X_subspaces.get(kp.modality_id)
            if X_m is None:
                continue
            descriptor = self._build_descriptor(kp.modality_id, X_m, kp.x, kp.y)
            if descriptor is None:
                continue
            node = self._match_or_create_descriptor(descriptor, gmem_i)
            kp.node_id = node.node_id
            matched.append((kp, node))

        sem_node = self._ensure_work_semantic_from_keypoints(matched, gmem_ii)
        if sem_node is None:
            return [node for _, node in matched]

        for kp, node in matched:
            self._write_keypoint_relation(sem_node, kp, node, gmem_ii)
        for prev in self.prev_keypoints:
            if prev.node_id is None:
                continue
            for kp, node in matched:
                dx, dy = kp.x - prev.x, kp.y - prev.y
                span = math.sqrt(dx * dx + dy * dy)
                if span < self.cfg.self_edge_min_span:
                    continue
                gmem_ii.add_relation_edge(
                    sem_node,
                    prev.node_id,
                    node.node_id,
                    math.log(span + 1e-5),
                    math.atan2(dy, dx),
                    role='temporal_adjacent',
                    span=span,
                    area=max(1.0, kp.score),
                )
        self.prev_keypoints = list(keypoints)
        sem_node.texture_density = 0.9 * sem_node.texture_density + 0.1 * len(keypoints)
        sem_node.detail_pressure = 0.9 * sem_node.detail_pressure + 0.1 * self.current_state_observation.novelty
        sem_node.frontier_strength = self.current_state_observation.frontier
        sem_node.gap_strength = self.current_state_observation.gap
        return [node for _, node in matched]

    def _modality_topology(self, modality_id: int) -> str:
        spec = self._feature_spec(modality_id)
        if spec.get('topology'):
            return spec['topology']
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
        def needs_reset(value: Optional[torch.Tensor]) -> bool:
            return value is None or value.shape[-2] != H or value.shape[-1] != W

        if needs_reset(self.ior_map):
            self.ior_map = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.m_aversion):
            self.m_aversion = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.semantic_history):
            self.semantic_history = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.boundary_visited):
            self.boundary_visited = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.boundary_rejected):
            self.boundary_rejected = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.boundary_suppression):
            self.boundary_suppression = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.dynamic_matrix):
            self.dynamic_matrix = torch.zeros((1, 1, H, W), device=self.device)
        if needs_reset(self.work_window):
            self.work_window = self._circle_mask(self.fixation_point, self.cfg.window_r_min, H, W)

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
            mask = self.optimizer.valid_view_mask(drive)
            valid = torch.nonzero(mask[0, 0] > 0.0, as_tuple=False)
            if valid.numel() > 0:
                ridx = torch.randint(0, valid.shape[0], (1,), device=self.device).item()
                yx = valid[ridx]
                return int(yx[1].item()), int(yx[0].item())
            idx = torch.randint(0, H * W, (1,), device=self.device).item()
        else:
            idx = torch.argmax(drive).item()
        return int(idx % W), int(idx // W)

    def _suppress_invalid_view(self, drive: torch.Tensor, invalid_value: float = 0.0) -> torch.Tensor:
        return self.optimizer.suppress_invalid_view(drive, invalid_value=invalid_value)

    def _mark_map_visited(self, target: torch.Tensor, pt: Tuple[int, int], sigma: float, weight: float = 1.0) -> torch.Tensor:
        H, W = target.shape[-2], target.shape[-1]
        dist_sq = self._distance_sq_from(pt, H, W)
        mark = weight * torch.exp(-dist_sq / (2 * sigma ** 2)).view(1, 1, H, W)
        return torch.max(target, mark.clamp(max=1.0))

    def _local_window_mean(self, value: torch.Tensor, pt: Tuple[int, int], radius: int) -> float:
        x, y = pt
        H, W = value.shape[-2], value.shape[-1]
        r = max(0, int(radius))
        x1, x2 = max(0, x - r), min(W, x + r + 1)
        y1, y2 = max(0, y - r), min(H, y + r + 1)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        return float(value[:, :, y1:y2, x1:x2].mean().item())

    def _local_window_max(self, value: torch.Tensor, pt: Tuple[int, int], radius: int) -> float:
        x, y = pt
        H, W = value.shape[-2], value.shape[-1]
        r = max(0, int(radius))
        x1, x2 = max(0, x - r), min(W, x + r + 1)
        y1, y2 = max(0, y - r), min(H, y + r + 1)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        return float(value[:, :, y1:y2, x1:x2].max().item())

    def _micro_contour_response(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = fields['boundary']
        contour = (
            self.cfg.micro_w_edge * fields.get('edge', torch.zeros_like(ref))
            + self.cfg.micro_w_corner * fields.get('corner', torch.zeros_like(ref))
            + self.cfg.micro_w_junction * fields.get('junction', torch.zeros_like(ref))
            + self.cfg.micro_w_terminator * fields.get('terminator', torch.zeros_like(ref))
        )
        return self._normalize_map(contour, ref)

    def _tangent_gate_at(self, ref: torch.Tensor, pt: Tuple[int, int],
                         tangent: Optional[Tuple[float, float]]) -> torch.Tensor:
        if tangent is None:
            return torch.ones_like(ref)
        tx, ty = tangent
        H, W = ref.shape[-2], ref.shape[-1]
        cx, cy = pt
        x_grid, y_grid = self._xy_grids(H, W)
        vx = (x_grid - cx).float()
        vy = (y_grid - cy).float()
        norm = torch.sqrt(vx ** 2 + vy ** 2 + 1e-6)
        cosang = ((vx * tx + vy * ty) / norm).clamp(-1.0, 1.0)
        angle = torch.acos(cosang)
        dist_gate = torch.exp(-((norm - self.cfg.trace_step) ** 2) / (2 * self.cfg.sigma_trace_step ** 2))
        angle_gate = torch.exp(-(angle ** 2) / (2 * self.cfg.sigma_trace_angle ** 2))
        return (dist_gate * angle_gate).view(1, 1, H, W)

    def _boundary_hit_confidence_at(self, fields: Dict[str, torch.Tensor],
                                    pt: Optional[Tuple[int, int]] = None) -> float:
        point = pt if pt is not None else self.fixation_point
        return self._local_window_mean(fields['boundary'], point, self.cfg.boundary_hit_radius)

    def _estimate_boundary_tangent(self, fields: Dict[str, torch.Tensor],
                                   pt: Optional[Tuple[int, int]] = None) -> Optional[Tuple[float, float]]:
        point = pt if pt is not None else self.fixation_point
        x, y = point
        source = fields.get('mask', fields['boundary'])
        H, W = source.shape[-2], source.shape[-1]
        x0, x1 = max(0, x - 1), min(W - 1, x + 1)
        y0, y1 = max(0, y - 1), min(H - 1, y + 1)
        gx = float(source[0, 0, y, x1].item() - source[0, 0, y, x0].item())
        gy = float(source[0, 0, y1, x].item() - source[0, 0, y0, x].item())
        norm = math.sqrt(gx * gx + gy * gy)

        if norm <= 1e-6 and self.anchor_position is not None:
            ax, ay = self.anchor_position
            gx = float(x - ax)
            gy = float(y - ay)
            norm = math.sqrt(gx * gx + gy * gy)
        if norm <= 1e-6:
            return self.last_boundary_tangent

        nx, ny = gx / norm, gy / norm
        tangent = (-ny, nx)
        if self.last_boundary_tangent is not None:
            dot = tangent[0] * self.last_boundary_tangent[0] + tangent[1] * self.last_boundary_tangent[1]
            if dot < 0:
                tangent = (-tangent[0], -tangent[1])
        return tangent

    def _boundary_forward_support_at(self, fields: Dict[str, torch.Tensor],
                                     tangent: Optional[Tuple[float, float]],
                                     pt: Optional[Tuple[int, int]] = None) -> float:
        point = pt if pt is not None else self.fixation_point
        contour = self._micro_contour_response(fields)
        gate = self._tangent_gate_at(fields['boundary'], point, tangent)
        support = fields['boundary'] * contour * gate
        return float(support.max().item())

    def _boundary_role_confidence_at(self, fields: Dict[str, torch.Tensor],
                                     pt: Optional[Tuple[int, int]] = None) -> float:
        point = pt if pt is not None else self.fixation_point
        ref = fields['boundary']
        role_map = (
            self.cfg.micro_w_corner * fields.get('corner', torch.zeros_like(ref))
            + self.cfg.micro_w_junction * fields.get('junction', torch.zeros_like(ref))
            + self.cfg.micro_w_terminator * fields.get('terminator', torch.zeros_like(ref))
        )
        if role_map.max() <= 1e-6:
            return 0.0
        role_map = self._normalize_map(role_map, ref)
        return self._local_window_max(role_map, point, self.cfg.boundary_hit_radius)

    def _best_boundary_tangent(self, fields: Dict[str, torch.Tensor],
                               pt: Optional[Tuple[int, int]] = None) -> Tuple[Optional[Tuple[float, float]], float]:
        tangent = self._estimate_boundary_tangent(fields, pt)
        if tangent is None:
            return None, self._boundary_forward_support_at(fields, None, pt)
        forward = self._boundary_forward_support_at(fields, tangent, pt)
        reverse_tangent = (-tangent[0], -tangent[1])
        reverse = self._boundary_forward_support_at(fields, reverse_tangent, pt)
        if reverse > forward:
            return reverse_tangent, reverse
        return tangent, forward

    def _pending_boundary_arrived(self) -> bool:
        if self.pending_boundary_target is None:
            return False
        tx, ty = self.pending_boundary_target
        cx, cy = self.fixation_point
        dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        return dist <= self.cfg.boundary_pending_radius

    def _try_enter_contour_trace_from_seek(self, fields: Dict[str, torch.Tensor]) -> bool:
        if not self._pending_boundary_arrived():
            return False

        point = self.fixation_point
        hit = self._boundary_hit_confidence_at(fields, point)
        tangent, fwd = self._best_boundary_tangent(fields, point)
        self.boundary_hit_confidence = hit
        self.boundary_forward_support = fwd
        self.pending_boundary_target = None

        if hit > self.cfg.theta_boundary_hit_enter and fwd > self.cfg.theta_trace_fwd_enter:
            self.micro_intention = MicroIntention.CONTOUR_TRACE
            self.last_boundary_tangent = tangent
            self.trace_age = 0
            self.boundary_start_point = point
            return True

        if self.boundary_rejected is not None:
            self.boundary_rejected = self._mark_map_visited(
                self.boundary_rejected,
                point,
                self.cfg.sigma_boundary_reject,
                1.0,
            )
        return False

    def _mark_boundary_trace_progress(self, pt: Tuple[int, int]):
        if self.boundary_visited is not None:
            self.boundary_visited = self._mark_map_visited(
                self.boundary_visited,
                pt,
                self.cfg.sigma_boundary_visit,
                1.0,
            )
        if self.boundary_suppression is not None:
            self.boundary_suppression = self._mark_map_visited(
                self.boundary_suppression,
                pt,
                self.cfg.sigma_boundary_suppression,
                1.0,
            )

    # ---- Dynamic local potential and vector guidance ----
    def _vector_step_bounds(self, window_state: WindowState) -> Tuple[float, float]:
        s_min = max(1.0, float(self.cfg.kappa_step_min) * window_state.r_eff)
        s_max = max(s_min + 1.0, float(self.cfg.kappa_step_max) * window_state.r_eff)
        return s_min, s_max

    def _vector_candidate_mask(self, ref: torch.Tensor, window_state: WindowState) -> torch.Tensor:
        H, W = ref.shape[-2], ref.shape[-1]
        s_min, s_max = self._vector_step_bounds(window_state)
        dist = torch.sqrt(self._distance_sq_from(self.fixation_point, H, W).float() + 1e-6).view(1, 1, H, W)
        ring = ((dist >= s_min) & (dist <= s_max)).float()
        delta = max(1, int(float(self.cfg.kappa_surface_delta) * window_state.r_eff))
        omega = self._morph_dilate(window_state.mask, delta) * self._valid_view_mask_like(ref)
        boundary = self._mask_boundary(window_state.mask, max(1, int(0.06 * window_state.r_eff)))
        candidate = omega * torch.max(boundary, ring)
        return candidate.clamp(0.0, 1.0)

    def _weighted_vector_from_map(self, value: torch.Tensor, window_state: WindowState,
                                  candidate_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        ref = value
        H, W = ref.shape[-2], ref.shape[-1]
        cx, cy = self.fixation_point
        if candidate_mask is None:
            candidate_mask = self._vector_candidate_mask(ref, window_state)
        weights = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        weights = weights * candidate_mask
        den = float(torch.sum(weights).item())
        if den <= 1e-6:
            return 0.0, 0.0
        x_grid, y_grid = self._xy_grids(H, W)
        vx = float(torch.sum(weights[0, 0] * (x_grid - cx).float()).item() / den)
        vy = float(torch.sum(weights[0, 0] * (y_grid - cy).float()).item() / den)
        return vx, vy

    def _clip_vector_step(self, vector: Tuple[float, float], window_state: WindowState) -> Tuple[float, float]:
        vx, vy = vector
        norm = math.sqrt(vx * vx + vy * vy)
        if norm <= 1e-6:
            return 0.0, 0.0
        s_min, s_max = self._vector_step_bounds(window_state)
        target_norm = max(s_min, min(s_max, norm))
        scale = target_norm / norm
        return vx * scale, vy * scale

    def _clamp_point_to_valid(self, pt: Tuple[int, int], ref: torch.Tensor) -> Tuple[int, int]:
        H, W = ref.shape[-2], ref.shape[-1]
        y1, y2, x1, x2 = self.optimizer.valid_view_bounds(H, W)
        x = int(max(x1, min(pt[0], max(x1, x2 - 1))))
        y = int(max(y1, min(pt[1], max(y1, y2 - 1))))
        return x, y

    def _align_tangent(self, tangent: Tuple[float, float]) -> Tuple[float, float]:
        tx, ty = tangent
        norm = math.sqrt(tx * tx + ty * ty)
        if norm <= 1e-6:
            return tangent
        tx, ty = tx / norm, ty / norm
        ref = self.prev_local_vector
        if math.sqrt(ref[0] * ref[0] + ref[1] * ref[1]) <= 1e-6 and self.last_boundary_tangent is not None:
            ref = self.last_boundary_tangent
        if ref[0] * tx + ref[1] * ty < 0:
            tx, ty = -tx, -ty
        return tx, ty

    def _estimate_edge_tangent_from_inputs(self, X_subspaces: Dict[int, torch.Tensor],
                                           window_state: WindowState) -> Optional[Tuple[float, float]]:
        X_ori = X_subspaces.get(4)
        if X_ori is not None and X_ori.shape[1] > 0:
            angle = X_ori[:, 0:1]
            weight = SimilarityEngine.compute_gate_map(X_ori, self.cfg, 4) * window_state.mask
            den = float(torch.sum(weight).item())
            if den > 1e-6:
                sx = float(torch.sum(torch.cos(2.0 * angle) * weight).item() / den)
                sy = float(torch.sum(torch.sin(2.0 * angle) * weight).item() / den)
                theta = 0.5 * math.atan2(sy, sx)
                return self._align_tangent((math.cos(theta), math.sin(theta)))

        X_grad = X_subspaces.get(0)
        if X_grad is not None and X_grad.shape[1] > 0:
            if X_grad.shape[1] >= 4:
                strength = X_grad[:, 0:1].abs()
                normal_theta = X_grad[:, 1:2]
            elif X_grad.shape[1] >= 2:
                gx, gy = X_grad[:, 0:1], X_grad[:, 1:2]
                strength = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
                normal_theta = torch.atan2(gy, gx)
            else:
                strength = X_grad[:, 0:1].abs()
                normal_theta = torch.zeros_like(strength)
            weight = self._normalize_map(strength, window_state.mask) * window_state.mask
            den = float(torch.sum(weight).item())
            if den > 1e-6:
                tangent_theta = normal_theta + math.pi / 2.0
                sx = float(torch.sum(torch.cos(2.0 * tangent_theta) * weight).item() / den)
                sy = float(torch.sum(torch.sin(2.0 * tangent_theta) * weight).item() / den)
                theta = 0.5 * math.atan2(sy, sx)
                return self._align_tangent((math.cos(theta), math.sin(theta)))

        vx, vy = self.prev_local_vector
        if math.sqrt(vx * vx + vy * vy) > 1e-6:
            return self._align_tangent((vx, vy))
        return self.last_boundary_tangent

    def _butterfly_potential(self, X_subspaces: Dict[int, torch.Tensor], window_state: WindowState,
                             S_maps: Dict[int, torch.Tensor], gmem_i: GmemoryI) -> torch.Tensor:
        ref = window_state.mask
        edge_support = torch.zeros_like(ref)
        for nid, S in S_maps.items():
            node = gmem_i.nodes.get(nid)
            if node is not None and self._support_dim_for_node(node) == 1:
                edge_support = torch.max(edge_support, self._normalize_map(S, ref))
        if edge_support.max() <= 1e-6 and self.optimizer.I_str is not None:
            edge_support = self._normalize_map(self.optimizer.I_str, ref)
        edge_inside = edge_support * window_state.mask
        den = float(torch.sum(edge_inside).item())
        if den <= 1e-6:
            return torch.zeros_like(ref)

        x_grid, y_grid = self._xy_grids(ref.shape[-2], ref.shape[-1])
        ex = float(torch.sum(edge_inside[0, 0] * x_grid.float()).item() / den)
        ey = float(torch.sum(edge_inside[0, 0] * y_grid.float()).item() / den)
        tangent = self._estimate_edge_tangent_from_inputs(X_subspaces, window_state)
        if tangent is None:
            return torch.zeros_like(ref)

        theta = math.atan2(tangent[1], tangent[0])
        dx = x_grid.float() - ex
        dy = y_grid.float() - ey
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        angle = torch.atan2(dy, dx)
        step = max(1.0, float(self.cfg.kappa_edge_step) * window_state.r_eff)
        sigma_r = max(2.0, 0.30 * step)
        sigma_theta = max(0.15, float(self.cfg.sigma_trace_angle))
        da_fwd = torch.atan2(torch.sin(angle - theta), torch.cos(angle - theta))
        da_bwd = torch.atan2(torch.sin(angle - theta - math.pi), torch.cos(angle - theta - math.pi))
        radial = torch.exp(-((dist - step) ** 2) / (2.0 * sigma_r ** 2))
        angular = torch.exp(-(da_fwd ** 2) / (2.0 * sigma_theta ** 2)) + torch.exp(-(da_bwd ** 2) / (2.0 * sigma_theta ** 2))
        low_band = ((edge_support > self.cfg.theta_edge_low) & (edge_support < self.cfg.theta_edge_high)).float()
        high_band = (edge_support >= self.cfg.theta_edge_high).float()
        support_gate = low_band + 0.35 * high_band
        potential = radial.view(1, 1, *dist.shape) * angular.view(1, 1, *dist.shape)
        potential = potential * support_gate * (1.0 - self.m_aversion.clamp(0.0, 1.0))
        potential = potential * self._valid_view_mask_like(ref)
        return self._normalize_map(potential, ref)

    def _surface_potential(self, window_state: WindowState, S_maps: Dict[int, torch.Tensor],
                           gmem_i: GmemoryI) -> torch.Tensor:
        ref = window_state.mask
        surface_support = torch.zeros_like(ref)
        for nid, S in S_maps.items():
            node = gmem_i.nodes.get(nid)
            if node is not None and self._support_dim_for_node(node) == 2:
                surface_support = torch.max(surface_support, self._normalize_map(S, ref))
        if surface_support.max() <= 1e-6 and self.optimizer.I_con1 is not None:
            surface_support = self._normalize_map(self.optimizer.I_con1, ref)

        observed = self.m_aversion.clamp(0.0, 1.0) * window_state.mask
        if observed.max() <= 1e-6:
            observed = self._circle_mask(self.fixation_point, self.cfg.window_seed_radius, ref.shape[-2], ref.shape[-1])
            observed = observed.to(ref.device) * window_state.mask
        delta = max(1, int(float(self.cfg.kappa_surface_delta) * window_state.r_eff))
        shell = (self._morph_dilate(observed, delta) - observed).clamp(0.0, 1.0)
        low_band = ((surface_support > self.cfg.theta_surf_low) & (surface_support < self.cfg.theta_surf_high)).float()
        high_band = (surface_support >= self.cfg.theta_surf_high).float()
        boundary = self._mask_boundary(window_state.mask, 1)
        support = surface_support + 0.35 * low_band + 0.15 * high_band
        potential = (shell + 0.35 * boundary) * support
        potential = potential * (1.0 - window_state.barrier.clamp(0.0, 1.0))
        potential = potential * (1.0 - self.m_aversion.clamp(0.0, 1.0)) * self._valid_view_mask_like(ref)
        return self._normalize_map(potential, ref)

    def _event_potential(self, X_subspaces: Dict[int, torch.Tensor], window_state: WindowState) -> torch.Tensor:
        ref = window_state.mask
        event = torch.zeros_like(ref)
        for mod_id in (2, 3, 4):
            X_m = X_subspaces.get(mod_id)
            if X_m is None or X_m.shape[1] <= 0:
                continue
            gate = SimilarityEngine.compute_gate_map(X_m, self.cfg, mod_id)
            local = self._normalize_map(X_m.abs().max(dim=1, keepdim=True).values, ref)
            event = torch.max(event, gate * local)
        event = event * (event > self.cfg.theta_event).float()
        event = event * (1.0 - self.m_aversion.clamp(0.0, 1.0)) * window_state.mask
        return self._normalize_map(event, ref)

    def _gap_potential(self, window_state: WindowState, gmem_ii: GmemoryII,
                       S_maps: Dict[int, torch.Tensor]) -> torch.Tensor:
        ref = window_state.mask
        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is None or self.anchor_position is None:
            return torch.zeros_like(ref)
        ax, ay = self.anchor_position
        H, W = ref.shape[-2], ref.shape[-1]
        x_grid, y_grid = self._xy_grids(H, W)
        potential = torch.zeros_like(ref)
        sigma = max(3.0, 0.45 * window_state.r_eff)
        for edge in sem_node.iter_relation_edges():
            if edge.src_id == edge.dst_id:
                continue
            missing = max(0.0, 1.0 - float(edge.confidence))
            if edge.dst_id in S_maps:
                observed = float(torch.max(S_maps[edge.dst_id] * window_state.mask).item())
                missing = max(missing, max(0.0, 0.35 - observed))
            if missing <= 1e-6:
                continue
            pred_r = math.exp(edge.rho_offset)
            pred_x = ax + pred_r * math.cos(edge.theta_offset)
            pred_y = ay + pred_r * math.sin(edge.theta_offset)
            dist_sq = (x_grid.float() - pred_x) ** 2 + (y_grid.float() - pred_y) ** 2
            compat = S_maps.get(edge.dst_id, torch.ones_like(ref))
            compat = torch.clamp(self._normalize_map(compat, ref), min=0.15)
            potential += missing * torch.exp(-dist_sq / (2.0 * sigma ** 2)).view(1, 1, H, W) * compat
        return self._suppress_invalid_view(self._normalize_map(potential, ref), invalid_value=0.0)

    def _update_dynamic_potential(self, X_subspaces: Dict[int, torch.Tensor],
                                  S_maps: Dict[int, torch.Tensor], gmem_i: GmemoryI,
                                  gmem_ii: GmemoryII, window_state: WindowState) -> torch.Tensor:
        ref = window_state.mask
        if self.dynamic_matrix is None or self.dynamic_matrix.shape[-2:] != ref.shape[-2:]:
            self.dynamic_matrix = torch.zeros_like(ref)
        edge = self._butterfly_potential(X_subspaces, window_state, S_maps, gmem_i)
        surface = self._surface_potential(window_state, S_maps, gmem_i)
        event = self._event_potential(X_subspaces, window_state)
        gap = self._gap_potential(window_state, gmem_ii, S_maps)
        ret = self.m_aversion.clamp(0.0, 1.0) * self._valid_view_mask_like(ref)
        local = self._morph_dilate(window_state.mask, max(1, int(float(self.cfg.kappa_surface_delta) * window_state.r_eff)))
        local = local * self._valid_view_mask_like(ref)
        old = self.dynamic_matrix * float(self.cfg.dynamic_decay)
        update = (
            self.cfg.dynamic_w_edge * edge
            + self.cfg.dynamic_w_surface * surface
            + self.cfg.dynamic_w_event * event
            + self.cfg.dynamic_w_gap * gap
            - self.cfg.dynamic_w_return * ret
        )
        self.dynamic_matrix = torch.where(local > 0.0, (old + update).clamp(0.0, 1.0), old.clamp(0.0, 1.0))
        self.matrix1 = self.dynamic_matrix
        self.debug_optimizer['potential_components'] = {
            'edge': float(edge.max().item()),
            'surface': float(surface.max().item()),
            'event': float(event.max().item()),
            'gap': float(gap.max().item()),
            'return': float(ret.max().item()),
        }
        return self.dynamic_matrix

    def _low_gpos_vector(self, S_maps: Dict[int, torch.Tensor], gmem_i: GmemoryI,
                         window_state: WindowState) -> Tuple[float, float]:
        ref = window_state.mask
        low = torch.zeros_like(ref)
        for nid, S in S_maps.items():
            node = gmem_i.nodes.get(nid)
            if node is None:
                continue
            low_theta, high_theta = self._node_thresholds(node)
            band = S * ((S > low_theta) & (S < high_theta)).float()
            if self._support_dim_for_node(node) in {1, 2}:
                low = torch.max(low, self._normalize_map(band, ref))
        return self._weighted_vector_from_map(low, window_state)

    def _novel_vector(self, X_subspaces: Dict[int, torch.Tensor], S_maps: Dict[int, torch.Tensor],
                      window_state: WindowState) -> Tuple[float, float]:
        raw = self._raw_response_map(X_subspaces)
        explained = self._explained_response_map(S_maps, raw)
        novelty = (raw * (1.0 - explained)).clamp(0.0, 1.0) * window_state.mask
        return self._weighted_vector_from_map(novelty, window_state)

    def _gap_vector(self, window_state: WindowState, gmem_ii: GmemoryII,
                    S_maps: Dict[int, torch.Tensor]) -> Tuple[float, float]:
        return self._weighted_vector_from_map(self._gap_potential(window_state, gmem_ii, S_maps), window_state)

    def _return_vector(self, window_state: WindowState) -> Tuple[float, float]:
        return self._weighted_vector_from_map(self.m_aversion.clamp(0.0, 1.0), window_state)

    def _compose_local_vector(self, X_subspaces: Dict[int, torch.Tensor], S_maps: Dict[int, torch.Tensor],
                              gmem_i: GmemoryI, gmem_ii: GmemoryII,
                              window_state: WindowState, potential: torch.Tensor) -> Tuple[Tuple[float, float], Dict[str, Tuple[float, float]]]:
        v_p = self._weighted_vector_from_map(potential, window_state)
        v_low = self._low_gpos_vector(S_maps, gmem_i, window_state)
        v_gap = self._gap_vector(window_state, gmem_ii, S_maps)
        v_novel = self._novel_vector(X_subspaces, S_maps, window_state)
        v_return = self._return_vector(window_state)
        v_inertia = self.prev_local_vector
        vx = (
            self.cfg.vector_w_potential * v_p[0]
            + self.cfg.vector_w_low_gpos * v_low[0]
            + self.cfg.vector_w_gap * v_gap[0]
            + self.cfg.vector_w_novel * v_novel[0]
            + self.cfg.vector_w_inertia * v_inertia[0]
            - self.cfg.vector_w_return * v_return[0]
        )
        vy = (
            self.cfg.vector_w_potential * v_p[1]
            + self.cfg.vector_w_low_gpos * v_low[1]
            + self.cfg.vector_w_gap * v_gap[1]
            + self.cfg.vector_w_novel * v_novel[1]
            + self.cfg.vector_w_inertia * v_inertia[1]
            - self.cfg.vector_w_return * v_return[1]
        )
        parts = {
            'potential': v_p,
            'low_gpos': v_low,
            'gap': v_gap,
            'novel': v_novel,
            'inertia': v_inertia,
            'return': v_return,
        }
        return (vx, vy), parts

    def _decide_next_saccade_micro_vector(self, X_subspaces: Dict[int, torch.Tensor],
                                          S_maps: Dict[int, torch.Tensor], gmem_i: GmemoryI,
                                          gmem_ii: GmemoryII,
                                          window_state: WindowState) -> VectorDecision:
        self._mark_micro_fixation()
        potential = self._update_dynamic_potential(X_subspaces, S_maps, gmem_i, gmem_ii, window_state)
        vector, parts = self._compose_local_vector(X_subspaces, S_maps, gmem_i, gmem_ii, window_state, potential)
        norm = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
        self.debug_optimizer['vector_components'] = parts
        self.debug_optimizer['vector_norm'] = norm

        if norm < float(self.cfg.theta_vector_norm):
            self.fail_energy = float(self.cfg.fail_decay) * self.fail_energy + float(self.cfg.fail_increment)
            return VectorDecision(vector=vector, target=self.fixation_point, norm=norm, potential=potential)

        step_x, step_y = self._clip_vector_step(vector, window_state)
        target = (int(round(self.fixation_point[0] + step_x)), int(round(self.fixation_point[1] + step_y)))
        target = self._clamp_point_to_valid(target, potential)
        step_norm = math.sqrt(step_x * step_x + step_y * step_y)
        if step_norm > 1e-6:
            self.prev_local_vector = (step_x / step_norm, step_y / step_norm)
            self.last_boundary_tangent = self.prev_local_vector
        self.fail_energy = float(self.cfg.fail_decay) * self.fail_energy
        return VectorDecision(vector=vector, target=target, norm=norm, potential=potential)

    def _ensure_work_semantic_from_local_nodes(self, local_nodes: List[ModalityNode],
                                               gmem_ii: GmemoryII,
                                               window_state: WindowState) -> Optional[SemanticNode]:
        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is not None:
            return sem_node
        if not local_nodes:
            return None
        anchor_node = max(local_nodes, key=lambda node: self._support_dim_for_node(node))
        sem_node = gmem_ii.add_semantic_node(anchor_node.node_id)
        self.active_semantic_id = sem_node.node_id
        self.anchor_position = self.fixation_point
        self.semantic_mask = window_state.mask.clone()
        sem_node.is_completed = False
        for node in local_nodes:
            sem_node.child_node_ids.add(node.node_id)
        return sem_node

    def _record_local_nodes_in_work_semantic(self, local_nodes: List[ModalityNode],
                                             gmem_ii: GmemoryII,
                                             window_state: WindowState):
        sem_node = self._ensure_work_semantic_from_local_nodes(local_nodes, gmem_ii, window_state)
        if sem_node is None:
            return
        for node in local_nodes:
            kp = KeypointCandidate(
                x=self.fixation_point[0],
                y=self.fixation_point[1],
                kind='known_confirm',
                support_dim=self._support_dim_for_node(node),
                modality_id=node.modality_id,
                score=1.0,
                node_id=node.node_id,
            )
            self._write_keypoint_relation(sem_node, kp, node, gmem_ii)

    def _maybe_complete_work(self, obs: StateObservation, gmem_i: GmemoryI,
                             gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII') -> bool:
        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is None:
            return False
        if (
            obs.completion > self.cfg.theta_complete
            and obs.frontier < self.cfg.theta_frontier_close
            and obs.novelty < self.cfg.theta_novel_close
            and obs.gap < self.cfg.theta_gap_close
        ):
            sem_node.frontier_strength = obs.frontier
            sem_node.gap_strength = obs.gap
            self._transition_to_macro(gmem_i, gmem_ii, gmem_iii, exit_type='completed')
            self._set_behavior_state(OptimizerState.MACRO_SEARCH)
            return True
        return False

    def _reset_micro_context(self, H: int, W: int, intention: MicroIntention = MicroIntention.SURFACE_CONFIRM):
        self.micro_intention = intention
        self.micro_repeat_count = 0
        self.interior_sample_count = 0
        self.color_mean = None
        self.color_m2 = None
        self.color_residual = 1.0
        self.boundary_coverage = 0.0
        self.boundary_visited = torch.zeros((1, 1, H, W), device=self.device)
        self.boundary_rejected = torch.zeros((1, 1, H, W), device=self.device)
        self.boundary_suppression = torch.zeros((1, 1, H, W), device=self.device)
        self.last_boundary_tangent = None
        self.boundary_start_point = None
        self.pending_boundary_target = None
        self.trace_age = 0
        self.boundary_hit_confidence = 0.0
        self.boundary_forward_support = 0.0
        self.boundary_role_confidence = 0.0
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
        if self.boundary_rejected is not None:
            self.boundary_rejected = self.boundary_rejected * self.cfg.boundary_reject_decay
        coverage_suppression = self.cfg.boundary_coverage_suppression * self.boundary_visited
        rejected = self.boundary_rejected if self.boundary_rejected is not None else torch.zeros_like(boundary)
        suppression = torch.max(coverage_suppression, rejected)
        unseen = (1.0 - suppression).clamp(0.0, 1.0)
        drive = (
            self.cfg.micro_w_boundary * boundary
            + self.cfg.micro_w_unseen_boundary * boundary * unseen
        )
        drive = drive * self._micro_span_gate(boundary) * unseen * self._micro_seek_gate(boundary)
        return drive

    def _micro_tangent_gate(self, ref: torch.Tensor) -> torch.Tensor:
        return self._tangent_gate_at(ref, self.fixation_point, self.last_boundary_tangent)

    def _micro_contour_trace_target(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        boundary = fields['boundary']
        contour = self._micro_contour_response(fields)
        suppression = self.boundary_suppression if self.boundary_suppression is not None else torch.zeros_like(boundary)
        drive = boundary * contour * self._micro_tangent_gate(boundary) * (1.0 - suppression).clamp(0.0, 1.0)
        hit = self._boundary_hit_confidence_at(fields)
        fwd = self._boundary_forward_support_at(fields, self.last_boundary_tangent)
        role = self._boundary_role_confidence_at(fields)
        self.boundary_hit_confidence = hit
        self.boundary_forward_support = fwd
        self.boundary_role_confidence = role

        if self.boundary_coverage >= self.cfg.theta_bd_cover and self.color_residual < self.cfg.theta_res:
            self.micro_intention = MicroIntention.RESUME_OR_EXIT
        elif (
            self.trace_age >= self.cfg.trace_min_age
            and fwd < self.cfg.theta_trace_fwd_exit
            and hit < self.cfg.theta_boundary_hit_exit
            and role < self.cfg.theta_boundary_role
        ):
            self.micro_intention = MicroIntention.BOUNDARY_SEEK
            self.pending_boundary_target = None
            self.trace_age = 0
        elif role >= self.cfg.theta_boundary_role and fwd < self.cfg.theta_trace_fwd_exit:
            tangent, _ = self._best_boundary_tangent(fields)
            self.last_boundary_tangent = tangent
            self.trace_age = 0
        else:
            self.trace_age += 1
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

        if self.micro_intention == MicroIntention.BOUNDARY_SEEK:
            self._try_enter_contour_trace_from_seek(fields)

        intention_for_drive = self.micro_intention
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

        drive = self._suppress_invalid_view(drive, invalid_value=0.0)
        next_pt = self._argmax_point(drive)
        self.matrix1 = drive

        if intention_for_drive == MicroIntention.BOUNDARY_SEEK and self.micro_intention == MicroIntention.BOUNDARY_SEEK:
            self.pending_boundary_target = next_pt if drive.max() > 1e-6 else None

        if intention_for_drive == MicroIntention.CONTOUR_TRACE:
            if self.boundary_hit_confidence > self.cfg.theta_boundary_hit_exit:
                self._mark_boundary_trace_progress(self.fixation_point)
            if self.micro_intention != MicroIntention.CONTOUR_TRACE:
                return next_pt, drive
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
        drive = self._suppress_invalid_view(drive, invalid_value=-1e6)
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
        
        I_macro = self._suppress_invalid_view(I_macro, invalid_value=0.0)
        
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
        drive = self._suppress_invalid_view(drive, invalid_value=-1e6)
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
        drive = self._suppress_invalid_view(self.semantic_mask, invalid_value=0.0)
        if drive.max() < 1e-6:
            return self._argmax_point(drive)
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
        self.boundary_rejected = torch.zeros((B, C, H, W), device=self.device)
        self.boundary_suppression = torch.zeros((B, C, H, W), device=self.device)
        self.pending_boundary_target = None
        self.trace_age = 0
        self.boundary_hit_confidence = 0.0
        self.boundary_forward_support = 0.0
        self.boundary_role_confidence = 0.0

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

    # ---- Optimizer state handlers ----
    def _projection_nodes_for_optimizer(self, gmem_i: GmemoryI) -> List[ModalityNode]:
        return list(gmem_i.nodes.values())

    def _jump_fixation(self, target: Tuple[int, int], ref: torch.Tensor):
        self.prev_fixation_point = self.fixation_point
        self.fixation_point = self._clamp_point_to_valid(target, ref)

    def _run_macro_search(self, gmem_ii: GmemoryII, S_maps: Dict[int, torch.Tensor]):
        self.debug_optimizer.setdefault('executed_handlers', []).append('MACRO_SEARCH')
        self.anchor_position = self.fixation_point
        target, drive = self._decide_next_saccade_macro(gmem_ii, S_maps)
        self._jump_fixation(target, drive)
        self.current_interest_map = drive

    def _activate_review_semantic(self, sem_node: SemanticNode, S_maps: Dict[int, torch.Tensor],
                                  H: int, W: int):
        ref = next(iter(S_maps.values()), torch.zeros((1, 1, H, W), device=self.device))
        sem_node.activation_level += 2.0
        sem_node.state_flag = NeuronState.ACTIVE
        sem_node.timer = self.cfg.tau_active_II
        self.E_input_gmem_ii_acc[sem_node.node_id] += 1.0
        self.active_semantic_id = sem_node.node_id
        self.anchor_position = self.fixation_point
        for edge in sem_node.iter_relation_edges():
            if edge.src_id == edge.dst_id:
                continue
            pred_r = math.exp(edge.rho_offset)
            pred_x = int(self.anchor_position[0] + pred_r * math.cos(edge.theta_offset))
            pred_y = int(self.anchor_position[1] + pred_r * math.sin(edge.theta_offset))
            pred_x, pred_y = self._clamp_point_to_valid((pred_x, pred_y), ref)
            self.optimizer.add_expectation(edge.dst_id, (pred_x, pred_y), H, W)

    def _run_review_verify(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                           local_nodes: List[ModalityNode], gmem_i: GmemoryI,
                           gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII',
                           S_maps: Dict[int, torch.Tensor], window_state: WindowState,
                           obs: StateObservation):
        self.debug_optimizer.setdefault('executed_handlers', []).append('REVIEW_VERIFY')
        self.debug_optimizer['review_target_at_entry'] = self.review_target
        ref = window_state.mask
        recall = self._semantic_recall_map(gmem_ii, S_maps, ref)
        self.current_interest_map = recall

        if self.review_target is not None:
            tx, ty = self._clamp_point_to_valid(self.review_target, ref)
            cx, cy = self.fixation_point
            dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            if dist > max(1.0, float(self.cfg.boundary_pending_radius)):
                self.debug_optimizer['review_outcome'] = 'jump'
                self._jump_fixation((tx, ty), ref)
                return

        recognized_sem_id = self._bfs_recognize(local_nodes, gmem_ii, self.cfg.k_depth)
        if recognized_sem_id is not None:
            self.debug_optimizer['review_outcome'] = 'recognized'
            self.debug_optimizer['recognized_semantic_id'] = recognized_sem_id
            H, W = ref.shape[-2], ref.shape[-1]
            sem_node = gmem_ii.semantic_nodes[recognized_sem_id]
            self._activate_review_semantic(sem_node, S_maps, H, W)
            for node in local_nodes:
                self.optimizer.I_exp.pop(node.node_id, None)
            self.review_target = None
            self.review_recent_fail = 0.0
            if self.suspend_stack and obs.resume_value > self.cfg.resume_value_threshold:
                self._set_behavior_state(OptimizerState.SUSPEND_RESUME)
            else:
                self.active_semantic_id = None
                self.anchor_position = None
                self.semantic_mask = None
                self.prev_keypoints = []
                self._set_behavior_state(OptimizerState.MACRO_SEARCH)
            return

        self.debug_optimizer['review_outcome'] = 'unrecognized'
        self.review_recent_fail = float(self.cfg.review_recent_decay) * self.review_recent_fail + float(self.cfg.fail_increment)
        self.review_target = None
        if local_nodes or obs.novelty > self.cfg.theta_novel:
            self._record_local_nodes_in_work_semantic(local_nodes, gmem_ii, window_state)
            self._set_behavior_state(OptimizerState.MICRO_BUILD)
        elif self.suspend_stack and obs.resume_value > self.cfg.resume_value_threshold:
            self._set_behavior_state(OptimizerState.SUSPEND_RESUME)
        else:
            self._set_behavior_state(OptimizerState.MACRO_SEARCH)

    def _run_suspend_resume(self, window_state: WindowState):
        self.debug_optimizer.setdefault('executed_handlers', []).append('SUSPEND_RESUME')
        if self._restore_suspended_work():
            if self.anchor_position is not None:
                self._jump_fixation(self.anchor_position, window_state.mask)
            self.current_interest_map = self.semantic_mask if self.semantic_mask is not None else window_state.work_response
            self._set_behavior_state(OptimizerState.MICRO_BUILD)
            return
        self.current_interest_map = window_state.work_response
        self._set_behavior_state(OptimizerState.MACRO_SEARCH)

    def _run_micro_build(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                         local_nodes: List[ModalityNode], gmem_i: GmemoryI,
                         gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII',
                         S_maps: Dict[int, torch.Tensor], window_state: WindowState,
                         obs: StateObservation):
        self.debug_optimizer.setdefault('executed_handlers', []).append('MICRO_BUILD')
        if local_nodes:
            self._record_local_nodes_in_work_semantic(local_nodes, gmem_ii, window_state)

        keypoints = self._sample_window_keypoints(X_subspaces, S_maps, gmem_i, window_state)
        written_nodes = self._write_window_keypoints(keypoints, X_subspaces, gmem_i, gmem_ii)
        self.debug_optimizer['written_keypoints'] = len(written_nodes)
        for node in written_nodes:
            self.E_input_gmem_i_acc[node.node_id] += 1.0
        if written_nodes:
            S_maps.update(gpos.l1_footprint_projection(X_subspaces, written_nodes))

        sem_node = self._current_semantic_node(gmem_ii)
        if sem_node is not None and self.semantic_mask is None:
            self.semantic_mask = window_state.mask.clone()
        if sem_node is not None:
            sem_node.frontier_strength = obs.frontier
            sem_node.gap_strength = obs.gap

        if self._maybe_complete_work(obs, gmem_i, gmem_ii, gmem_iii):
            self._run_macro_search(gmem_ii, S_maps)
            return

        decision = self._decide_next_saccade_micro_vector(X_subspaces, S_maps, gmem_i, gmem_ii, window_state)
        self._jump_fixation(decision.target, decision.potential)
        self.current_interest_map = decision.potential
        self.micro_explore_time = getattr(self, 'micro_explore_time', 0.0) + 1.0
        if self.anchor_position is not None:
            ax, ay = self.anchor_position
            dist = math.sqrt((self.fixation_point[0] - ax) ** 2 + (self.fixation_point[1] - ay) ** 2)
            self.micro_max_span = max(getattr(self, 'micro_max_span', 0.0), dist)

    def _execute_optimizer_state(self, X_subspaces: Dict[int, torch.Tensor], gpos: Gposition,
                                 local_nodes: List[ModalityNode], gmem_i: GmemoryI,
                                 gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII',
                                 S_maps: Dict[int, torch.Tensor],
                                 window_state: WindowState, obs: StateObservation):
        if self.behavior_state == OptimizerState.REVIEW_VERIFY:
            self._run_review_verify(X_subspaces, gpos, local_nodes, gmem_i, gmem_ii, gmem_iii, S_maps, window_state, obs)
            return
        if self.behavior_state == OptimizerState.SUSPEND_RESUME:
            self._run_suspend_resume(window_state)
            return
        if self.behavior_state == OptimizerState.MICRO_BUILD:
            self._run_micro_build(X_subspaces, gpos, local_nodes, gmem_i, gmem_ii, gmem_iii, S_maps, window_state, obs)
            return
        self._run_macro_search(gmem_ii, S_maps)

    def run_step(self, X_subspaces: Dict[int, torch.Tensor],
                 gmem_i: GmemoryI, gmem_ii: GmemoryII, gmem_iii: 'GmemoryIII', gpos: Gposition):
        if not X_subspaces:
            return
        # Capture selection, execution and post-transition state separately.
        self.debug_optimizer = {
            'state_before': self.behavior_state.name,
            'fixation_before': self.fixation_point,
            'executed_handlers': [],
        }
        self.current_step += 1
        if self.optimizer.base_interest is None:
            B, C, H0, W0 = self._first_map_shape(X_subspaces)
            self.optimizer.initialize_base_interest(X_subspaces, H0, W0)

        cx, cy = self.fixation_point
        B, C, H, W = self._first_map_shape(X_subspaces)
        self._ensure_runtime_maps(H, W)
        self.fixation_point = self._clamp_point_to_valid((cx, cy), self.optimizer.base_interest)
        cx, cy = self.fixation_point
        self.review_recent_fail *= float(self.cfg.review_recent_decay)

        # Step 1: foveal feature extraction and real-time GposI projection.
        local_nodes = self._extract_and_match_local_features(X_subspaces, gmem_i, cx, cy)
        for n in local_nodes:
            self.E_input_gmem_i_acc[n.node_id] += 1.0

        active_gmem_i = self._projection_nodes_for_optimizer(gmem_i)
        S_maps_footprint = gpos.l1_footprint_projection(X_subspaces, active_gmem_i)

        # Step 2: dynamic work window and state competition.
        self._update_semantic_energy_from_overlap(gmem_ii, S_maps_footprint, cx, cy)
        window_state = self._select_dynamic_window(X_subspaces, S_maps_footprint, gmem_ii)
        obs = self._compute_state_observation(window_state, X_subspaces, S_maps_footprint, gmem_ii)
        scores = self._score_optimizer_states(obs)
        next_state = self._select_behavior_state(obs, scores)
        self._set_behavior_state(next_state)
        self.debug_optimizer['state'] = self.behavior_state.name
        self.debug_optimizer['state_scores'] = scores.as_dict()
        self.debug_optimizer['state_observation'] = obs
        self.debug_optimizer['window_scores'] = window_state.scores

        # Step 3: state-specific behavior and unified neuron dynamics.
        self._execute_optimizer_state(
            X_subspaces, gpos, local_nodes, gmem_i, gmem_ii, gmem_iii,
            S_maps_footprint, window_state, obs
        )
        self._tick_all_nodes(gmem_i, gmem_ii, gmem_iii)
        self.debug_optimizer.update({
            'state_after': self.behavior_state.name,
            'fixation_after': self.fixation_point,
            'fail_energy_after': self.fail_energy,
            'review_recent_fail_after': self.review_recent_fail,
            'suspend_depth': len(self.suspend_stack),
        })


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

    def _as_4d(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 2:
            return value.unsqueeze(0).unsqueeze(0)
        if value.dim() == 3:
            return value.unsqueeze(0)
        return value

    def _prepare_grad_subspace(self, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.detach()
        if grad.dim() == 5 and grad.shape[-1] == 2:
            dx = grad[..., 0]
            dy = grad[..., 1]
            if dx.dim() == 3:
                dx = dx.unsqueeze(1)
                dy = dy.unsqueeze(1)
        elif grad.dim() == 4 and grad.shape[-1] == 2:
            dx = grad[..., 0].unsqueeze(1)
            dy = grad[..., 1].unsqueeze(1)
        else:
            grad = self._as_4d(grad)
            if grad.shape[1] >= 4:
                return grad
            if grad.shape[1] >= 2:
                dx = grad[:, 0:1]
                dy = grad[:, 1:2]
            else:
                strength = grad[:, 0:1].abs()
                theta = torch.zeros_like(strength)
                return torch.cat([strength, theta, torch.zeros_like(strength), torch.zeros_like(strength)], dim=1)

        strength = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        theta = torch.atan2(dy, dx)
        return torch.cat([strength, theta, dx, dy], dim=1)

    def _prepare_orientation_subspace(self, orientation: torch.Tensor) -> torch.Tensor:
        orientation = self._as_4d(orientation.detach())
        if orientation.numel() > 0 and float(orientation.detach().abs().max().item()) <= 1.05:
            return orientation * (math.pi / 2.0)
        return orientation

    def _first_available_feature(self, features: dict, names: List[str]) -> Optional[torch.Tensor]:
        for name in names:
            value = features.get(name)
            if value is not None:
                return value
        return None

    def handle_new_view(self, features: dict, H_orig: int = None, W_orig: int = None):
        X_subspaces = {}
        if 'grad' in features and features['grad'] is not None:
            X_subspaces[0] = self._prepare_grad_subspace(features['grad'])

        surface = self._first_available_feature(features, ['rgb', 'RGB', 'image', 'x', 'cropped', 'hue'])
        if surface is not None:
            X_subspaces[1] = self._as_4d(surface.detach())
        if 'curvature' in features and features['curvature'] is not None:
            X_subspaces[2] = self._as_4d(features['curvature'].detach())
        if 'aspect' in features and features['aspect'] is not None:
            X_subspaces[3] = self._as_4d(features['aspect'].detach())
        if 'orientation' in features and features['orientation'] is not None:
            X_subspaces[4] = self._prepare_orientation_subspace(features['orientation'])

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

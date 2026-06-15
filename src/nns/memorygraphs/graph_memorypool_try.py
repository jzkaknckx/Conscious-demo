import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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
    
    # [修改] 通道属性划分 (STRENGTH vs CONTINUITY_SURFACE vs CONTINUITY_TRACE vs IGNORE)
    feature_attributes = {
        0: {0: 'STRENGTH', 1: 'CONTINUITY_TRACE'},   # grad: 0=intensity, 1=orientation
        1: {0: 'CONTINUITY_SURFACE', 1: 'CONTINUITY_SURFACE'}, # hue, etc.
        2: {0: 'STRENGTH', 1: 'STRENGTH', 2: 'STRENGTH'},     # curvature components
        3: {0: 'CONTINUITY_TRACE', 1: 'CONTINUITY_TRACE', 2: 'CONTINUITY_TRACE'},  # aspect
        4: {0: 'CONTINUITY_TRACE', 1: 'CONTINUITY_TRACE', 2: 'CONTINUITY_TRACE'},  # orientation
    }
    sigma_surf = 0.1
    sigma_trace = 0.5
    
    # [新增] 动态写入门限
    tau_str_gate = 0.2
    tau_surf_gate = 0.5
    tau_trace_gate = 0.6

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
    
    # Interest Map New Params
    interest_margin_radius = 24
    alpha_local = 0.5
    beta_periphery = 0.8
    gamma_remote = 0.2
    saccade_sigma = 50
    foveal_sigma = 25
    
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
    lambda_str = 2.0
    mask_dilation_steps = 20


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


# =============================
# Gmemory: Modality Nodes & Semantic Nodes
# =============================
class ModalityNode:
    """模态孤立特征层节点 (GI)"""
    def __init__(self, node_id: int, modality_id: int, prototype: torch.Tensor):
        self.node_id = node_id
        self.modality_id = modality_id
        self.prototype = prototype
        
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
        self.peripheral_links: Dict[int, Dict[str, float]] = {}
        
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
                self.timer = cfg.tau_refractory_II
                self.activation_level = -1.0
        else:
            self.activation_level = self.activation_level * cfg.decay_rate_II + E_input
            if self.activation_level > cfg.T_excite_II:
                self.state_flag = NeuronState.ACTIVE
                self.activation_level = 1.0
                self.timer = cfg.tau_active_II


class GmemoryI:
    """Gmemory I 层：模态孤立特征层"""
    def __init__(self):
        self.nodes: Dict[int, ModalityNode] = {}
        self.next_node_id = 0
        
    def add_node(self, modality_id: int, prototype: torch.Tensor) -> ModalityNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = ModalityNode(nid, modality_id, prototype)
        self.nodes[nid] = node
        return node


class GmemoryII:
    """Gmemory II 层：锚点-外围拓扑层"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.semantic_nodes: Dict[int, SemanticNode] = {}
        self.next_node_id = 0

    def add_semantic_node(self, anchor_id: int) -> SemanticNode:
        nid = self.next_node_id
        self.next_node_id += 1
        node = SemanticNode(nid, anchor_id)
        self.semantic_nodes[nid] = node
        return node

    def add_peripheral(self, semantic_node: SemanticNode, peri_id: int, 
                       rho_offset: float, theta_offset: float,
                       lambda_rho: Optional[float] = None, gamma_theta: Optional[float] = None):
        """添加外围特征拓扑"""
        l_rho = lambda_rho if lambda_rho is not None else self.cfg.default_lambda_rho
        g_theta = gamma_theta if gamma_theta is not None else self.cfg.default_gamma_theta
        semantic_node.peripheral_links[peri_id] = {
            'rho_offset': rho_offset,
            'theta_offset': theta_offset,
            'lambda_rho': l_rho,
            'gamma_theta': g_theta
        }


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
            B, C_m, H, W = X_m.shape
            
            S_m = torch.sum(X_m * w_m, dim=1, keepdim=True)
            
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            S_maps[node.node_id] = S_m * weight
            
        return S_maps

    def l1_footprint_projection(self, X_subspaces: Dict[int, torch.Tensor], 
                                target_nodes: List[ModalityNode]) -> Dict[int, torch.Tensor]:
        """
        生成足迹专用投影：不论是否处于不应期，只要脱离了平静态 (A!=0)，均提供整图响应投射以制造区域脚印。
        """
        S_maps = {}
        active_nodes = [n for n in target_nodes if n.state_flag != NeuronState.CALM]
        for node in active_nodes:
            mod_id = node.modality_id
            X_m = X_subspaces.get(mod_id)
            if X_m is None:
                continue
                
            w_m = node.prototype.to(self.device).view(1, -1, 1, 1)
            S_m = torch.sum(X_m * w_m, dim=1, keepdim=True)
            
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
        
        for peri_id, link in semantic_node.peripheral_links.items():
            S_peri = S_maps.get(peri_id)
            if S_peri is not None:
                S_LP_peri = F.grid_sample(S_peri, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                D_n = self._generalized_distance_transform(
                    S_LP_peri, link['rho_offset'], link['theta_offset'], 
                    link['lambda_rho'], link['gamma_theta']
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
                for pid in node.peripheral_links:
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
        self.anchor_position: Optional[Tuple[int, int]] = None
        self.current_step = 0

        self.current_interest_map = None
        
        # Semantic Mask / Macro-Micro Saccade
        self.semantic_mask: Optional[torch.Tensor] = None
        self.sigma_micro: float = 50.0

        # trail
        self.matrix1 = None
        self.matrix2 = None
        self.matrix3 = None

    def _extract_and_match_local_features(self, X_subspaces: Dict[int, torch.Tensor], 
                                          gmem_i: GmemoryI, x: int, y: int, 
                                          similarity_threshold: float = 0.85) -> List[ModalityNode]:
        active_nodes = []
        for mod_id, X_m in X_subspaces.items():
            C_m = X_m.shape[1]
            H, W = X_m.shape[-2], X_m.shape[-1]
            mod_attrs = self.cfg.feature_attributes.get(mod_id, {})
            
            valid_mask = torch.zeros((1, C_m), device=self.device)
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
                        
            if torch.sum(valid_mask) == 0:
                continue

            local_vec = X_m[0, :, y, x].unsqueeze(0) * valid_mask
            local_norm = F.normalize(local_vec, p=2, dim=1)
            
            matched_node = None
            best_sim = -1.0
            
            for node in gmem_i.nodes.values():
                if node.modality_id == mod_id:
                    node_norm = F.normalize(node.prototype.unsqueeze(0), p=2, dim=1)
                    sim = torch.sum(local_norm * node_norm).item()
                    if sim > best_sim:
                        best_sim = sim
                        matched_node = node
            
            if matched_node and best_sim >= similarity_threshold:
                matched_node.count += 1
                matched_node.last_seen = time.time()
                active_nodes.append(matched_node)
            else:
                new_node = gmem_i.add_node(mod_id, local_vec.squeeze(0).detach().clone())
                active_nodes.append(new_node)
                
        return active_nodes

    def _bfs_recognize(self, local_nodes, gmem_ii: GmemoryII, k_depth: int = 2):
        from collections import deque
        queue = deque([(n.node_id, 0) for n in local_nodes])
        visited_modality = set([n.node_id for n in local_nodes])
        
        modality_to_semantics = defaultdict(list)
        for sid, sem in gmem_ii.semantic_nodes.items():
            modality_to_semantics[sem.anchor_id].append(sid)
            for pid in sem.peripheral_links.keys():
                modality_to_semantics[pid].append(sid)
                
        recognized_semantics = set()
        while queue:
            mid, depth = queue.popleft()
            
            sids = modality_to_semantics[mid]
            for sid in sids:
                recognized_semantics.add(sid)
                if depth < k_depth:
                    sem = gmem_ii.semantic_nodes[sid]
                    neighbors = [sem.anchor_id] + list(sem.peripheral_links.keys())
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
            mod_id = node.modality_id
            attrs = self.cfg.feature_attributes.get(mod_id, {})
            has_continuity_surface = any(attr == 'CONTINUITY_SURFACE' for attr in attrs.values())
            
            if has_continuity_surface and node.node_id in S_maps:
                M_resp_con += S_maps[node.node_id]
                count += 1
                
        if count > 0:
            M_resp_con /= count

        # 1. Permeability Map
        I_str = self.optimizer.I_str
        I_con1 = self.optimizer.I_con1
        I_con2 = self.optimizer.I_con2
        # P_map = M_resp_con * torch.sigmoid(1.0 - self.cfg.lambda_str * I_str)
        # P_map = self.optimizer.I_con1 * torch.sigmoid((1.0 - self.cfg.lambda_str * I_str))
        I_con1_norm = I_con1 / (I_con1.max() + 1e-6)
        I_con2_norm = I_con2 / (I_con2.max() + 1e-6)
        I_str_norm = I_str / (I_str.max() + 1e-6)
        I_con1_clamp = torch.clamp(1.0 - self.cfg.lambda_con1 * I_con1, 0.0, 1.0)
        I_con2_clamp = torch.clamp(1.0 - self.cfg.lambda_con2 * I_con2, 0.0, 1.0)
        I_str_clamp = torch.clamp(1.0 - self.cfg.lambda_str * I_str, 0.0, 1.0)
        P_surf_map = I_con1_norm * I_str_clamp
        P_edge_map = I_str_norm * I_con1_clamp
        P_trace_map = I_con2_norm * I_con1_clamp * I_str_clamp

        self.matrix1 = P_surf_map
        self.matrix3 = P_edge_map       

        # 2. Seed Initialization
        cx, cy = p_macro
        print('cx, cy', cx, cy)
        V = torch.zeros((B, 1, H, W), device=self.device)
        V[0, 0, cy, cx] = 1.0

        # 3. Iterative Matrix Dilation
        self.matrix2 = []
        pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for _ in range(self.cfg.mask_dilation_steps):
            self.matrix2.append(V)
            V = pool(V) * P_surf_map

        edge_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for _ in range(5):
            self.matrix2.append(V)
            V = edge_pool(V)
        '''
        for _ in range(20):
            self.matrix2.append(V)
            V = V + edge_pool(V) * P_trace_map
        '''
        # 4. Final Mask Output
        alpha = self.cfg.alpha_mask
        beta = self.cfg.beta_mask
        M_semantic = torch.sigmoid(alpha * V - beta)
        M_semantic[M_semantic < 0.0068] = 0

        return M_semantic

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
        
        alpha_ior = 0.5
        gamma_dist = 0.001
        
        drive = I_map - alpha_ior * self.ior_map - gamma_dist * torch.sqrt(dist_sq).view(1, 1, H, W)
        if drive.max() < -1e3:
            idx = torch.randint(0, H*W, (1,)).item()
        else:
            idx = torch.argmax(drive).item()
        return (int(idx % W), int(idx // W))

    def _decide_next_saccade_macro(self, base_interest: torch.Tensor) -> Tuple[int, int]:
        H, W = base_interest.shape[-2], base_interest.shape[-1]
        kernel_size = 15
        padding = kernel_size // 2
        
        if self.optimizer.I_con1 is not None:
            target_map = self.optimizer.I_con1.clone()
        else:
            target_map = base_interest.clone()
            
        if self.optimizer.sum_I_spatial_II is not None:
            sum_II = self.optimizer.sum_I_spatial_II
            if sum_II.max() > 1e-6:
                norm_II = torch.clamp(sum_II / sum_II.max(), 0.0, 1.0)
                target_map = target_map * (1.0 - norm_II)
                
        K_lowpass = F.avg_pool2d(target_map, kernel_size, stride=1, padding=padding)
        idx = torch.argmax(K_lowpass).item()
        return (int(idx % W), int(idx // W))

    def _decide_next_saccade_micro(self, base_interest: torch.Tensor, semantic_mask: torch.Tensor) -> Tuple[int, int]:
        H, W = base_interest.shape[-2], base_interest.shape[-1]
        cx, cy = self.fixation_point
        
        # sigma_foveal = getattr(self.cfg, 'foveal_sigma', 25.0)
        sigma_foveal = 10
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        
        aversion_pt = torch.exp(-dist_sq / (2 * sigma_foveal**2)).view(1, 1, H, W)
        self.m_aversion = torch.max(self.m_aversion, aversion_pt)
        
        drive = base_interest * semantic_mask * (1.0 - self.m_aversion)
        if drive.max() < 1e-6:
            print("drive.max() < 1e-6")
            idx = torch.randint(0, H*W, (1,)).item()
        else:
            idx = torch.argmax(drive).item()
        return (int(idx % W), int(idx // W))

    def _check_micro_exit(self, semantic_mask: torch.Tensor) -> bool:
        if semantic_mask is None:
            return True
        sum_mask = torch.sum(semantic_mask).item()
        print(sum_mask, "sum_mask")
        if sum_mask < 1e-5:
            return True
        covered = torch.sum(self.m_aversion * semantic_mask).item()
        rho = covered / sum_mask
        theta_exit = 0.85
        print(rho, "rho")
        return rho > theta_exit

    def _transition_to_macro(self, gmem_i: GmemoryI, gmem_ii: GmemoryII):
        self._trigger_macro_tick_update(gmem_i, gmem_ii)
        if self.active_semantic_id is not None:
            sem_node = gmem_ii.semantic_nodes.get(self.active_semantic_id)
            if sem_node:
                for p in self.peripheral_buffer:
                    if p['nid'] not in sem_node.peripheral_links:
                        gmem_ii.add_peripheral(sem_node, p['nid'], p['rho'], p['theta'])
        self.peripheral_buffer.clear()
        self.active_semantic_id = None
        self.anchor_position = None
        self.semantic_mask = None
        self.state = MainPhase.LEARN_MACRO
        B, C, H, W = self.m_aversion.shape
        self.m_aversion = torch.zeros((B, C, H, W), device=self.device)

    def _trigger_macro_tick_update(self, gmem_i: GmemoryI, gmem_ii: GmemoryII):
        for n in gmem_i.nodes.values():
            n.tick_update(self.E_input_gmem_i_acc[n.node_id], self.cfg)
        for n in gmem_ii.semantic_nodes.values():
            n.tick_update(self.E_input_gmem_ii_acc[n.node_id], self.cfg)
        self.E_input_gmem_i_acc.clear()
        self.E_input_gmem_ii_acc.clear()

    def run_step(self, X_subspaces: Dict[int, torch.Tensor], 
                 gmem_i: GmemoryI, gmem_ii: GmemoryII, gpos: Gposition):
        self.current_step += 1
        cx, cy = self.fixation_point
        B, C, H, W = list(X_subspaces.values())[0].shape
        cx, cy = max(0, min(cx, W-1)), max(0, min(cy, H-1))

        if getattr(self, 'ior_map', None) is None:
            self.ior_map = torch.zeros((1, 1, H, W), device=self.device)
            self.m_aversion = torch.zeros((1, 1, H, W), device=self.device)
            self.peripheral_buffer = []

        local_nodes = self._extract_and_match_local_features(X_subspaces, gmem_i, cx, cy)
        print('local_nodes: ', len(local_nodes))
        
        if not hasattr(self, 'E_input_gmem_i_acc'):
            self.E_input_gmem_i_acc = defaultdict(float)
            self.E_input_gmem_ii_acc = defaultdict(float)
            
        for n in local_nodes:
            self.E_input_gmem_i_acc[n.node_id] += 1.0

        active_gmem_i = [n for n in gmem_i.nodes.values() if n.state_flag != NeuronState.CALM]
        active_gmem_ii = [n for n in gmem_ii.semantic_nodes.values() if n.state_flag != NeuronState.CALM]

        # Generate routing map for query retrieval
        req_nodes = list(gmem_i.nodes.values())
        S_maps_retrieval = gpos.l1_modality_routing_projection(X_subspaces, req_nodes)
        
        # Generate spatial footprint projection mapping 
        S_maps_footprint = gpos.l1_footprint_projection(X_subspaces, req_nodes)

        if self.state == MainPhase.REVIEW:
            recognized_sem_id = self._bfs_recognize(local_nodes, gmem_ii, self.cfg.k_depth)
            if recognized_sem_id is not None:
                 if gmem_ii.semantic_nodes[recognized_sem_id].state_flag != NeuronState.ACTIVE:
                     gmem_ii.semantic_nodes[recognized_sem_id].activation_level += 2.0
                     gmem_ii.semantic_nodes[recognized_sem_id].state_flag = NeuronState.ACTIVE
                     gmem_ii.semantic_nodes[recognized_sem_id].timer = self.cfg.tau_active_II
                     
                 sem_node = gmem_ii.semantic_nodes[recognized_sem_id]
                 if self.anchor_position is None:
                     self.anchor_position = (cx, cy)
                 ax, ay = self.anchor_position
                 
                 for pid, link in sem_node.peripheral_links.items():
                     pred_r = math.exp(link['rho_offset'])
                     pred_theta = link['theta_offset']
                     pred_x = int(ax + pred_r * math.cos(pred_theta))
                     pred_y = int(ay + pred_r * math.sin(pred_theta))
                     pred_x, pred_y = max(0, min(pred_x, W-1)), max(0, min(pred_y, H-1))
                     self.optimizer.add_expectation(pid, (pred_x, pred_y), H, W)
                     
                 for n in local_nodes:
                     if n.node_id in self.optimizer.I_exp:
                         self.optimizer.I_exp.pop(n.node_id, None)
            else:
                 if local_nodes:
                     self.state = MainPhase.LEARN_MICRO
                     new_sem_node = gmem_ii.add_semantic_node(local_nodes[0].node_id)
                     self.active_semantic_id = new_sem_node.node_id
                     self.anchor_position = (cx, cy)
                     self.semantic_mask = self._generate_semantic_mask(X_subspaces, gpos, local_nodes, (cx, cy))
                 else:
                     self._transition_to_macro(gmem_i, gmem_ii)
                     
        elif self.state == MainPhase.LEARN_MICRO:
            sem_node = gmem_ii.semantic_nodes.get(self.active_semantic_id)
            if sem_node and self.anchor_position is not None:
                ax, ay = self.anchor_position
                r = math.sqrt((cx - ax)**2 + (cy - ay)**2) + 1e-5
                rho = math.log(r)
                theta = math.atan2(cy - ay, cx - ax)
                
                for n in local_nodes:
                    if n.node_id != sem_node.anchor_id:
                        buffered_nids = [p['nid'] for p in self.peripheral_buffer]
                        if n.node_id not in buffered_nids:
                            self.peripheral_buffer.append({'nid': n.node_id, 'rho': rho, 'theta': theta})
            else:
                self._transition_to_macro(gmem_i, gmem_ii)

        elif self.state == MainPhase.LEARN_MACRO:
            self.anchor_position = (cx, cy)
            if local_nodes:
                new_sem_node = gmem_ii.add_semantic_node(local_nodes[0].node_id)
                self.active_semantic_id = new_sem_node.node_id
                self.semantic_mask = self._generate_semantic_mask(X_subspaces, gpos, local_nodes, (cx, cy))
                self.state = MainPhase.LEARN_MICRO

        # Execute decision logic based on current state
        self.prev_fixation_point = self.fixation_point
        base_interest = self.optimizer.base_interest
        if self.semantic_mask is not None:
            print(self.semantic_mask[0, 0, cy, cx], "semantic_mask[0, 0, cy, cx]")

        if self.state == MainPhase.REVIEW:
            I_map = self.optimizer.calculate_I_map(active_gmem_i, active_gmem_ii, S_maps_footprint, (cx, cy))
            self.fixation_point = self._decide_next_saccade_review(I_map)
            self.current_interest_map = I_map
        elif self.state == MainPhase.LEARN_MACRO:
            self.fixation_point = self._decide_next_saccade_macro(base_interest)
            self.current_interest_map = base_interest
        elif self.state == MainPhase.LEARN_MICRO:
            if self._check_micro_exit(self.semantic_mask):
                self._transition_to_macro(gmem_i, gmem_ii)
                self.fixation_point = self._decide_next_saccade_macro(base_interest)
                self.current_interest_map = base_interest
            else:
                self.fixation_point = self._decide_next_saccade_micro(base_interest, self.semantic_mask)
                self.current_interest_map = base_interest * self.semantic_mask


# =============================
# MultilevelCoordinator
# =============================
class MultilevelCoordinator:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.gmem_i = GmemoryI()
        self.gmem_ii = GmemoryII(cfg)
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
            
        self.controller.run_step(X_subspaces, self.gmem_i, self.gmem_ii, self.gpos)
        self.current_view = self.controller.fixation_point
        self.viewroute.append(self.current_view)

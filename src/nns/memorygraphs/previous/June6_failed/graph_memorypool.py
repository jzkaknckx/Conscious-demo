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
# Unified Config (v2.1)
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
        0: 0.4,
        1: 0.6,
        2: 0.5,
    }

    # Neuron Dynamics
    T_excite = 0.8
    T_inject = 0.5
    tau_active = 5
    tau_refractory = 10
    decay_rate = 0.95
    
    # Interest Map New Params
    interest_margin_radius = 3
    alpha_local = 0.5
    beta_periphery = 0.8
    gamma_remote = 0.2
    saccade_sigma = 200.0
    
    # Topology Mount Params
    D_thres = 30.0
    k_depth = 2


# =============================
# Controller State Machine
# =============================
class NeuronState(Enum):
    REFRACTORY = -1
    CALM = 0
    ACTIVE = 1

class MainPhase(Enum):
    REVIEW = 1
    LEARN = 2


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
                self.activation_level = 0.0
        elif self.state_flag == NeuronState.ACTIVE and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.REFRACTORY
                self.timer = cfg.tau_refractory
                self.activation_level = 0.0
        else: # CALM
            self.activation_level = self.activation_level * cfg.decay_rate + E_input
            if self.activation_level > cfg.T_excite:
                self.state_flag = NeuronState.ACTIVE
                self.activation_level = 1.0
                self.timer = cfg.tau_active


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
                self.activation_level = 0.0
        elif self.state_flag == NeuronState.ACTIVE and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.state_flag = NeuronState.REFRACTORY
                self.timer = cfg.tau_refractory
                self.activation_level = 0.0
        else:
            self.activation_level = self.activation_level * cfg.decay_rate + E_input
            if self.activation_level > cfg.T_excite:
                self.state_flag = NeuronState.ACTIVE
                self.activation_level = 1.0
                self.timer = cfg.tau_active


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
        active_nodes = [n for n in target_nodes if n.activation_level > self.cfg.T_inject and n.state_flag != NeuronState.REFRACTORY]
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
# Controller
# =============================
class Controller:
    """控制器：基于动态兴趣矩阵与双塔交互的检视引擎"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        self.state = MainPhase.REVIEW
        self.interest_map: Optional[torch.Tensor] = None
        self.base_interest: Optional[torch.Tensor] = None
        
        self.fixation_point = (cfg.W // 2, cfg.H // 2)
        self.prev_fixation_point = self.fixation_point
        
        self.active_semantic_id: Optional[int] = None 
        self.anchor_position: Optional[Tuple[int, int]] = None
        self.current_step = 0

    def initialize_interest_map(self, X_subspaces: Dict[int, torch.Tensor], H_orig: int = None, W_orig: int = None, r: int = 3):
        r = self.cfg.interest_margin_radius
        B, C, H, W = list(X_subspaces.values())[0].shape
        self.base_interest = torch.zeros((B, 1, H, W), device=self.device)
        
        if 0 in X_subspaces:
             self.base_interest += torch.norm(X_subspaces[0], dim=1, keepdim=True)
        if 1 in X_subspaces:
             self.base_interest += torch.norm(X_subspaces[1], dim=1, keepdim=True)

        # 压低外围兴趣
        h1, h2 = H//2-(H_orig-r)//2, H//2+(H_orig-r)//2
        w1, w2 = W//2-(W_orig-r)//2, W//2+(W_orig-r)//2
        self.base_interest[:, :, :h1, :] = 0; self.base_interest[:, :, h2:, :] = 0
        self.base_interest[:, :, :, :w1] = 0; self.base_interest[:, :, :, w2:] = 0

        b_max = self.base_interest.max() + 1e-5
        self.base_interest /= b_max
        
        self.interest_map = self.base_interest.clone()

    def _extract_and_match_local_features(self, X_subspaces: Dict[int, torch.Tensor], 
                                          gmem_i: GmemoryI, x: int, y: int, 
                                          similarity_threshold: float = 0.85) -> List[ModalityNode]:
        active_nodes = []
        for mod_id, X_m in X_subspaces.items():
            local_vec = X_m[0, :, y, x].unsqueeze(0)
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

    def post_step_update_interest(self, cx: int, cy: int, M_resp: torch.Tensor):
        H, W = M_resp.shape[-2], M_resp.shape[-1]
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        sigma_local = self.cfg.saccade_sigma
        G_sigma = torch.exp(-dist_sq / (2 * sigma_local**2)).view(1, 1, H, W)
        
        M_local = M_resp * G_sigma
        
        padding = 5
        kernel_size = 11
        blurred_M_local = F.avg_pool2d(M_local, kernel_size, stride=1, padding=padding)
        M_periphery = F.relu(blurred_M_local - M_local)
        M_remote = F.relu(M_resp - M_local)
        
        alpha, beta, gamma = self.cfg.alpha_local, self.cfg.beta_periphery, self.cfg.gamma_remote
        self.interest_map = F.relu(self.interest_map - alpha * M_local + beta * M_periphery + gamma * M_remote)
        
        saccade_target_map = self.interest_map * torch.exp(-dist_sq / (2 * self.cfg.saccade_sigma**2)).view(1, 1, H, W)
        # Avoid zero maps logic
        if saccade_target_map.max() < 1e-6:
            idx = torch.randint(0, H*W, (1,)).item()
        else:
            idx = torch.argmax(saccade_target_map).item()
        
        self.prev_fixation_point = self.fixation_point
        self.fixation_point = (int(idx % W), int(idx // W))

    def run_step(self, X_subspaces: Dict[int, torch.Tensor], 
                 gmem_i: GmemoryI, gmem_ii: GmemoryII, gpos: Gposition):
        self.current_step += 1
        cx, cy = self.fixation_point
        B, C, H, W = list(X_subspaces.values())[0].shape
        cx, cy = max(0, min(cx, W-1)), max(0, min(cy, H-1))

        local_nodes = self._extract_and_match_local_features(X_subspaces, gmem_i, cx, cy)
        
        E_input_gmem_i = defaultdict(float)
        E_input_gmem_ii = defaultdict(float)
        
        for n in local_nodes:
            E_input_gmem_i[n.node_id] += 1.0
            
        req_nodes = list(gmem_i.nodes.values())
        S_maps = gpos.l1_modality_routing_projection(X_subspaces, req_nodes)
        
        M_resp = self.base_interest.clone()
        if S_maps:
            sum_smaps = torch.sum(torch.stack(list(S_maps.values())), dim=0)
            M_resp = M_resp + sum_smaps
            M_resp /= (M_resp.max() + 1e-5)

        if self.state == MainPhase.REVIEW:
            recognized_sem_id = self._bfs_recognize(local_nodes, gmem_ii, self.cfg.k_depth)
            if recognized_sem_id is not None:
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
                     
                     dist_sq = (torch.arange(W, device=self.device).view(1, W) - pred_x)**2 + \
                               (torch.arange(H, device=self.device).view(H, 1) - pred_y)**2
                     peak = 2.0 * torch.exp(-dist_sq / (2 * 10.0**2)).view(1, 1, H, W)
                     self.interest_map += peak
                     
                 for n in local_nodes:
                     n.state_flag = NeuronState.REFRACTORY
                     n.timer = self.cfg.tau_refractory
            else:
                 self.state = MainPhase.LEARN
                 if local_nodes:
                     new_sem_node = gmem_ii.add_semantic_node(local_nodes[0].node_id)
                     self.active_semantic_id = new_sem_node.node_id
                     self.anchor_position = (cx, cy)
                     
        elif self.state == MainPhase.LEARN:
            px, py = self.prev_fixation_point
            dist = math.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < self.cfg.D_thres:
                if self.active_semantic_id is not None and self.anchor_position is not None:
                    if self.active_semantic_id in gmem_ii.semantic_nodes:
                        sem_node = gmem_ii.semantic_nodes[self.active_semantic_id]
                        ax, ay = self.anchor_position
                        r = math.sqrt((cx - ax)**2 + (cy - ay)**2) + 1e-5
                        rho = math.log(r)
                        theta = math.atan2(cy - ay, cx - ax)
                        
                        for n in local_nodes:
                            if n.node_id != sem_node.anchor_id and n.node_id not in sem_node.peripheral_links:
                                gmem_ii.add_peripheral(sem_node, n.node_id, rho, theta)
            else:
                if self.active_semantic_id is not None:
                    E_input_gmem_ii[self.active_semantic_id] += 1.0
                if local_nodes:
                    new_sem_node = gmem_ii.add_semantic_node(local_nodes[0].node_id)
                    self.active_semantic_id = new_sem_node.node_id
                    self.anchor_position = (cx, cy)

        for n in gmem_i.nodes.values():
            n.tick_update(E_input_gmem_i[n.node_id], self.cfg)
        for n in gmem_ii.semantic_nodes.values():
            n.tick_update(E_input_gmem_ii[n.node_id], self.cfg)

        self.post_step_update_interest(cx, cy, M_resp)


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
        
        if self.controller.base_interest is None:
            B, C, H, W = list(X_subspaces.values())[0].shape
            H_o = H_orig if H_orig is not None else H
            W_o = W_orig if W_orig is not None else W
            self.controller.initialize_interest_map(X_subspaces, H_o, W_o)
            
        self.controller.run_step(X_subspaces, self.gmem_i, self.gmem_ii, self.gpos)
        self.current_view = self.controller.fixation_point
        self.viewroute.append(self.current_view)

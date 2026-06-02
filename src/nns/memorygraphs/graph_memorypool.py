import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# Unified Config (v2.0)
# =============================
class MemoryConfig:
    H = 256
    W = 256
    device = torch.device('cpu')
    
    # 极坐标采样配置 (Log-Polar Config)
    lp_rho_bins = 64        # 极径采样精度
    lp_theta_bins = 64      # 极角采样精度
    lp_base_radius = 2.0    # 对数极坐标的基础起算半径
    
    # 广义距离变换形变约束 (GDT Constraints)
    default_lambda_rho = 1.5   # 默认的尺度游离惩罚系数
    default_gamma_theta = 2.0  # 默认的角度游离惩罚系数
    
    # 模态权重平衡 (Modality Normalization)
    modality_weights = {
        0: 0.4,  # edge
        1: 0.6,  # hue
        2: 0.5,  # RG/BY (待实现)
    }

    # --- 新增：拓扑演化与控制器参数 ---
    global_scaler_init = 1.0       # 动态度量尺初始值
    ema_alpha = 0.05               # 畸变微调更新率
    deformation_threshold = 2.5    # 触发新事物分支的形变容忍度阈值
    saccade_explore_sigma = 15.0   # 探索状态下的高斯平滑核参数
    saccade_consolidate_gain = 5.0 # 校验状态下预测点兴趣值的注入增益


# =============================
# Controller State Machine
# =============================
class SaccadeState(Enum):
    EXPLORE = 1      # 全局探索
    INSPECT = 2      # 局部审查
    CONSOLIDATE = 3  # 巩固校验


# =============================
# Gmemory: Modality Nodes & Semantic Nodes
# =============================
class ModalityNode:
    """模态孤立特征层节点 (GI)"""
    def __init__(self, node_id: int, modality_id: int, prototype: torch.Tensor):
        self.node_id = node_id
        self.modality_id = modality_id
        self.prototype = prototype  # 低维子空间原型向量 w_m
        
        self.count = 1
        self.last_seen = 0
        self.stability_score = 1.0


class SemanticNode:
    """锚点-外围拓扑层节点 (GII，语义对象)"""
    def __init__(self, node_id: int, anchor_id: int):
        self.node_id = node_id
        self.anchor_id = anchor_id
        
        # 记录每个 Peripheral 允许的空间游离容忍度参数
        # { peri_id: {'rho_offset': float, 'theta_offset': float, 'lambda_rho': float, 'gamma_theta': float} }
        self.peripheral_links: Dict[int, Dict[str, float]] = {}


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

    def inverse_stability(self, semantic_node: SemanticNode, gmem_i: GmemoryI):
        """稳定性倒置规则 (Stability Inversion)"""
        anchor_node = gmem_i.nodes.get(semantic_node.anchor_id)
        if not anchor_node:
            return
            
        best_peri_id = -1
        best_peri_score = -1.0
        
        # 查找稳定性超越 Anchor 的 Peripheral
        for peri_id in semantic_node.peripheral_links.keys():
            peri_node = gmem_i.nodes.get(peri_id)
            if peri_node and peri_node.stability_score > best_peri_score:
                best_peri_score = peri_node.stability_score
                best_peri_id = peri_id
                
        if best_peri_score > anchor_node.stability_score:
            # 执行倒置
            link = semantic_node.peripheral_links.pop(best_peri_id)
            new_anchor_id = best_peri_id
            old_anchor_id = semantic_node.anchor_id
            
            # 原有的拓扑向量取逆: theta_new = (theta_old + pi) % 2pi
            # rho 保持不变 (距离是不变的)
            new_theta = (link['theta_offset'] + math.pi) % (2 * math.pi)
            
            semantic_node.anchor_id = new_anchor_id
            semantic_node.peripheral_links[old_anchor_id] = {
                'rho_offset': link['rho_offset'],
                'theta_offset': new_theta,
                'lambda_rho': link['lambda_rho'],
                'gamma_theta': link['gamma_theta']
            }

            # TODO: 完善其他外围节点的跨级坐标映射修正

    def star_topology_compression(self, semantic_node: SemanticNode, child_semantic_node: SemanticNode):
        """菊花状拓扑退化 (Star Topology Compression): A -> B -> C 转 A -> C"""
        # child_semantic_node 的 anchor 是 semantic_node 的一个 peripheral
        b_id = child_semantic_node.anchor_id
        if b_id not in semantic_node.peripheral_links:
            return
            
        link_AB = semantic_node.peripheral_links[b_id]
        rho_AB = link_AB['rho_offset']
        theta_AB = link_AB['theta_offset']

        for c_id, link_BC in child_semantic_node.peripheral_links.items():
            rho_BC = link_BC['rho_offset']
            theta_BC = link_BC['theta_offset']
            
            # 在笛卡尔坐标系下计算过渡的跨级偏移量
            # 半径 r = e^rho
            r_AB = math.exp(rho_AB)
            r_BC = math.exp(rho_BC)
            
            dx = r_AB * math.cos(theta_AB) + r_BC * math.cos(theta_BC)
            dy = r_AB * math.sin(theta_AB) + r_BC * math.sin(theta_BC)
            
            r_AC = math.sqrt(dx*dx + dy*dy) + 1e-6
            rho_AC = math.log(r_AC)
            theta_AC = math.atan2(dy, dx)
            
            self.add_peripheral(
                semantic_node, c_id, 
                rho_AC, theta_AC, 
                max(link_AB['lambda_rho'], link_BC['lambda_rho']), 
                max(link_AB['gamma_theta'], link_BC['gamma_theta'])
            )

    def distortion_correction_and_branch(self, semantic_node: SemanticNode, peri_id: int, 
                                         matched_rho: float, matched_theta: float, energy: float):
        """畸变纠正与新事物分支"""
        if peri_id not in semantic_node.peripheral_links:
            return
            
        if energy <= self.cfg.deformation_threshold:
            # 误差在范围内，使用 EMA 更新
            link = semantic_node.peripheral_links[peri_id]
            alpha = self.cfg.ema_alpha
            link['rho_offset'] = (1 - alpha) * link['rho_offset'] + alpha * matched_rho
            
            # 夹角插值需要考虑周期性
            diff_theta = (matched_theta - link['theta_offset'] + math.pi) % (2 * math.pi) - math.pi
            link['theta_offset'] = (link['theta_offset'] + alpha * diff_theta) % (2 * math.pi)
        else:
            # 误差过大，触发分支
            new_node = self.add_semantic_node(semantic_node.anchor_id)
            # 克隆原有 links
            for pid, plink in semantic_node.peripheral_links.items():
                if pid != peri_id:
                    new_node.peripheral_links[pid] = plink.copy()
            # 挂载形变较大的新位置
            self.add_peripheral(new_node, peri_id, matched_rho, matched_theta, 
                                self.cfg.default_lambda_rho, self.cfg.default_gamma_theta)


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
        执行 1x1 卷积（点积）计算相似度。
        """
        S_maps = {}
        for node in target_nodes:
            mod_id = node.modality_id
            X_m = X_subspaces.get(mod_id)
            if X_m is None:
                continue
                
            w_m = node.prototype.to(self.device)  # [C_m]
            # X_m: [B, C_m, H, W]
            B, C_m, H, W = X_m.shape
            
            # 1x1 投射
            w_m_view = w_m.view(1, C_m, 1, 1).expand(B, -1, -1, -1)
            S_m = torch.sum(X_m * w_m_view, dim=1, keepdim=True) # [B, 1, H, W]
            
            # 模态加权平衡
            weight = self.cfg.modality_weights.get(mod_id, 1.0)
            S_maps[node.node_id] = S_m * weight
            
        return S_maps

    def _create_log_polar_grid(self, H: int, W: int, xc: int, yc: int) -> torch.Tensor:
        """生成对数极坐标网格"""
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
        
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0) # [1, theta_bins, rho_bins, 2]
        return grid

    def _generalized_distance_transform(self, S_LP: torch.Tensor, rho_offset: float, theta_offset: float, 
                                        lambda_rho: float, gamma_theta: float) -> torch.Tensor:
        """广义距离变换算法 (Generalized Distance Transform, GDT)"""
        _, C, theta_bins, rho_bins = S_LP.shape
        # S_LP: [1, 1, theta_bins, rho_bins]
        
        # 为了高效，此处使用平滑卷积来近似模拟 GDT 的可容忍形变膨胀
        padding = (2, 2, 2, 2)
        S_padded = F.pad(S_LP, padding, mode='circular' if theta_bins else 'replicate')
        kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25
        D_map = F.conv2d(S_padded, kernel)
        
        return D_map

    def l2_structure_synthesis(self, S_maps: Dict[int, torch.Tensor], semantic_node: SemanticNode):
        """L2层：结构合成与连续尺度解算"""
        S_anc = S_maps.get(semantic_node.anchor_id)
        if S_anc is None:
            return None
        
        B, C, H, W = S_anc.shape
        
        # 1. 锚点候选锁定 (Anchor Proposal)
        pooled = F.max_pool2d(S_anc, kernel_size=7, stride=1, padding=3)
        peaks = (S_anc == pooled) & (S_anc > 0.1)
        
        peak_indices = torch.nonzero(peaks[0, 0])
        if peak_indices.numel() == 0:
            return None
            
        best_idx = torch.argmax(S_anc[0, 0, peak_indices[:, 0], peak_indices[:, 1]])
        yc, xc = int(peak_indices[best_idx, 0]), int(peak_indices[best_idx, 1])

        # 2. 对数极坐标扭曲 (Log-Polar Warping) & 3. GDT
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

        # 4. 全局拓扑共振解算
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
    """控制器状态机：维护动态兴趣矩阵并控制眼跳与双塔交互"""
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        self.global_scaler = cfg.global_scaler_init
        self.state = SaccadeState.EXPLORE
        
        self.interest_map: Optional[torch.Tensor] = None
        self.base_interest: Optional[torch.Tensor] = None
        self.m_trace: Optional[torch.Tensor] = None
        self.m_comp: Optional[torch.Tensor] = None
        
        self.fixation_point = (cfg.W // 2, cfg.H // 2)

    def initialize_interest_map(self, X_subspaces: Dict[int, torch.Tensor]):
        """生成基础层兴趣图"""
        B, C, H, W = list(X_subspaces.values())[0].shape
        self.base_interest = torch.zeros((B, 1, H, W), device=self.device)
        self.m_trace = torch.ones((B, 1, H, W), device=self.device)
        self.m_comp = torch.ones((B, 1, H, W), device=self.device)
        
        if 0 in X_subspaces:
             self.base_interest += torch.norm(X_subspaces[0], dim=1, keepdim=True)
        if 1 in X_subspaces:
             self.base_interest += torch.norm(X_subspaces[1], dim=1, keepdim=True)
             
        b_max = self.base_interest.max() + 1e-5
        self.base_interest /= b_max
        self.update_interest_map()

    def update_interest_map(self):
        if self.base_interest is not None:
             self.interest_map = self.base_interest * self.m_trace * self.m_comp

    def run_step(self, X_subspaces: Dict[int, torch.Tensor], 
                 gmem_i: GmemoryI, gmem_ii: GmemoryII, gpos: Gposition):
        """执行一个周期：Top-Down 匹配与 Bottom-Up 眼跳"""
        # --- Top-Down: 跨塔结构注入与匹配 ---
        required_nodes = list(gmem_i.nodes.values())
        if not required_nodes:
            self._execute_saccade_explore()
            return
            
        S_maps = gpos.l1_modality_routing_projection(X_subspaces, required_nodes)
        
        matched_targets = []
        for sem_node in gmem_ii.semantic_nodes.values():
            res = gpos.l2_structure_synthesis(S_maps, sem_node)
            if res is not None:
                matched_targets.append((sem_node, res))
                
        # --- 控制器根据返回结果切换状态与物理闭环 ---
        if self.state == SaccadeState.EXPLORE:
            if matched_targets:
                best_match = max(matched_targets, key=lambda x: x[1]['score'])
                self.global_scaler = best_match[1]['s_star']
                
                self.state = SaccadeState.INSPECT
                self.fixation_point = (best_match[1]['x'], best_match[1]['y'])
                
                # Bottom-Up 兴趣反馈: 在匹配原点抬高兴趣值
                self._inject_gaussian_peak(self.fixation_point[0], self.fixation_point[1], gain=3.0)
            else:
                self._execute_saccade_explore()
                
        elif self.state == SaccadeState.INSPECT:
            # TODO: 局部寻找新的外围节点，建立或维护 Peripheral
            self.state = SaccadeState.CONSOLIDATE
            
        elif self.state == SaccadeState.CONSOLIDATE:
            # 校验缺失部件
            predicted_x = self.fixation_point[0] + 20
            predicted_y = self.fixation_point[1] + 20
            
            # 在兴趣图强制注入高斯峰，制导眼跳过去验证
            self._inject_gaussian_peak(predicted_x, predicted_y, gain=self.cfg.saccade_consolidate_gain)
            self.fixation_point = (predicted_x, predicted_y)
            
            self.state = SaccadeState.EXPLORE
            
    def _execute_saccade_explore(self):
        """全局探索长程眼跳"""
        if self.interest_map is not None:
            smoothed = F.avg_pool2d(self.interest_map, kernel_size=15, stride=1, padding=7)
            idx = torch.argmax(smoothed).item()
            W = self.interest_map.shape[-1]
            self.fixation_point = (int(idx % W), int(idx // W))
            
    def _inject_gaussian_peak(self, x: int, y: int, gain: float):
        """Bottom-up 兴趣反馈注入高斯峰"""
        if self.interest_map is None: return
        H, W = self.interest_map.shape[-2], self.interest_map.shape[-1]
        
        y_grid = torch.arange(H, device=self.device).view(H, 1)
        x_grid = torch.arange(W, device=self.device).view(1, W)
        
        dist_sq = (x_grid - x)**2 + (y_grid - y)**2
        sigma = self.cfg.saccade_explore_sigma
        peak = gain * torch.exp(-dist_sq / (2 * sigma**2))
        
        self.interest_map += peak.view(1, 1, H, W)

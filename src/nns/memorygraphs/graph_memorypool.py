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
    
    # Activation Dynamics Params
    tau_active = 0.1
    W_genesis_threshold = 2.0
    tau_recognize = 0.6
    decay_rate = 0.95
    activation_gain = 0.5
    coactivation_eta = 0.5
    spatial_affinity_threshold = 0.1


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
        
        # 激活状态
        self.activation_level = 0.0
        self.last_update_step = 0

    def update_activation(self, step: int, gain: float, decay_rate: float):
        delta_t = step - self.last_update_step
        if delta_t > 0:
            self.activation_level = self.activation_level * (decay_rate ** delta_t)
        self.activation_level = min(1.0, self.activation_level + gain)
        self.last_update_step = step


class SemanticNode:
    """锚点-外围拓扑层节点 (GII，语义对象)"""
    def __init__(self, node_id: int, anchor_id: int):
        self.node_id = node_id
        self.anchor_id = anchor_id
        
        # 记录每个 Peripheral 允许的空间游离容忍度参数
        # { peri_id: {'rho_offset': float, 'theta_offset': float, 'lambda_rho': float, 'gamma_theta': float} }
        self.peripheral_links: Dict[int, Dict[str, float]] = {}
        
        # 激活状态
        self.activation_level = 0.0
        self.last_update_step = 0

    def update_activation(self, step: int, gain: float, decay_rate: float):
        delta_t = step - self.last_update_step
        if delta_t > 0:
            self.activation_level = self.activation_level * (decay_rate ** delta_t)
        self.activation_level = min(1.0, self.activation_level + gain)
        self.last_update_step = step


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
        self.co_activation_matrix = defaultdict(float)

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

    def distortion_correction_and_branch(self, semantic_node: SemanticNode, peri_id: int, 
                                         matched_rho: float, matched_theta: float, energy: float,
                                         m_spatial: float = 1.0):
        """畸变纠正与新事物分支"""
        if peri_id not in semantic_node.peripheral_links:
            return
            
        if energy <= self.cfg.deformation_threshold and m_spatial > self.cfg.spatial_affinity_threshold:
            # 误差在范围内，且结构连续，使用 EMA 更新
            link = semantic_node.peripheral_links[peri_id]
            alpha = self.cfg.ema_alpha
            link['rho_offset'] = (1 - alpha) * link['rho_offset'] + alpha * matched_rho
            
            # 夹角插值需要考虑周期性
            diff_theta = (matched_theta - link['theta_offset'] + math.pi) % (2 * math.pi) - math.pi
            link['theta_offset'] = (link['theta_offset'] + alpha * diff_theta) % (2 * math.pi)
        else:
            # 误差过大或跨越断言，触发分支
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
        执行 1x1 卷积（点积）计算相似度。仅检索活跃节点！
        """
        S_maps = {}
        active_nodes = [n for n in target_nodes if n.activation_level >= self.cfg.tau_active]
        for node in active_nodes:
            mod_id = node.modality_id
            X_m = X_subspaces.get(mod_id)
            if X_m is None:
                continue
                
            w_m = node.prototype.to(self.device).view(1, -1, 1, 1)  # [1, C_m, 1, 1]
            B, C_m, H, W = X_m.shape
            
            # 1x1 投射
            S_m = torch.sum(X_m * w_m, dim=1, keepdim=True) # [B, 1, H, W]
            
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
        padding = (2, 2, 2, 2)
        S_padded = F.pad(S_LP, padding, mode='replicate')
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
        
        self.active_semantic_id: Optional[int] = None 
        self.anchor_position: Optional[Tuple[int, int]] = None
        self.current_step = 0

    def initialize_interest_map(self, X_subspaces: Dict[int, torch.Tensor], H_orig: int = None, W_orig: int = None, r: int = 3):
        """生成基础层兴趣图"""
        B, C, H, W = list(X_subspaces.values())[0].shape
        self.base_interest = torch.zeros((B, 1, H, W), device=self.device)
        self.m_trace = torch.ones((B, 1, H, W), device=self.device)
        self.m_comp = torch.ones((B, 1, H, W), device=self.device)
        
        if 0 in X_subspaces:
             self.base_interest += torch.norm(X_subspaces[0], dim=1, keepdim=True)
        if 1 in X_subspaces:
             self.base_interest += torch.norm(X_subspaces[1], dim=1, keepdim=True)

        h1, h2 = H//2-(H_orig-r)//2, H//2+(H_orig-r)//2
        w1, w2 = W//2-(W_orig-r)//2, W//2+(W_orig-r)//2
        self.base_interest[:, :, :h1, :] = 0; self.base_interest[:, :, h2:, :] = 0
        self.base_interest[:, :, :, :w1] = 0; self.base_interest[:, :, :, w2:] = 0

        b_max = self.base_interest.max() + 1e-5
        self.base_interest /= b_max
        self._update_interest_map_internal()

    def _update_interest_map_internal(self):
        if self.base_interest is not None:
             self.interest_map = self.base_interest * self.m_trace * self.m_comp

    def post_step_update_interest(self, cx: int, cy: int):
        if self.base_interest is not None:
            H, W = self.base_interest.shape[-2], self.base_interest.shape[-1]
            y_grid = torch.arange(H, device=self.device).view(H, 1)
            x_grid = torch.arange(W, device=self.device).view(1, W)
            dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            sigma = 15.0
            peak = 1.0 * torch.exp(-dist_sq / (2 * sigma**2))
            
            # recovery
            # self.m_trace = 1.0 - (1.0 - self.m_trace) * 0.8
            # suppression
            self.m_trace = torch.clamp(self.m_trace - peak.view(1, 1, H, W)*0.5, 0.0, 1.0)
            self._update_interest_map_internal()

    def _extract_and_match_local_features(self, X_subspaces: Dict[int, torch.Tensor], 
                                          gmem_i: GmemoryI, x: int, y: int, 
                                          similarity_threshold: float = 0.85) -> List[ModalityNode]:
        """自下而上：提取当前注视点的特征，并在 Gmemory I 中匹配或创建新节点"""
        active_nodes = []
        for mod_id, X_m in X_subspaces.items():
            local_vec = X_m[0, :, y, x].unsqueeze(0) # [1, C]
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

    def _compute_spatial_affinity(self, x1, y1, x2, y2, X_subspaces):
        # 简单判定两点在特征图上的欧氏差异作为连通性阻力
        if 1 in X_subspaces:
            h1 = X_subspaces[1][0, :, y1, x1]
            h2 = X_subspaces[1][0, :, y2, x2]
            dist = torch.norm(h1 - h2).item()
            return max(0.0, 1.0 - dist)
        return 1.0

    def _background_memory_consolidation(self, active_gmem1_nodes, gmem_i, gmem_ii, cx, cy, X_subspaces):
        # 提取全局活跃 GmemI 节点，执行赫布共激活
        global_active = [n for n in gmem_i.nodes.values() if n.activation_level >= self.cfg.tau_active]
        for i in range(len(global_active)):
            for j in range(i+1, len(global_active)):
                n1 = global_active[i]
                n2 = global_active[j]
                dw = self.cfg.coactivation_eta * n1.activation_level * n2.activation_level
                gmem_ii.co_activation_matrix[(n1.node_id, n2.node_id)] += dw
                
                # 创世纪 (Genesis) 判断
                if gmem_ii.co_activation_matrix[(n1.node_id, n2.node_id)] > self.cfg.W_genesis_threshold:
                    bound = False
                    for sem_node in gmem_ii.semantic_nodes.values():
                        # [修复] 1. 明确防止生成完全重复的拓扑边 (防止无限增殖)
                        if (sem_node.anchor_id == n1.node_id and n2.node_id in sem_node.peripheral_links) or \
                           (sem_node.anchor_id == n2.node_id and n1.node_id in sem_node.peripheral_links):
                            bound = True
                            break
                        # 2. 原始的活跃度拓扑绑定检查
                        if sem_node.activation_level > self.cfg.tau_recognize:
                            members = [sem_node.anchor_id] + list(sem_node.peripheral_links.keys())
                            if n1.node_id in members or n2.node_id in members:
                                bound = True
                                break
                    if not bound:
                        # 对于在后台自发产生的联系，分配极坐标偏移。
                        # 因为没有精确的两节点在原始空间的对应距离记录，使用占位假设或启发式距离
                        new_sem_node = gmem_ii.add_semantic_node(n1.node_id)
                        gmem_ii.add_peripheral(new_sem_node, n2.node_id, rho_offset=0.5, theta_offset=0.0)
                        new_sem_node.update_activation(self.current_step, 1.0, self.cfg.decay_rate)
                        
                    # [修复] 无论是否 bound，都必须清空共激活累积，防止死循环无限生成
                    gmem_ii.co_activation_matrix[(n1.node_id, n2.node_id)] = 0.0

    def run_step(self, X_subspaces: Dict[int, torch.Tensor], 
                 gmem_i: GmemoryI, gmem_ii: GmemoryII, gpos: Gposition):
        """执行一个周期：Top-Down 匹配与 Bottom-Up 眼跳"""
        self.current_step += 1
        cx, cy = self.fixation_point
        B, C, H, W = list(X_subspaces.values())[0].shape
        cx, cy = max(0, min(cx, W-1)), max(0, min(cy, H-1))

        # [修复] 衰减 top-down 的期待注意力，防止先前注入的高斯峰无限累积
        if self.m_comp is not None:
            self.m_comp = 1.0 + (self.m_comp - 1.0) * 0.8
            self._update_interest_map_internal()

        # -----------------------------
        # 节点时间衰减更新
        # -----------------------------
        for node in gmem_i.nodes.values():
            node.update_activation(self.current_step, 0.0, self.cfg.decay_rate)
        for node in gmem_ii.semantic_nodes.values():
            node.update_activation(self.current_step, 0.0, self.cfg.decay_rate)

        # -----------------------------
        # 底部向上 (Bottom-Up) 特征获取
        # -----------------------------
        active_gmem1_nodes = self._extract_and_match_local_features(X_subspaces, gmem_i, cx, cy)
        for n in active_gmem1_nodes:
            # 激活刚刚捕获的节点
            n.update_activation(self.current_step, self.cfg.activation_gain, self.cfg.decay_rate)

        # -----------------------------
        # 顶部向下 (Top-Down) 全局拓扑检索
        # -----------------------------
        required_nodes = list(gmem_i.nodes.values())
        S_maps = gpos.l1_modality_routing_projection(X_subspaces, required_nodes)
        
        matched_targets = []
        for sem_node in gmem_ii.semantic_nodes.values():
            res = gpos.l2_structure_synthesis(S_maps, sem_node)
            if res is not None:
                matched_targets.append((sem_node, res))
                
        if matched_targets:
            best_match = max(matched_targets, key=lambda x: x[1]['score'])
            if best_match[1]['score'] > 1.5:
                # 顶部向下共振确认，推高语义节点的活跃度
                best_match[0].update_activation(self.current_step, self.cfg.activation_gain, self.cfg.decay_rate)

        # -----------------------------
        # 后台记忆整合 (Hebbian Genesis)
        # -----------------------------
        self._background_memory_consolidation(active_gmem1_nodes, gmem_i, gmem_ii, cx, cy, X_subspaces)
        
        # -----------------------------
        # 信息及状态量观测标量
        # -----------------------------
        I_loc = 0.0
        if self.interest_map is not None:
            r = 10
            x0 = max(0, cx - r); x1 = min(W, cx + r)
            y0 = max(0, cy - r); y1 = min(H, cy + r)
            # [修复] 将 .sum() 替换为 .max()，以合理评估局部是否存在未被彻底抑制的显著特征
            I_loc = float(self.interest_map[0, 0, y0:y1, x0:x1].max().item())
            
        A_star_sem = -1.0
        A_star_node = None
        for k_node in gmem_ii.semantic_nodes.values():
            if k_node.activation_level > A_star_sem:
                A_star_sem = k_node.activation_level
                A_star_node = k_node

        # -----------------------------
        # 控制器效能转移模型判定
        # -----------------------------
        if self.state == SaccadeState.EXPLORE:
            # 如果发现了新颖高亮兴趣点，或者局部特征爆发
            local_gmem1_activation = sum([n.activation_level for n in active_gmem1_nodes])
            # [修复] 阈值因为 sum 改为了 max，需要同步下调至合理区间 (如 0.8)
            if local_gmem1_activation > 1.0 or I_loc > 0.8:
                self.state = SaccadeState.INSPECT
                self.anchor_position = (cx, cy) # 设置审查基准点
                self._execute_saccade_short_range(self.anchor_position[0], self.anchor_position[1])
            else:
                self._execute_saccade_explore()

        elif self.state == SaccadeState.INSPECT:
            if A_star_sem > self.cfg.tau_recognize:
                # 认出了某个已有部件主体，进入校验环节
                self.state = SaccadeState.CONSOLIDATE
                self.active_semantic_id = A_star_node.node_id if A_star_node else None
                
                # 预测其外围坐标
                pred_x, pred_y = self._predict_peripheral_fixation(A_star_node, gmem_ii)
                self._inject_gaussian_peak(pred_x, pred_y, gain=self.cfg.saccade_consolidate_gain)
                self.fixation_point = (pred_x, pred_y)
            else:
                if I_loc < 0.5:
                    # 彻底耗干了局部的未知，是个毫无认知的新物，撤退让后台建图
                    self.state = SaccadeState.EXPLORE
                    self._execute_saccade_explore()
                else:
                    # 继续聚拢提取
                    anch_x, anch_y = self.anchor_position if self.anchor_position else (cx, cy)
                    self._execute_saccade_short_range(anch_x, anch_y)

        elif self.state == SaccadeState.CONSOLIDATE:
            if A_star_sem < self.cfg.tau_recognize:
                # 图坍塌，跌落回探索
                self.state = SaccadeState.EXPLORE
                self._execute_saccade_explore()
            elif I_loc < 0.1:
                # 验证成功并且兴趣探空，完成图校正 (现在基于 max 可以正常触发)
                if self.active_semantic_id is not None:
                    sem_node = gmem_ii.semantic_nodes.get(self.active_semantic_id)
                    if sem_node and self.anchor_position:
                        m_spatial = self._compute_spatial_affinity(self.anchor_position[0], self.anchor_position[1], cx, cy, X_subspaces)
                        self._correct_distortion(sem_node, gmem_ii, active_gmem1_nodes, cx, cy, m_spatial)
                        
                self.state = SaccadeState.EXPLORE
                self._execute_saccade_explore()
            else:
                # 保持制导查询
                pred_x, pred_y = self._predict_peripheral_fixation(A_star_node, gmem_ii)
                self._inject_gaussian_peak(pred_x, pred_y, gain=self.cfg.saccade_consolidate_gain)
                self.fixation_point = (pred_x, pred_y)

    def _correct_distortion(self, sem_node, gmem_ii, active_gmem1_nodes, cx, cy, m_spatial):
        ax, ay = self.anchor_position
        dx, dy = cx - ax, cy - ay
        r_actual = math.sqrt(dx**2 + dy**2) + 1e-5
        rho_actual = math.log(r_actual)
        theta_actual = math.atan2(dy, dx)
        if active_gmem1_nodes:
            peri_id = active_gmem1_nodes[0].node_id
            if peri_id in sem_node.peripheral_links:
                expected = sem_node.peripheral_links[peri_id]
                energy = abs(rho_actual - expected['rho_offset']) + abs(theta_actual - expected['theta_offset'])
                gmem_ii.distortion_correction_and_branch(sem_node, peri_id, rho_actual, theta_actual, energy, m_spatial)

    def _predict_peripheral_fixation(self, sem_node, gmem_ii):
        ax, ay = self.anchor_position if self.anchor_position else self.fixation_point
        if sem_node and sem_node.peripheral_links:
            target_peri_id = list(sem_node.peripheral_links.keys())[0]
            link = sem_node.peripheral_links[target_peri_id]
            pred_r = math.exp(link['rho_offset'])
            pred_theta = link['theta_offset']
            pred_x = int(ax + pred_r * math.cos(pred_theta))
            pred_y = int(ay + pred_r * math.sin(pred_theta))
            return pred_x, pred_y
        return ax, ay

    def _execute_saccade_explore(self):
        """全局探索长程眼跳"""
        if self.interest_map is not None:
            # smoothed = F.avg_pool2d(self.interest_map, kernel_size=15, stride=1, padding=7)
            smoothed = self.interest_map
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
        
        # [修复] 注入到 m_comp (Top-Down 期盼组件)，而不是直接修改会被立即覆盖的 interest_map
        self.m_comp = torch.clamp(self.m_comp + peak.view(1, 1, H, W), 1.0, 10.0)
        self._update_interest_map_internal()
    
    def _execute_saccade_short_range(self, center_x: int, center_y: int):
        """局部短程眼跳，用于 INSPECT 状态下寻找显著外围特征"""
        if self.interest_map is not None:
            H, W = self.interest_map.shape[-2], self.interest_map.shape[-1]
            y_grid = torch.arange(H, device=self.device).view(H, 1)
            x_grid = torch.arange(W, device=self.device).view(1, W)
            dist_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
            
            local_mask = torch.exp(-dist_sq / (2 * 15**2)) 
            noise = torch.rand((1, 1, H, W), device=self.device) * 0.1
            local_interest = self.interest_map * local_mask.view(1, 1, H, W) + noise
            
            idx = torch.argmax(local_interest).item()
            self.fixation_point = (int(idx % W), int(idx // W))


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
        
    def handle_new_view(self, features: dict, H: int = None, W: int = None):
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
            self.controller.initialize_interest_map(X_subspaces)
            
        self.controller.run_step(X_subspaces, self.gmem_i, self.gmem_ii, self.gpos)
        self.current_view = self.controller.fixation_point
        self.viewroute.append(self.current_view)
        self.controller.post_step_update_interest(self.current_view[0], self.current_view[1])
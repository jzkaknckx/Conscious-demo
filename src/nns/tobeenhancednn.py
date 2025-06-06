
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from typing import List, Optional, Tuple, Union, Any, Dict
from collections import deque

class SmoothLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SmoothLayer, self).__init__()
        
        # 定义卷积层：输入通道数为 3，输出通道数为 3，卷积核大小为 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=3, bias=False)
        
        # 定义算子
        # kernel = torch.tensor([[0.025, 0.025, 0.025],
        #                       [0.025, 0.8, 0.025],
        #                       [0.025, 0.025, 0.025]], dtype=torch.float32)
        kernel = torch.full((kernel_size, kernel_size), 1/(kernel_size*kernel_size), dtype=torch.float32)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)
        
        # 设置卷积核权重
        for channel in range(3):
            self.conv.weight.data[channel] = kernel
        
        # 固定卷积核权重，不参与训练
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)
    
class SmoothLayerGauss(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma):
        super(SmoothLayerGauss, self).__init__()
        
        # 定义卷积层：输入通道数为 3，输出通道数为 3，卷积核大小为 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=3, bias=False)
        
        # 定义算子
        # kernel = torch.tensor([[0.025, 0.025, 0.025],
        #                       [0.025, 0.8, 0.025],
        #                       [0.025, 0.025, 0.025]], dtype=torch.float32)
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)
        
        # 设置卷积核权重
        for channel in range(3):
            self.conv.weight.data[channel] = kernel
        
        # 固定卷积核权重，不参与训练
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class EfficientRetinaLayer(nn.Module):
    """
        模拟视网膜映射
        将前一层输入按与中央凹距离调整投射精度,中央直接映射,外部池化
        
        Args:
            output_size: 输出尺寸
            center_size: 中心区域尺寸
            gamma: 池化核衰减速度
            tau: 指数衰减系数
            eps: 防止除0
    
    """
    def __init__(self, 
                 output_size=512, 
                 center_size=256, 
                 gamma=0.8, 
                 tau=0.5, 
                 eps=1e-6):
        super().__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.gamma = gamma  # 池化核衰减速度
        self.tau = tau      # 指数衰减系数
        self.eps = eps
        
        # 预计算坐标映射系统
        self.register_buffer('grid_map', self._precompute_grid_new())

    def _precompute_grid(self):
        """
            预计算输出特征图到输入特征图的坐标映射关系

        Returns:
            # (1, H, W, 2)
        """
        # 生成输出坐标系
        out_y = torch.linspace(-1, 1, self.output_size)
        out_x = torch.linspace(-1, 1, self.output_size)
        grid_y, grid_x = torch.meshgrid(out_y, out_x, indexing='ij')
        
        # 计算极坐标参数
        rho = torch.sqrt(grid_x**2 + grid_y**2)  # 归一化半径
        
        # 初始化输入坐标映射
        in_grid = torch.zeros(self.output_size, self.output_size, 2)
        
        # 中心区域直接映射
        center_mask = rho <= (self.center_size/self.output_size)
        in_grid[center_mask] = torch.stack([grid_x[center_mask], 
                                           grid_y[center_mask]], dim=-1)
        
        # 外围区域动态计算
        outer_mask = ~center_mask
        scaled_rho = rho[outer_mask]
        
        # 指数衰减映射函数
        scale_factor = self.tau ** (scaled_rho * self.output_size / self.center_size)
        scaled_rho = scaled_rho * scale_factor
        
        # 角度保持连续
        theta = torch.atan2(grid_y[outer_mask], grid_x[outer_mask])
        
        # 更新外围坐标
        in_grid[outer_mask] = torch.stack([
            scaled_rho * torch.cos(theta),
            scaled_rho * torch.sin(theta)
        ], dim=-1)
        
        return in_grid.unsqueeze(0)  # (1, H, W, 2)

    def _precompute_grid_new(self):
        # 生成输出坐标系
        out_y = torch.linspace(-1, 1, self.output_size)
        out_x = torch.linspace(-1, 1, self.output_size)
        grid_y, grid_x = torch.meshgrid(out_y, out_x, indexing='ij')
        
        # 计算极坐标参数
        rho = torch.sqrt(grid_x**2 + grid_y**2)  # 归一化半径
        
        # 初始化输入坐标映射
        in_grid = torch.zeros(self.output_size, self.output_size, 2)
        
        # 中心区域直接映射（保持原样）
        center_radius = self.center_size / self.output_size
        center_mask = rho <= center_radius
        
        in_grid[center_mask] = torch.stack([grid_x[center_mask], 
                                        grid_y[center_mask]], dim=-1)
        
        # 过渡区域平滑处理
        transition_width = 0.1  # 过渡带宽度比例
        transition_start = center_radius * (1 - transition_width)
        transition_end = center_radius * (1 + transition_width)
        
        # 计算平滑过渡权重（sigmoid函数）
        transition_ratio = (rho - transition_start) / (transition_end - transition_start)
        blend_weight = torch.sigmoid(10 * (transition_ratio - 0.5))  # 陡峭过渡
        
        # 外围区域坐标变换（保持原逻辑但应用平滑）
        outer_mask = rho > transition_start
        scaled_rho = rho[outer_mask]
        
        # 原始缩放因子
        original_scale = self.tau ** (scaled_rho * self.output_size / self.center_size)
        
        # 混合缩放因子
        blended_scale = 1 * (1 - blend_weight[outer_mask]) + original_scale * blend_weight[outer_mask]
        scaled_rho = scaled_rho * blended_scale
        
        # 极坐标转笛卡尔（保持角度连续）
        theta = torch.atan2(grid_y[outer_mask], grid_x[outer_mask])
        in_grid[outer_mask] = torch.stack([
            scaled_rho * torch.cos(theta),
            scaled_rho * torch.sin(theta)
        ], dim=-1)
        
        return in_grid.unsqueeze(0)

    def _dynamic_pooling(self, x):
        B, C, H, W = x.shape
        
        # 生成采样网格
        grid = self.grid_map.repeat(B, 1, 1, 1)
        
        # 自适应尺寸调整
        if H != W or H != self.output_size:
            grid = grid * torch.tensor([W/self.output_size, 
                                       H/self.output_size], 
                                      device=x.device)
        
        # 双线性采样+自适应池化
        sampled = F.grid_sample(x, grid, align_corners=False)
        
        # 动态模糊核
        # blur_kernel = self._get_blur_kernel(C, H, W)
        
        '''
        test
        '''
        straight_kernel = torch.tensor([1], dtype=torch.float32)
        straight_kernel = straight_kernel.view(1, 1, 1, 1).repeat(3, 1, 1, 1)
        '''
        end
        '''
                
        # return F.conv2d(sampled, blur_kernel, padding='same', groups=3), F.conv2d(sampled, straight_kernel, padding='same', groups=3)
        return sampled

    def _get_blur_kernel(self, C, H, W):
        # 基于相对位置的动态核尺寸
        max_radius = min(H, W) // 2
        kernel_size = int(max_radius ** self.gamma) | 1  # 保证奇数
        
        # 生成高斯核
        sigma = kernel_size / 3.0
        ax = torch.arange(-kernel_size//2, kernel_size//2, dtype=torch.float32)     #change1
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        kernel /= kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    def forward(self, x):
        # 尺寸预处理
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(x, size=self.output_size, mode='bilinear')
        
        # 动态池化处理
        return self._dynamic_pooling(x)

class EdgeFlowLayer(nn.Module):
    def __init__(self, flowLayerCacheMax):
        super().__init__()
        self.flowLayerCacheMax = flowLayerCacheMax
        self.grad_x_cache = []
        self.grad_y_cache = []
        
        kernel_melted = self.kernel_generate(flowLayerCacheMax)[0]
        kernel_frozen = self.kernel_generate(flowLayerCacheMax)[1]
        self.register_buffer('kernel_melted', kernel_melted)
        self.register_buffer('kernel_frozen', kernel_frozen)
    
    def kernel_generate(self, kernel_size):
        
        assert kernel_size >= 3 , "Kernel too Small."
        ODD = bool(kernel_size % 2)
        size = int((kernel_size - 1) / 2) if ODD else int(kernel_size / 2)
        
        kernel_f_partial = kernel_m_partial = torch.zeros([size, size])
        for i in range(size):
            for j in range(size):
                kernel_m_partial[i, j] = size - math.fabs(i-j)
                kernel_f_partial[i, j] = j - i
                
        # kernel_melted generation
        # [size, size] -> [size, size * 2] or [size, size * 2 + 1] -> [size * 2, size * 2] or [size * 2 + 1, size * 2 + 1]
        kernel_melted = torch.cat([kernel_m_partial, torch.zeros(size, int(ODD)), (torch.flip(kernel_m_partial, dims=[1])) * (-1)], dim=1) 
        kernel_melted = torch.cat([kernel_melted, torch.zeros(int(ODD), size * 2 + int(ODD)), torch.flip(kernel_melted, dims=[0, 1])], dim=0)
        
        # kernel_frozen generation
        # [size, size] -> [size, size * 2] or [size, size * 2 + 1] -> [size * 2, size * 2] or [size * 2 + 1, size * 2 + 1]
        kernel_frozen = torch.cat([kernel_f_partial, torch.ones(size, int(ODD)) * (size), (torch.flip(kernel_f_partial, dims=[1]))], dim=1) 
        kernel_frozen = torch.cat([kernel_frozen, torch.ones(int(ODD), size * 2 + int(ODD)) * (-1 * size), torch.flip(kernel_frozen, dims=[0])], dim=0)
        if ODD : kernel_frozen[size, size] = 0
        
        return kernel_melted, kernel_frozen

    def forward(self, grad_x, grad_y):
        # 输入缓存
        self.grad_x_cache.append(grad_x)
        self.grad_y_cache.append(grad_y)
        
        if len(self.grad_x_cache) > self.flowLayerCacheMax:
            # 更新缓存
            self.grad_x_cache.pop(0)
            self.grad_y_cache.pop(0)
            
            # grad.shape = [1, C, Cache, H, W]
            grad_x_tensor = torch.stack(self.grad_x_cache, dim=2)
            Bx, Cx, Tx, Hx, Wx = grad_x_tensor.shape
            grad_y_tensor = torch.stack(self.grad_y_cache, dim=2)
            By, Cy, Ty, Hy, Wy = grad_y_tensor.shape
            assert (Bx, Cx, Hx, Wx, Tx) == (By, Cy, Hy, Wy, Ty), "Grad_x Does not Match Grad_y."
            
            # 时序卷积
            kernel_melted, kernel_frozen = self.kernel_generate(self.flowLayerCacheMax) # [Cache, Cache] 
            kernel_melted_x = kernel_melted.view(Bx, 1, self.flowLayerCacheMax, 1, self.flowLayerCacheMax).repeat(1, Cx, 1, 1, 1) # [1, C, Cache, 1, Cache]
            kernel_melted_y = kernel_melted.view(Bx, 1, self.flowLayerCacheMax, self.flowLayerCacheMax, 1).repeat(1, Cx, 1, 1, 1) # [1, C, Cache, Cache, 1]
            kernel_frozen_x = kernel_frozen.view(Bx, 1, self.flowLayerCacheMax, 1, self.flowLayerCacheMax).repeat(1, Cx, 1, 1, 1) # [1, C, Cache, 1, Cache]
            kernel_frozen_y = kernel_frozen.view(Bx, 1, self.flowLayerCacheMax, self.flowLayerCacheMax, 1).repeat(1, Cx, 1, 1, 1) # [1, C, Cache, Cache, 1]
            
            melted_x = F.conv3d(grad_x_tensor, kernel_melted_x, padding=0, groups=Cx).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            melted_y = F.conv3d(grad_y_tensor, kernel_melted_y, padding=0, groups=Cx).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            frozen_x = F.conv3d(grad_x_tensor, kernel_frozen_x, padding=0, groups=Cx).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            frozen_y = F.conv3d(grad_y_tensor, kernel_frozen_y, padding=0, groups=Cx).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            
            return melted_x, melted_y, frozen_x, frozen_y
            
        return None

class EdgeFlowLayerN(nn.Module):
    def __init__(self, flowLayerCacheMax=5, EdgeFlowLayerNtau = 10):
        super().__init__()
        self.flowLayerCacheMax = flowLayerCacheMax
        self.EdgeFlowLayerNtau = EdgeFlowLayerNtau
        self.grad_cache = []
        
        kernel_melted = self.kernel_generateN(flowLayerCacheMax, EdgeFlowLayerNtau)[0]
        kernel_frozen = self.kernel_generateN(flowLayerCacheMax, EdgeFlowLayerNtau)[1]
        self.register_buffer('kernel_melted', kernel_melted)
        self.register_buffer('kernel_frozen', kernel_frozen)
    
    '''
    def kernel_generate(self, kernel_size):
        
        assert kernel_size >= 3 , "Kernel too Small."
        ODD = bool(kernel_size % 2)
        size = int((kernel_size - 1) / 2) if ODD else int(kernel_size / 2)
        
        kernel_f_partial = kernel_m_partial = torch.zeros([size, size])
        for i in range(size):
            for j in range(size):
                kernel_m_partial[i, j] = size - math.fabs(i-j)
                kernel_f_partial[i, j] = j - i
                
        # kernel_melted generation
        # [size, size] -> [size, size * 2] or [size, size * 2 + 1] -> [size * 2, size * 2] or [size * 2 + 1, size * 2 + 1]
        kernel_melted = torch.cat([kernel_m_partial, torch.zeros(size, int(ODD)), (torch.flip(kernel_m_partial, dims=[1])) * (-1)], dim=1) 
        kernel_melted = torch.cat([kernel_melted, torch.zeros(int(ODD), size * 2 + int(ODD)), torch.flip(kernel_melted, dims=[0, 1])], dim=0)
        
        # kernel_frozen generation
        # [size, size] -> [size, size * 2] or [size, size * 2 + 1] -> [size * 2, size * 2] or [size * 2 + 1, size * 2 + 1]
        kernel_frozen = torch.cat([kernel_f_partial, torch.ones(size, int(ODD)) * (size), (torch.flip(kernel_f_partial, dims=[1]))], dim=1) 
        kernel_frozen = torch.cat([kernel_frozen, torch.ones(int(ODD), size * 2 + int(ODD)) * (-1 * size), torch.flip(kernel_frozen, dims=[0])], dim=0)
        if ODD : kernel_frozen[size, size] = 0
                  
        return kernel_melted, kernel_frozen
    '''
    
    def kernel_generateN(self,kernel_size, EdgeFlowLayerNtau):  
        assert kernel_size >= 3 , "Kernel too Small."
        # if EdgeFlowLayerNtau == -1 :
            
        
        ODD = bool(kernel_size % 2)
        size = int((kernel_size - 1) / 2) if ODD else int(kernel_size / 2)
        # grid
        kernel_f_stick = torch.exp(EdgeFlowLayerNtau / torch.arange(1, size + 1, dtype=torch.float32)).view(-1, 1) - torch.ones(size, 1)
        i, j = torch.meshgrid(torch.arange(1, size + 1, dtype=torch.float32), torch.arange(1, size + 1, dtype=torch.float32), indexing='ij')
        
        # elements
        i_sq = i ** 2
        j_sq = j ** 2
        denominator = (i_sq + j_sq) ** 1.5
        abs_diff_sq = torch.abs(i_sq - j_sq)                                                  # i ** 2 - j ** 2
        exponent_f = EdgeFlowLayerNtau * abs_diff_sq / denominator                                          # tau * (i ** 2 - j ** 2) / (i**2 + j**2) ** 1.5
        times_ij = i * j                                                                      # i ** 2 - j ** 2
        exponent_m = EdgeFlowLayerNtau * times_ij / denominator                                             # tau * (i * j) / (i**2 + j**2) ** 1.5
        
        # kernel_f_partial
        factor_f = torch.where(i > j, 1.0, -1.0)                                              # mask i != j
        values_f = factor_f * (torch.exp(exponent_f) - torch.ones_like(exponent_f))           # kernel_f(mask)
        mask_f_i_eq_j = (i == j)                                                              # mask i == j
        kernel_f_partial = torch.where(mask_f_i_eq_j, torch.zeros_like(values_f), values_f)   # kernel_f(full)
        
        kernel_f_partial = torch.flip(kernel_f_partial, dims=[0, 1])
        kernel_f_stick = torch.flip(kernel_f_stick, dims=[0])
        
        # kernel_frozen generation
        if ODD : 
            # [size, size] -> [size, size * 2 + 1]
            kernel_frozen = torch.cat([kernel_f_partial, kernel_f_stick, (torch.flip(kernel_f_partial, dims=[1]))], dim=1)
            # [1, size * 2 + 1]
            kernel_f_stick = kernel_f_stick.reshape(1, size) * (-1)
            kernel_f_bar = torch.cat([kernel_f_stick, torch.zeros(1, 1), torch.flip(kernel_f_stick, dims=[1])], dim = 1)
            #[size, size * 2 + 1] -> [size * 2 + 1, size * 2 + 1]
            kernel_frozen = torch.cat([kernel_frozen, kernel_f_bar, torch.flip(kernel_frozen, dims=[0])], dim=0)
            
        else :
            # [size, size] -> [size, size * 2]
            kernel_frozen = torch.cat([kernel_f_partial, (torch.flip(kernel_f_partial, dims=[1]))], dim=1)
            # [size, size * 2] -> [size * 2, size * 2]
            kernel_frozen = torch.cat([kernel_frozen, torch.flip(kernel_frozen, dims=[0])], dim=0)
        
        # max -> 1
        kernel_frozen = kernel_frozen / torch.max(kernel_frozen)
        
        # kernel_m_partial                      
        kernel_m_partial = torch.exp(exponent_m)
        kernel_m_partial = torch.flip(kernel_m_partial, dims=[0, 1])                                   
        kernel_m_partial = kernel_m_partial - torch.ones_like(exponent_m)
        
        # kernel_melted generation
        # [size, size] -> [size, size * 2] or [size, size * 2 + 1] -> [size * 2, size * 2] or [size * 2 + 1, size * 2 + 1]
        kernel_melted = torch.cat([kernel_m_partial, torch.zeros(size, int(ODD)), (torch.flip(kernel_m_partial, dims=[1])) * (-1)], dim=1) 
        kernel_melted = torch.cat([kernel_melted, torch.zeros(int(ODD), size * 2 + int(ODD)), torch.flip(kernel_melted, dims=[0, 1])], dim=0)
        
        # max -> 1
        kernel_melted = kernel_melted / torch.max(kernel_melted)
        
        return kernel_melted, kernel_frozen

class KalmanOpticalFlowLayer(nn.Module):
    def __init__(self, img_channels=3, state_dim=4, obs_dim=2):
        """
        卡尔曼滤波光流计算层
        
        :param img_channels: 输入图像的通道数
        :param state_dim: 状态向量维度（这里设计为[x位移, y位移, x速度, y速度]）
        :param obs_dim: 观测向量维度（对应光流的直接观测值维度）
        """
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # 卡尔曼滤波参数（可学习或固定）
        self.F = nn.Parameter(torch.eye(state_dim), requires_grad=True)  # 状态转移矩阵（初始假设匀速运动）
        self.H = nn.Parameter(torch.randn(obs_dim, state_dim), requires_grad=True)  # 观测矩阵
        self.Q = nn.Parameter(torch.eye(state_dim) * 1e-3, requires_grad=True)  # 过程噪声协方差
        self.R = nn.Parameter(torch.eye(obs_dim) * 1e-2, requires_grad=True)  # 观测噪声协方差
        
        # 观测提取网络（从图像和边缘检测结果中提取观测值）
        self.obs_extractor = nn.Sequential(
            nn.Conv2d(img_channels*2 + 1, 32, kernel_size=3, padding=1),  # 输入包含当前帧、前一帧和边缘图
            nn.ReLU(),
            nn.Conv2d(32, obs_dim, kernel_size=3, padding=1)  # 输出观测值（对应光流的直接估计）
        )

    def _initialize_state(self, x):
        """初始化状态：首张图像无运动，速度为0"""
        B, T, C, H, W = x.shape  # 输入形状 [B, 时间步T, C, H, W]
        init_state = torch.zeros(B, self.state_dim, H, W, device=x.device)
        init_cov = torch.eye(self.state_dim, device=x.device).view(1, self.state_dim, self.state_dim, 1, 1)
        init_cov = init_cov.repeat(B, 1, 1, H, W)  # 初始协方差矩阵
        return init_state, init_cov

    def forward(self, imgs, edges):
        """
        :param imgs: 输入图像序列 [B, T, C, H, W]
        :param edges: 对应边缘检测结果 [B, T, 1, H, W]（假设单通道边缘图）
        :return: 光流场序列 [B, T-1, 2, H, W]（每个时间步的xy位移）
        """
        B, T, C, H, W = imgs.shape
        assert edges.shape == (B, T, 1, H, W), "边缘图形状不匹配"
        
        # 初始化状态和协方差
        state, cov = self._initialize_state(imgs)
        optical_flows = []

        for t in range(1, T):
            # 前一帧和当前帧数据
            prev_img = imgs[:, t-1]  # [B, C, H, W]
            curr_img = imgs[:, t]
            prev_edge = edges[:, t-1]
            curr_edge = edges[:, t]
            
            # 步骤1：预测（时间更新）
            pred_state = (self.F @ state.view(B, self.state_dim, -1)).view(B, self.state_dim, H, W)  # F*state
            pred_cov = self.F @ cov @ self.F.transpose(-2, -1) + self.Q  # F*P*F^T + Q

            # 步骤2：观测提取（从图像对和边缘图中计算观测光流）
            obs_input = torch.cat([prev_img, curr_img, prev_edge, curr_edge], dim=1)  # 拼接前后帧+前后边缘
            obs_flow = self.obs_extractor(obs_input)  # [B, obs_dim, H, W]（直接观测的光流）

            # 步骤3：更新（测量更新）
            residual = obs_flow - (self.H @ pred_state.view(B, self.state_dim, -1)).view(B, self.obs_dim, H, W)
            S = self.H @ pred_cov @ self.H.transpose(-2, -1) + self.R  # H*P*H^T + R
            K = pred_cov @ self.H.transpose(-2, -1) @ torch.inverse(S)  # 卡尔曼增益
            state = pred_state + (K @ residual.view(B, self.obs_dim, -1)).view(B, self.state_dim, H, W)
            cov = (torch.eye(self.state_dim, device=imgs.device) - K @ self.H) @ pred_cov

            # 记录当前时间步的光流（取状态中的位移分量）
            optical_flows.append(state[:, :2])  # 状态前两维是xy位移

        return torch.stack(optical_flows, dim=1)  # [B, T-1, 2, H, W]

    def forward(self, grad):
        
        dim = len(grad)
        if dim == 2:
            grad_x, grad_y = grad
            grad_tensor = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        else:
            grad_tensor = grad
        self.grad_cache.append(grad_tensor)
        
        if len(self.grad_cache) > self.flowLayerCacheMax:
            # 更新缓存
            self.grad_cache.pop(0)
            
            # grad.shape = [1, C, Cache, H, W]
            grad_tensor = torch.stack(self.grad_cache, dim=2)
            B, C, T, H, W = grad_tensor.shape
            
            # 时序卷积
            kernel_melted_x = self.kernel_melted.view(B, 1, self.flowLayerCacheMax, 1, self.flowLayerCacheMax).repeat(1, C, 1, 1, 1) # [1, C, Cache, 1, Cache]
            kernel_melted_y = self.kernel_melted.view(B, 1, self.flowLayerCacheMax, self.flowLayerCacheMax, 1).repeat(1, C, 1, 1, 1) # [1, C, Cache, Cache, 1]
            kernel_frozen_x = self.kernel_frozen.view(B, 1, self.flowLayerCacheMax, 1, self.flowLayerCacheMax).repeat(1, C, 1, 1, 1) # [1, C, Cache, 1, Cache]
            kernel_frozen_y = self.kernel_frozen.view(B, 1, self.flowLayerCacheMax, self.flowLayerCacheMax, 1).repeat(1, C, 1, 1, 1) # [1, C, Cache, Cache, 1]
            
            melted_x = F.conv3d(grad_tensor, kernel_melted_x, padding=0, groups=C).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            melted_y = F.conv3d(grad_tensor, kernel_melted_y, padding=0, groups=C).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            frozen_x = F.conv3d(grad_tensor, kernel_frozen_x, padding=0, groups=C).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            frozen_y = F.conv3d(grad_tensor, kernel_frozen_y, padding=0, groups=C).squeeze(2) # [1, C, 1, H, W] -> [1, C, H, W]
            
            return melted_x, melted_y, frozen_x, frozen_y
            
        return None
    
class EnhancedEdgeDetectionLayer(nn.Module): # to do 
    def __init__(self, R=3, alpha=1.0, beta=1.0, angel_sensiticvity=0.1):
        """
        初始化 EnhancedEdgeDetectionLayer 模块。

        参数:
            R (int): LateralField 的半径，定义周围像素的影响范围，默认值为 3。
            alpha (float): 距离衰减参数，控制远距离像素的影响衰减速度，默认值为 1.0。
            beta (float): 邻域影响强度，控制周围像素对中心像素的增强或削弱程度，默认值为 1.0。
        """
        super(EnhancedEdgeDetectionLayer, self).__init__()
        self.R = R  # LateralField 的半径
        self.alpha = alpha  # 距离衰减参数
        self.beta = beta  # 邻域影响强度
        self.angel_sensiticvity = angel_sensiticvity

        # 生成偏移量列表，排除 (0,0)，表示中心像素以外的邻域偏移
        self.offsets = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if not (dx == 0 and dy == 0)]

        # 预先计算每个偏移的距离 d 和方向 phi
        self.d = {}
        self.phi = {}
        for dx, dy in self.offsets:
            d = math.sqrt(dx**2 + dy**2)  # 欧几里得距离
            phi = math.atan2(dy, dx)  # 偏移方向，范围 [-pi, pi]
            self.d[(dx, dy)] = d
            self.phi[(dx, dy)] = phi

    def forward(self, grad_x, grad_y):
        """
        前向传播函数，计算增强后的边缘强度。

        输入:
            grad_x (torch.Tensor): x 方向的梯度，形状为 (B, 1, H, W)。
            grad_y (torch.Tensor): y 方向的梯度，形状为 (B, 1, H, W)。

        输出:
            enhanced (torch.Tensor): 增强后的边缘强度，形状为 (B, 1, H, W)。
        """
        # 获取输入张量的形状
        B, _, H, W = grad_x.shape

        # 计算每个像素的边缘强度和方向
        strength = torch.sqrt(grad_x**2 + grad_y**2)  # 边缘强度，(B, 1, H, W)
        theta = torch.atan2(grad_y, grad_x)  # 梯度方向，(B, 1, H, W)，范围 [-pi, pi]

        # 初始化增强后的输出张量
        enhanced = torch.zeros_like(strength)  # (B, 1, H, W)

        # 对每个 batch 单独处理
        for b in range(B):
            # 初始化当前 batch 的总影响
            sum_influence = torch.zeros_like(strength[b, 0])  # (H, W)

            # 对每个偏移计算邻域影响
            for dx, dy in self.offsets:
                d = self.d[(dx, dy)]  # 预计算的距离
                phi = self.phi[(dx, dy)]  # 预计算的方向

                # 计算邻域像素的坐标 (k, l)
                i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                k = i + dx
                l = j + dy

                valid = (k >= 0) & (k < H) & (l >= 0) & (l < W)

                # 获取邻域像素的边缘强度 strength_kl
                strength_kl = torch.zeros_like(strength[b, 0])  # (H, W)
                strength_kl[valid] = strength[b, 0, k[valid], l[valid]]

                # 计算方向差异的余弦项 cos(2 * (phi - theta))
                print(theta.shape)
                mask_theta = torch.abs(phi - theta[b, 0]) < self.angel_sensiticvity or (torch.abs(torch.abs(phi - theta[b, 0]) - math.pi) < self.angel_sensiticvity)
                mask_theta = torch.where((torch.abs(phi - theta[b, 0]) < self.angel_sensiticvity or (torch.abs(torch.abs(phi - theta[b, 0]) - math.pi) < self.angel_sensiticvity)), 1, 0)
                mask_theta = torch.where((torch.abs(torch.abs(phi - theta[b, 0]) - math.pi / 2) < self.angel_sensiticvity), -1, mask_theta)
                cos_term = torch.cos(2 * (phi - theta[b, 0]))  # (H, W)

                # 计算邻域影响 influence
                influence = strength_kl * math.exp(-d * self.alpha) * cos_term

                # 累加到总影响
                sum_influence += influence

            # 计算增强后的边缘强度
            enhanced[b, 0] = strength[b, 0] + self.beta * sum_influence

        return enhanced
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math

from trail.pic_tools import *
from trail.matrix_tools import *

torch.set_printoptions(profile="full",linewidth=512)

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)

class OpticalflowLayer(nn.Module):
    def __init__(self, window_size=5, sigma=1.0):
        """
        初始化 OpticalflowLayer。
        
        参数：
        - window_size: 邻域窗口大小（默认 5x5）
        - sigma: 高斯平滑的标准差
        """
        super(OpticalflowLayer, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        # 创建高斯核用于平滑梯度
        self.gaussian_kernel = self._create_gaussian_kernel(window_size, sigma)

    def _create_gaussian_kernel(self, size, sigma):
        """创建二维高斯核"""
        ax = torch.arange(size, dtype=torch.float32)
        ax = ax - size // 2
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, images, grad_x_list, grad_y_list):
        """
        前向传播，计算光流。
        
        输入：
        - images: [N, C, H, W]，图像序列，N >= 2
        - grad_x_list: [N, C, H, W]，水平梯度序列
        - grad_y_list: [N, C, H, W]，竖直梯度序列
        
        输出：
        - flow: [1, H, W, 2]，光流场 (delta_xout, delta_yout)
        """
        assert images.size(0) >= 2, "图像序列至少需要两张图像"
        
        # 提取最后两张图像及其梯度
        I1 = images[-2]  # 前一张图像
        I2 = images[-1]  # 当前图像
        I_x = grad_x_list[-1]  # 当前图像的水平梯度
        I_y = grad_y_list[-1]  # 当前图像的竖直梯度
        
        # 计算时间梯度
        I_t = I2 - I1
        
        # 对梯度进行高斯平滑
        I_x = F.conv2d(I_x, self.gaussian_kernel, padding=self.window_size//2, groups=I_x.size(1))
        I_y = F.conv2d(I_y, self.gaussian_kernel, padding=self.window_size//2, groups=I_y.size(1))
        # I_t = F.conv2d(I_t, self.gaussian_kernel, padding=self.window_size//2, groups=I_t.size(1))
        
        # 计算光流
        u, v = self._compute_flow(I_x, I_y, I_t)
        
        # 格式化输出为 [1, H, W, 2]
        flow = torch.stack([u, v], dim=-1).squeeze(0).squeeze(0).permute(1, 0, 2)  # [H, W, 2]
        return flow.unsqueeze(0)  # [1, H, W, 2]

    def _compute_flow(self, I_x, I_y, I_t):
        """
        使用 Lucas-Kanade 方法计算光流。
        
        输入：
        - I_x, I_y, I_t: [B, C, H, W]，空间和时间梯度
        
        输出：
        - u, v: [B, 1, H, W]，水平和竖直光流分量
        """
        B, C, H, W = I_x.shape
        # 使用 unfold 提取邻域 patches
        unfold = nn.Unfold(kernel_size=self.window_size, padding=self.window_size//2)
        
        I_x_patches = unfold(I_x).view(B, 1, self.window_size**2, H, W).repeat(1, 3, 1, 1, 1)
        I_y_patches = unfold(I_y).view(B, 1, self.window_size**2, H, W).repeat(1, 3, 1, 1, 1)
        I_t_patches = unfold(I_t).view(B, 3, self.window_size**2, H, W)
        
        # 计算 Lucas-Kanade 方程的 A 矩阵和 b 向量 # [B, C, H, W]
        A11 = (I_x_patches ** 2).sum(dim=2)  
        A12 = (I_x_patches * I_y_patches).sum(dim=2)
        A22 = (I_y_patches ** 2).sum(dim=2)
        b1 = -(I_x_patches * I_t_patches).sum(dim=2)
        b2 = -(I_y_patches * I_t_patches).sum(dim=2)
        
        # 计算行列式，避免除以零
        det = A11 * A22 - A12 ** 2
        det = torch.where(det == 0, torch.tensor(1e-10, device=det.device), det)
        
        # 求解光流 u, v
        u = (A22 * b1 - A12 * b2) / det
        v = (A11 * b2 - A12 * b1) / det
        
        # 平均跨通道
        u = u.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        v = v.mean(dim=1, keepdim=True)
        
        return u, v

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

    def forward(self, grad):
        
        dim = len(grad)
        if dim == 2:
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

class EnhancedEdgeDetectionLayer(nn.Module):
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
    
class LateralInfulentialOpticalFlowLayer(nn.Module):
    def __init__(self, lateralField=5, flowLayerCache=5, alpha=1.0, tau=1.0):
        """
        初始化 LateralInfulentialOpticalFlowLayer 模块。

            参数:
                LateralField (int): LateralField 的半径，光流节点的感受野，默认值为 5
                alpha (float): 距离衰减参数，控制节点对周围像素的敏感程度，默认值为
                beta (float): 时间衰减参数, 默认值为
        """
        super(LateralInfulentialOpticalFlowLayer, self).__init__()
        self.lateralField = lateralField  # LateralField 的半径
        self.alpha = alpha  # 距离衰减参数
        self.tau = tau  # 时间衰减参数
        self.flowLayerCache = flowLayerCache # FlowLayer 的缓存

        self.absx = None
        self.Cache = {}
        
        self.decayWithTime = [math.exp(- (t) * 1) for t in range(1, flowLayerCache + 1)]
        
        self.offsets = [dx for dx in range(-lateralField, lateralField+1) if not (dx == 0)]

    def forward(self, x):
        """
        前向传播函数，计算光流的xy分量。

            Warning:
                仅支持单批次输入
            
            输入:
                支持:
                    grad_x (torch.Tensor): (B, 1, H, W) + grad_y (torch.Tensor): (B, 1, H, W)
                    grad (torch.Tensor): (B, 1, H, W)
                    RGB (torch.Tensor): (B, 3, H, W)
                            B == 1
            输出:
            
        """
        # 输入 -> (B, 1, H, W)
        togray = T.Grayscale()
        
        dim = len(x)
        if dim == 2:
            grad_x, grad_y = x
            B, C, H, W = grad_x.shape
            x = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        elif dim == 1:
            B, C, H, W = x.shape
            if C == 3:
                x = togray(x)
        
        assert B == 1, "Only One Batch Once"
        
        # 插帧存入 Cache
        runFrame = len(self.Cache)
        if runFrame == 0:   self.absx = x
        else:               self.absx = torch.abs(x - self.absx)
        self.Cache.append(self.absx)
        if runFrame > self.FlowLayerCache:  self.Cache.pop(0)
        
        
        # 初始化当前 batch 的总影响
        sum_influence_x = sum_influence_y = torch.zeros(1, 1, H, W)  # (H, W)

        # 对每个偏移计算邻域影响
        for dx in self.offsets:

            # 邻域像素的坐标 (k, l)
            i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            k = i + dx
            l = j + dx

            valid_x = (k >= 0) & (k < H)
            valid_y = (l >= 0) & (l < W)

            # 获取邻域像素的边缘强度 strength_kl
            strength_kj = strength_il = torch.zeros(1, 1, H, W, self.FlowLayerCache)
            strength_kj[valid_x] = self.absx[0, 0, k[valid_x], j, :]
            strength_il[valid_y] = self.absx[0, 0, i, l[valid_y], :]

            influence_x = influence_y = torch.zeros(1, 1, H, W)
            for t in range(self.flowLayerCache):
               influence_x += self.decayWithTime[self.flowLayerCache - t - 1] * strength_kj[:, :, :, :, self.flowLayerCache - t - 1]
               influence_y += self.decayWithTime[self.flowLayerCache - t - 1] * strength_il[:, :, :, :, self.flowLayerCache - t - 1]

            # 累加到总影响
            sum_influence_x += influence_x * self.decayWithDistance[dx - 1]
            sum_influence_y += influence_y * self.decayWithDistance[dx - 1]

        return sum_influence_x, sum_influence_y

class RetinaModel(nn.Module):
    def __init__(self, cropped_size=1024, output_size=512, center_size=256, tau=0.01, flowLayerCacheMax=5, EdgeFlowLayerNtau = 10):
        super().__init__()
        self.preprocess = PreprocessLayer(cropped_size)
        self.projection = ProjectionLayer(output_size, center_size, tau)
        self.edge_detection = EdgeDetectionLayer()
        self.edge_flow = EdgeFlowLayerN(flowLayerCacheMax, EdgeFlowLayerNtau)
        
    def forward(self, x, center_x, center_y):
        x = self.preprocess(x, center_x, center_y)
        x = self.projection(x)
        grad_x, grad_y = self.edge_detection(x)
        velocity = self.edge_flow(grad_x, grad_y)
        
        return x, grad_x, grad_y, velocity

class RetinaModelO(nn.Module):
    def __init__(self, cropped_size=1024, output_size=512, center_size=256, tau=0.01, window_size=5, sigma=1.0, OpticalflowLayerCacheMax=10):
        super(RetinaModel, self).__init__()
        self.preprocess = PreprocessLayer(cropped_size)
        self.projection = ProjectionLayer(output_size, center_size, tau)
        self.edge_detection = EdgeDetectionLayer()
        self.optical_flow = OpticalflowLayer(window_size, sigma)
        self.OpticalflowLayerCacheMax = OpticalflowLayerCacheMax
        self.image_cache = []
        self.grad_x_cache = []
        self.grad_y_cache = []

    def forward(self, x, center_x, center_y):
        """
        前向传播。
        
        输入：
        - x: [B, C, H, W]，当前输入图像
        - crop_size: 裁切大小
        - center: 裁切中心点 (c_x, c_y)
        
        输出：
        - output: 当前图像处理结果
        - flow: 光流场 [1, H, W, 2] 或 None
        """
        # 处理当前图像
        x = self.preprocess(x, center_x, center_y)
        x = self.projection(x)
        grad_x, grad_y = self.edge_detection(x)
        
        # 更新缓存
        self.image_cache.append(x)
        self.grad_x_cache.append(grad_x)
        self.grad_y_cache.append(grad_y)
        
        # 控制缓存大小
        if len(self.image_cache) > self.OpticalflowLayerCacheMax:
            self.image_cache.pop(0)
            self.grad_x_cache.pop(0)
            self.grad_y_cache.pop(0)
        
        # 计算光流
        if len(self.image_cache) >= 2:
            images = torch.stack(self.image_cache, dim=0)
            grad_x_list = torch.stack(self.grad_x_cache, dim=0)
            grad_y_list = torch.stack(self.grad_y_cache, dim=0)
            flow = self.optical_flow(images, grad_x_list, grad_y_list)
            return x, flow
        return x, None  
    
# 验证测试
if __name__ == "__main__":
    '''
    较好的tau
        cropped_size = 1024
        output_size = 512
        center_size = 256
        tau = 0.01
    '''
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    
    # 准备输入
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # [1, 3, 1280, 1920]
    tensor, oringinal_image = image_to_tensor(image_path)
    # axes[0,0].imshow(oringinal_image)
    # axes[0,0].set_title('')
    
    cropped_size = 1024
    output_size = 512
    center_size = 256
    tau = 0.01
    flowLayerCacheMax=10
    EdgeFlowLayerNtau = 100
    
    preprocess = PreprocessLayer(cropped_size) 
    projection = ProjectionLayer(output_size, center_size, tau)
    edge_detection = EdgeDetectionLayer()
    layer = EnhancedEdgeDetectionLayer(R=10, alpha=2, beta=2)
    edge_flow1 = EdgeFlowLayerN()
    edge_flow2 = EdgeFlowLayerN()
    
    for i in range(1,15):
        x = preprocess(tensor, -1, -1)
        x = projection(x)
        grad_x, grad_y = edge_detection(x)
        enhanced = layer(grad_x, grad_y)
        velocity1 = edge_flow1([grad_x, grad_y])
        velocity2 = edge_flow2(enhanced)
    
    axes[0,0].imshow(edge_to_image(torch.sqrt(grad_x**2 + grad_y**2)))
    axes[0,1].imshow(edge_to_image(enhanced))
    axes[1,0].imshow(edge_to_image(velocity1[2]))
    axes[1,1].imshow(edge_to_image(velocity2[2]))
    
    '''
    model = RetinaModel()
    # output, _, _, _ = model(tensor, 960, 640)
    # axes[0,1].imshow(tensor_to_image(output))
    # axes[0,1].set_title('')
    
    output, grid_x, gris_y, _ = model(tensor, 970, 650)
    # axes[1,0].imshow(tensor_to_image(output))
    grid = torch.sqrt(grid_x**2 + gris_y**2)
    
    for i in range(1,15):
        output, _, _, velocity = model(tensor, 970, 650)
    
    # print(velocity[0].var(), torch.max(velocity[0]), torch.min(velocity[0]))
    # print(velocity[1].var(), torch.max(velocity[1]), torch.min(velocity[1]))
    print(velocity[2].var(), torch.max(velocity[2]), torch.min(velocity[2]))
    print(velocity[3].var(), torch.max(velocity[3]), torch.min(velocity[3]))
    axes[0,0].imshow(edge_to_image(velocity[0]))
    axes[0,1].imshow(edge_to_image(grid))
    axes[1,0].imshow(edge_to_image(velocity[2]))
    axes[1,1].imshow(edge_to_image(velocity[3]))

    # axes[0,1].imshow()
    # axes[0,1].set_title('')
    # axes[1,0].imshow()
    # axes[1,0].set_title('')
    # axes[1,1].imshow()
    # axes[1,1].set_title('')
    '''
    
    plt.show()
    
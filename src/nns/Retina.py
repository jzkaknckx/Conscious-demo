
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math
from typing import List, Optional, Tuple, Union, Any, Dict
from collections import deque

class PreprocessLayer(nn.Module):
    '''
        预处理
            切割为cropped_size*cropped_size,以(center_x, center_y)为中心,其余补零
            
            只能输入B = 1
            未限制(center_x, center_y)输入大小
    '''
    def __init__(self, cropped_size=1024):
        super().__init__()
        self.cropped_size = cropped_size
        self.resize = T.Resize(
            cropped_size,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True
        )

    def forward(self, x, center_x, center_y):
        
        B, C, H, W = x.shape
        device = x.device
        
        if(center_x > W or center_y > H):
            print("Invalid Center")
            
        if(center_x == -1 ): center_x = W//2
        if(center_y == -1 ): center_y = H//2

        # 计算动态裁切窗口
        # crop_h = min(self.cropped_size, H)
        # crop_w = min(self.cropped_size, W)
        crop_h = crop_w = self.cropped_size
        
        # 计算裁切区域坐标
        # start_x = torch.clamp(torch.tensor(center_x - crop_w//2), 0, W - crop_w)
        # start_y = torch.clamp(torch.tensor(center_y - crop_h//2), 0, H - crop_h)
        start_x = center_x - crop_w//2
        start_y = center_y - crop_h//2

        if(center_x < crop_w//2 or W - center_x < crop_w//2 or center_y < crop_h//2 or H - center_y < crop_h//2 or H < crop_h or W < crop_w):
            x = F.pad(x, (crop_w//2, crop_w//2, crop_h//2, crop_h//2))
            start_x += crop_w//2
            start_y += crop_h//2
        
        # 执行裁切
        cropped = []
        
        crop = x[0, :, 
                    int(start_y):int(start_y+crop_h),
                    int(start_x):int(start_x+crop_w)]
        
        # 等比缩放并保持原始比例
        scale = min(self.cropped_size/crop_h, self.cropped_size/crop_w)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        
        resized = T.functional.resize(
            crop, 
            size=[new_h, new_w],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True
        )
        
        # 边缘填充
        pad_h = (self.cropped_size - new_h)
        pad_w = (self.cropped_size - new_w)
        padded = T.functional.pad(
            resized,
            padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2],
            padding_mode='edge'
        )
        cropped.append(padded)
        
        return torch.cat(cropped, dim=0).unsqueeze(0)

class ProjectionLayer(nn.Module):
    '''
        前向投射
            要求输入正方形
            
            未缓存
    
    '''
    def __init__(self, output_size=512, center_size=256, tau=0.01):
        super().__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.tau = tau

    def forward(self, x):
        B, C, H_in, W_in = x.shape
        device = x.device
        assert H_in == W_in, "Input must be square"
        cropped_size = H_in

        # if self.tau == -1:
            
        
        # 生成输出网格坐标
        i, j = torch.meshgrid(torch.arange(self.output_size, device=device, dtype=torch.float),
                              torch.arange(self.output_size, device=device, dtype=torch.float),
                              indexing='ij')  # (output_size, output_size)
        center_out = (self.output_size - 1) / 2.0
        dx = i - center_out
        dy = j - center_out

        # 计算极坐标
        r_out = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)

        # 计算输入半径r_in
        mask = r_out > self.center_size / 2
        delta_r = torch.where(mask, r_out - self.center_size / 2 , torch.zeros_like(r_out))
        # r_in = torch.where(mask, self.tau * (delta_r) ** 3 / 6 + r_out, r_out)
        r_in = torch.where(mask, self.center_size / 2 + (torch.exp(self.tau * (delta_r)) -1) / self.tau, r_out)

        # 转换为输入坐标
        center_in = (cropped_size - 1) / 2.0
        x_in = center_in + r_in * torch.cos(theta)
        y_in = center_in + r_in * torch.sin(theta)

        # 归一化到[-1, 1]
        x_norm = (x_in / (cropped_size - 1)) * 2 - 1
        y_norm = (y_in / (cropped_size - 1)) * 2 - 1

        # 生成采样网格
        grid = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).clamp(-1, 1)  # (B, output_size, output_size, 2)

        # 应用网格采样
        projected = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
        return projected

class EdgeDetectionLayer(nn.Module):
    def __init__(self):
        super(EdgeDetectionLayer, self).__init__()
        # Sobel 算子：水平和竖直边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # 假设 x 的形状为 (B, C, H, W)
        # 对每个通道分别计算梯度
        
        '''todo 改成F.conv2d'''
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=1)
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad

class RGB2H(nn.Module):
    def __init__(self):
        super(RGB2H, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB image."
        r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)
        delta = max_val - min_val
        
        h_channel = torch.zeros_like(max_val)
        mask_rgb_equal = (delta < 1e-8)
        h_channel[mask_rgb_equal] = 0.0
        
        # 处理max为R的情况
        mask_max_r = (max_val == r) & (~mask_rgb_equal)
        h_channel[mask_max_r] = ((g[mask_max_r] - b[mask_max_r]) / delta[mask_max_r]) % 6
        
        # 处理max为G的情况
        mask_max_g = (max_val == g) & (~mask_rgb_equal)
        h_channel[mask_max_g] = ((b[mask_max_g] - r[mask_max_g]) / delta[mask_max_g]) + 2
        
        # 处理max为B的情况
        mask_max_b = (max_val == b) & (~mask_rgb_equal)
        h_channel[mask_max_b] = ((r[mask_max_b] - g[mask_max_b]) / delta[mask_max_b]) + 4
        
        h_channel = h_channel / 6.0
        h_channel = torch.clamp(h_channel, 0.0, 1.0)
        
        return h_channel.unsqueeze(1)

class FrameDifferenceLayer(nn.Module):
    '''
    差帧
    '''
    def __init__(self):
        super(FrameDifferenceLayer, self).__init__()
        self.x_old = None
        self.first_frame = True
    
    def forward(self, x):
        if self.first_frame:
            self.x_old = torch.zeros_like(x) 
            self.first_frame = False
        
        diff = torch.abs(x - self.x_old)
        self.x_old = x
        return diff

'''
todo 添加max_in_field冗余
'''

class OpticalFlowLayer(nn.Module):
    def __init__(self, lateralField=2, flowLayerCache=5, tau=0.5):
        """
        初始化 LateralInfulentialOpticalFlowLayer 模块。

            参数:
                LateralField(int): LateralField 的半径，光流节点的感受野，默认值为 2
                flowLayerCache(int): 缓存帧数，默认值为 5
                tau(float): 信号衰减
        """
        super(OpticalFlowLayer, self).__init__()
        self.lateralField = lateralField
        self.flowLayerCache = flowLayerCache
        self.tau = tau
        self.threshold = None
        
        assert flowLayerCache > 2 * lateralField, "flowLayerCache and lateralField has to satisfy: flowLayerCache >= 2 * lateralField + 1"
        
        self.cache = []
        self.cacheforMaxpooling = []
        self.offsets = [dx for dx in range(-lateralField, lateralField+1)] # frame = -lF -> 0 -> lF  ,2*lF+1 in total
        
        self.valid = None
        self.j = None
        self.i = None
    
    def gird_precompute(self, H, W):
        '''
        计算偏移网格
        '''

        valid = torch.tensor([False] * H * W * (2 * self.lateralField + 1) * 4, dtype=torch.bool).reshape(H, W, 2 * self.lateralField + 1, 4)
        j ,i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        for dx in range(-self.lateralField, self.lateralField+1):
            grid_in_x = i + dx # x
            grid_in_y = j + dx # y
            grid_out_x = i - dx # x
            grid_out_y = j - dx # y     
            
            valid[...,dx + self.lateralField, 0] = (grid_in_x >= 0) & (grid_in_x < W)
            valid[...,dx + self.lateralField, 1] = (grid_in_y >= 0) & (grid_in_y < H)
            valid[...,dx + self.lateralField, 2] = (grid_out_x >= 0) & (grid_out_x < W)
            valid[...,dx + self.lateralField, 3] = (grid_out_y >= 0) & (grid_out_y < H)
            
        return valid, j, i
    
    def forward(self, x, max_in_field):
        """
        计算光流的xy分量。

            Warning:
                仅支持单批次输入
            
            输入:
                支持:
                    grad_x (torch.Tensor): (B, 1, H, W) + grad_y (torch.Tensor): (B, 1, H, W)
                    grad (torch.Tensor): (B, 1, H, W)
                    RGB (torch.Tensor): (B, 3, H, W)
                            B == 1
            输出:
                sum_influence_x, sum_influence_y
        """
        
        # 输入 -> (B, 1, H, W)
        dim = len(x)
        if dim == 2:
            grad_x, grad_y = x
            B, C, H, W = grad_x.shape
            x = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        elif dim == 1:
            B, C, H, W = x.shape
            # if C == 3:
            #     togray = T.Grayscale()
            #     x = togray(x)
        assert B == 1, "Only One Batch Once"
        
        if self.valid is None:
            self.valid, self.j, self.i = self.gird_precompute(H, W)
        
        # influence by this frame
        influence = torch.zeros_like(x).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, self.lateralField*2+1, 2)
        # influence = torch.zeros(1, 1, H, W, self.lateralField*2+1, 2)
        for dx_idx, dx in enumerate(self.offsets):
            
            mask_x = self.valid[..., dx_idx, 0]
            mask_y = self.valid[..., dx_idx, 1]
            
            i_shifted = self.i + dx
            j_shifted = self.j + dx
            
            for c in range(C):
                strength_x = torch.zeros_like(x[0, 0])
                strength_y = torch.zeros_like(x[0, 0])
                
                strength_x[mask_x] = x[0, c, self.j[mask_x], i_shifted[mask_x]]
                strength_y[mask_y] = x[0, c, j_shifted[mask_y], self.i[mask_y]]
                
                influence[0, c, :, :, dx_idx, 0] = strength_x.unsqueeze(0).unsqueeze(0)  
                influence[0, c, :, :, dx_idx, 1] = strength_y.unsqueeze(0).unsqueeze(0)
        
        
        # stack of influence
        # (1, C, H, W, self.lateralField * 2 + 1, 2) * self.flowLayerCache  --stack-->  (1, C, H, W, self.lateralField * 2 + 1, 2, self.flowLayerCache) 
        #  B  C  H  W  - lateralField -> dx -> lateralField    horizontal/vertical      1 -> frame -> flowLayerCache
        # (1, C, H, W,        self.lateralField * 2 + 1,               2,                   self.flowLayerCache) 
        self.cache.append(influence)
        self.cacheforMaxpooling.append(max_in_field)
        
        sum_influence_x = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        sum_influence_y = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        # sum_influence_x = torch.zeros(1, 1, H, W, 2)
        # sum_influence_y = torch.zeros(1, 1, H, W, 2)
        max_in_fieldandtime = torch.zeros_like(x)
        flow = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 4)

        frames = len(self.cache)
        assert frames == len(self.cacheforMaxpooling), "Max pooling layer has to be engaged in the network at the same time as the flow layer"
        if frames > self.flowLayerCache:
            self.cache.pop(0)
            self.cacheforMaxpooling.pop(0)
            imCache = torch.stack(self.cache, dim = -1)
            poolingCache = torch.stack(self.cacheforMaxpooling, dim = -1)
            
            for sample in range(0, self.lateralField * 2 + 1):
                sum_influence_x[0, :, :, :, 0] += imCache[0, :, :, :, sample, 0, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_x[0, :, :, :, 1] += imCache[0, :, :, :, sample, 0, self.flowLayerCache - sample - 1]
                sum_influence_y[0, :, :, :, 0] += imCache[0, :, :, :, sample, 1, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_y[0, :, :, :, 1] += imCache[0, :, :, :, sample, 1, self.flowLayerCache - sample - 1]
            
            max_result = torch.max(poolingCache, dim=4)
            max_in_fieldandtime = max_result.values 
            flow[..., 0] = torch.relu(sum_influence_x[0, :, :, :, 0] - max_in_fieldandtime)
            flow[..., 1] = torch.relu(sum_influence_x[0, :, :, :, 1] - max_in_fieldandtime)
            flow[..., 2] = torch.relu(sum_influence_y[0, :, :, :, 0] - max_in_fieldandtime)
            flow[..., 3] = torch.relu(sum_influence_y[0, :, :, :, 1] - max_in_fieldandtime)
            
        else: print("not Enough Frames")
                
        factor = torch.exp(-self.tau * torch.ones_like(influence))
        self.cache = [tensor * factor for tensor in self.cache]

        return flow

def smooth_moving(input1, input2, velocity):
    center_x0, center_y0 = input1
    center_x1, center_y1 = input2
    dx = center_x1 - center_x0
    dy = center_y1 - center_y0
    distance = np.sqrt(dx**2 + dy**2)
    steps = int(distance // velocity)
    x = np.linspace(center_x0, center_x1, steps+1)
    y = np.linspace(center_y0, center_y1, steps+1)
    return x, y

class RetinaModel(nn.Module):
    def __init__(self, cropped_size=1024, output_size=512, center_size=256, projectiontau=0.01, lateralField=2, flowLayerCache=5, flowLayertau=0.5):
        '''
        较好的projectiontau
            cropped_size = 1024
            output_size = 512
            center_size = 256
            tau = 0.01
        '''
        super().__init__()
        self.preprocess = PreprocessLayer(cropped_size)
        self.projection = ProjectionLayer(output_size, center_size, projectiontau)
        self.edge_detection = EdgeDetectionLayer()
        self.rgb2h = RGB2H()
        self.frame_diff = FrameDifferenceLayer()
        self.maxpooling_for_flowlayer = nn.MaxPool2d(kernel_size=lateralField * 2 + 1, stride=1, padding=lateralField)
        self.flow = OpticalFlowLayer(lateralField, flowLayerCache, flowLayertau)
        
    def forward(self, x, center_x, center_y):
        x = self.preprocess(x, center_x, center_y)          # [1, C, H0, W0]
        x = self.projection(x)                              # [1, C, H, W]
        grad = self.edge_detection(x)                       # [1, 1, H, W, 2]
        h = self.rgb2h(x)                                   # [1, 1, H, W]
        diff = self.frame_diff(x)                           # [1, C, H, W]
        max_in_field = self.maxpooling_for_flowlayer(diff)  #
        flowvelocity = self.flow(diff, max_in_field)        # [1, C, H, W, 4]
        
        return x, grad, h, diff, flowvelocity
    
            
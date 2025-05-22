'''
20250506
    PreprocessLayer和ProjectionLayer初步完成 裁剪+环状映射
    会导致边缘扭曲
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from collections import deque

from trail.pic_tools import *
from trail.matrix_tools import *

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
        return grad_x, grad_y

class RetinaModel(nn.Module):
    def __init__(self, cropped_size=1024, output_size=512, center_size=256, tau=0.01):
        '''
        较好的projectiontau
            cropped_size = 1024
            output_size = 512
            center_size = 256
            tau = 0.01
        '''
        super().__init__()
        self.preprocess = PreprocessLayer(cropped_size)
        self.projection = ProjectionLayer(output_size, center_size, tau)
        self.edge_detection = EdgeDetectionLayer()
        # self.optical_flow = OpticalflowLayer(scale_factor=scale_factor)
        
    def forward(self, x, center_x=-1, center_y=-1):
        x = self.preprocess(x, center_x, center_y)
        x = self.projection(x)
        grad_x, grad_y = self.edge_detection(x)
        
        return x, grad_x, grad_y
    
'''
# 验证测试
if __name__ == "__main__":
    
    cropped_size = 1024
    output_size = 512
    center_size = 256
    tau = 0.01

    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    tensor, oringinal_image = image_to_tensor(image_path)

    # 初始化模型
    model = RetinaModel(cropped_size, output_size, center_size, tau)

    center_x = torch.tensor([128.0])
    center_y = torch.tensor([128.0])

     # 前向传播
    output = model(tensor, -1, -1)
    
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes[0,0].imshow(oringinal_image)
    axes[0,0].set_title('Original')
    axes[0,1].imshow(tensor_to_image(output[0]))
    axes[0,1].set_title('')
    axes[1,0].imshow(edge_to_image(output[1]))
    axes[1,0].set_title('')
    axes[1,1].imshow(edge_to_image(output[2]))
    axes[1,1].set_title('')
    
    plt.show()
    
'''  
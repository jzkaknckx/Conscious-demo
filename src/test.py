import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math
from typing import List, Optional, Tuple, Union, Any, Dict

from trail.pic_tools import *
from trail.matrix_tools import *

from nns.Retina import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
    ,RGB2H
    ,FrameDifferenceLayer
    # ,OpticalFlowLayer
    # ,RetinaModel
)

class OpticalFlowLayerO(nn.Module):
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
    
    def forward(self, x):
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
        
        sum_influence_x = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        sum_influence_y = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        # sum_influence_x = torch.zeros(1, 1, H, W, 2)
        # sum_influence_y = torch.zeros(1, 1, H, W, 2)
        flow = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 4)

        frames = len(self.cache)
        if frames > self.flowLayerCache:
            self.cache.pop(0)
            imCache = torch.stack(self.cache, dim = -1)
            
            for sample in range(0, self.lateralField * 2 + 1):
                sum_influence_x[0, :, :, :, 0] += imCache[0, :, :, :, sample, 0, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_x[0, :, :, :, 1] += imCache[0, :, :, :, sample, 0, self.flowLayerCache - sample - 1]
                sum_influence_y[0, :, :, :, 0] += imCache[0, :, :, :, sample, 1, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_y[0, :, :, :, 1] += imCache[0, :, :, :, sample, 1, self.flowLayerCache - sample - 1]
            
        else: print("not Enough Frames")
                
        factor = torch.exp(-self.tau * torch.ones_like(influence))
        self.cache = [tensor * factor for tensor in self.cache]

        return flow

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
        x = self.preprocess(x, center_x, center_y)
        x = self.projection(x)
        grad = self.edge_detection(x)
        h = self.rgb2h(x)
        diff = self.frame_diff(x)
        max_in_field = self.maxpooling_for_flowlayer(diff)
        flowvelocity = self.flow(diff, max_in_field)
        
        return x, flowvelocity, diff

if __name__ == "__main__":

    
    fig, axes = plt.subplots(3,3, figsize=(10, 5))
    
    # 准备输入
    image_path = 'src/picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # [1, 3, 1280, 1920]
    tensor, oringinal_image = image_to_tensor(image_path)
    # axes[0,0].imshow(oringinal_image)
    # axes[0,0].set_title('')
    
    # tensor = generate_graph_tensor(H=40, W=40, graph="Circle", R=10)
    
    r = RetinaModel()
    
    axes[0,0].imshow(tensor_to_image(tensor))
    
    input_center0 = (920, 920)
    input_center1 = (950, 950)
    
    centers_x, centers_y = smooth_moving(input_center0, input_center1, 1)
   
    window = 20
    windowx = 362
    windowy = 279
     
    for x, y in zip(centers_x, centers_y):
        print(x, y)
        out, velocity, diff = r(tensor, x, y)
    
    # stream_draw(torch.stack([vx, vy], dim=-1))
    # edge_to_image()
    
    axes[0,1].imshow(tensor_to_image(out))
    axes[0,2].imshow(tensor_to_image(diff[:, :, windowy-window : windowy+window, windowx-window : windowx+window]))
    axes[1,0].imshow(tensor_to_image(out[:, :, windowy-window : windowy+window, windowx-window : windowx+window]))
    axes[1,1].imshow(tensor_to_image(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 1].unsqueeze(0)))
    axes[1,2].imshow(edge_to_image(velocity[:, 0, :, :, 0].unsqueeze(0)))
    axes[2,0].imshow(edge_to_image(velocity[:, 0, :, :, 1].unsqueeze(0)))
    axes[2,1].imshow(edge_to_image(velocity[:, 0, :, :, 2].unsqueeze(0)))
    axes[2,2].imshow(edge_to_image(velocity[:, 0, :, :, 3].unsqueeze(0)))
    
    index = torch.argmax(velocity[:, 0, ..., 0])
    print(index) #279 362
    print(velocity[0,0,windowy,windowx,0])
    print(velocity[0,0,windowy,windowx,1])
    print(velocity[0,0, windowy-window : windowy+window, windowx-window : windowx+window,0])
    print(velocity[0,0, windowy-window : windowy+window, windowx-window : windowx+window,1])

    # maxv, index = torch.max(velocity[..., 1])
    # print(index.item())
    # maxv, index = torch.max(velocity[..., 2])
    # print(index.item())
    # maxv, index = torch.max(velocity[..., 3])
    # print(index.item())
    plt.show()


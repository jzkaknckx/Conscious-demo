
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from collections import deque

from trail.pic_tools import *
from trail.matrix_tools import *

torch.set_printoptions(profile="full",linewidth=512)

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)

class FrameDifferenceLayer(nn.Module):
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
    
class InfluenceSumLayer(nn.Module):
    def __init__(self, lateralField=2, flowLayerCache=5, tau=1.0):
        """
        初始化 LateralInfulentialOpticalFlowLayer 模块。

            参数:
                LateralField(int): LateralField 的半径，光流节点的感受野，默认值为 2
                flowLayerCache(int): 缓存帧数，默认值为 5
                tau(float): 信号衰减
        """
        super(InfluenceSumLayer, self).__init__()
        self.lateralField = lateralField
        self.flowLayerCache = flowLayerCache
        self.tau = tau
        
        assert flowLayerCache > 2 * lateralField, "flowLayerCache and lateralField has to satisfy: flowLayerCache >= 2 * lateralField + 1"
        
        self.cache = []
        self.offsets = [dx for dx in range(-lateralField, lateralField+1)] # frame = -lF -> 0 -> lF  ,2*lF+1 in total
        
        self.valid = None
        self.j = None
        self.i = None
    
    def gird_precompute(self, H, W):
    
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
            if C == 3:
                togray = T.Grayscale()
                x = togray(x)
        
        assert B == 1, "Only One Batch Once"
        
        if self.valid is None:
            self.valid, self.j, self.i = self.gird_precompute(H, W)
            
        # influence by this frame
        influence = torch.zeros(1, 1, H, W, self.lateralField*2+1, 2)
        for dx_idx, dx in enumerate(self.offsets):
            # 获取有效区域掩码
            mask_x = self.valid[..., dx_idx, 0]  # x方向偏移dx的有效区域 (H,W)
            mask_y = self.valid[..., dx_idx, 1]  # y方向偏移dx的有效区域 (H,W)
            
            # 计算平移后的坐标
            i_shifted = self.i + dx  # x方向平移后的x坐标
            j_shifted = self.j + dx  # y方向平移后的y坐标
            
            # 初始化强度矩阵
            strength_x = torch.zeros_like(x[0, 0])  # (H,W)
            strength_y = torch.zeros_like(x[0, 0])
            
            # 应用平移：将x向右平移dx，y向下平移dx
            strength_x[mask_x] = x[0, 0, self.j[mask_x], i_shifted[mask_x]]
            strength_y[mask_y] = x[0, 0, j_shifted[mask_y], self.i[mask_y]]
            
            # 将结果存入influence张量
            influence[..., dx_idx, 0] = strength_x.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            influence[..., dx_idx, 1] = strength_y.unsqueeze(0).unsqueeze(0)
        
        # stack of influence
        # (1, 1, H, W, self.lateralField * 2 + 1, 2) * self.flowLayerCache  --stack-->  (1, 1, H, W, self.lateralField * 2 + 1, 2, self.flowLayerCache) 
        #  B  C  H  W  - lateralField -> dx -> lateralField    horizontal/vertical      1 -> frame -> flowLayerCache
        # (1, 1, H, W,        self.lateralField * 2 + 1,               2,                   self.flowLayerCache) 
        self.cache.append(influence)
        
        sum_influence_x = sum_influence_y = torch.zeros(1, 1, H, W, 2)
        if len(self.cache) > self.flowLayerCache:
            self.cache.pop(0)
            imCache = torch.stack(self.cache, dim = -1)
            for sample in range(0, self.lateralField * 2 + 1):
                sum_influence_x[0, 0, :, :, 0] += imCache[0, 0, :, :, sample, 0, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_x[0, 0, :, :, 1] += imCache[0, 0, :, :, sample, 0, self.flowLayerCache - sample - 1]
                sum_influence_y[0, 0, :, :, 0] += imCache[0, 0, :, :, sample, 1, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_y[0, 0, :, :, 1] += imCache[0, 0, :, :, sample, 1, self.flowLayerCache - sample - 1]
            
        else: print("not Enough Frames")
                
        factor = torch.exp(-self.tau * torch.ones_like(influence))
        self.cache = [tensor * factor for tensor in self.cache]
                
        return sum_influence_x[..., 0], sum_influence_x[..., 1], sum_influence_y[..., 0], sum_influence_y[..., 1]
    
class InfluenceSumLayerN(nn.Module):
    def __init__(self, lateralField=2, flowLayerCache=5, tau=1.0):
        super(InfluenceSumLayerN, self).__init__()
        self.lateralField = lateralField
        self.flowLayerCache = flowLayerCache
        self.tau = tau
        
        assert flowLayerCache > 2 * lateralField, "flowLayerCache must satisfy: flowLayerCache > 2 * lateralField"
        
        self.cache = []
        # 分离水平和垂直方向的偏移
        self.offsets_x = [dx for dx in range(-lateralField, lateralField+1)]
        self.offsets_y = [dy for dy in range(-lateralField, lateralField+1)]
        
        self.valid = None
        self.j = None
        self.i = None
    
    def grid_precompute(self, H, W):
        # 修改为存储水平和垂直方向的独立偏移
        valid = torch.zeros(H, W, len(self.offsets_x), len(self.offsets_y), 4, dtype=torch.bool)
        j, i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        for dx_idx, dx in enumerate(self.offsets_x):
            for dy_idx, dy in enumerate(self.offsets_y):
                grid_in_x = i + dx
                grid_in_y = j + dy
                grid_out_x = i - dx
                grid_out_y = j - dy
                
                valid[..., dx_idx, dy_idx, 0] = (grid_in_x >= 0) & (grid_in_x < W)
                valid[..., dx_idx, dy_idx, 1] = (grid_in_y >= 0) & (grid_in_y < H)
                valid[..., dx_idx, dy_idx, 2] = (grid_out_x >= 0) & (grid_out_x < W)
                valid[..., dx_idx, dy_idx, 3] = (grid_out_y >= 0) & (grid_out_y < H)
                
        return valid, j, i
    
    def forward(self, x):
        # 保留原始梯度方向信息
        if isinstance(x, (list, tuple)) and len(x) == 2:
            grad_x, grad_y = x
            B, C, H, W = grad_x.shape
            # 关键修改：不再计算梯度幅值，直接使用原始分量
            x0 = grad_x  # 水平梯度分量
            x1 = grad_y  # 垂直梯度分量
        else:
            B, C, H, W = x.shape
            if C == 3:
                togray = T.Grayscale()
                x_gray = togray(x)
                # 灰度图复制到两个通道
                x0 = x_gray
                x1 = x_gray
            else:
                x0 = x
                x1 = x
        
        assert B == 1, "Only batch size 1 is supported"
        
        if self.valid is None:
            self.valid, self.j, self.i = self.grid_precompute(H, W)
            
        # 修改influence张量维度：包含水平和垂直偏移
        influence = torch.zeros(1, 1, H, W, len(self.offsets_x), len(self.offsets_y), 2)
        
        # 分别处理水平和垂直偏移
        for dx_idx, dx in enumerate(self.offsets_x):
            for dy_idx, dy in enumerate(self.offsets_y):
                # 水平方向处理
                mask_x = self.valid[..., dx_idx, dy_idx, 0]
                i_shifted = self.i + dx
                strength_x = torch.zeros_like(x0[0, 0])
                strength_x[mask_x] = x0[0, 0, self.j[mask_x], i_shifted[mask_x]]
                
                # 垂直方向处理
                mask_y = self.valid[..., dx_idx, dy_idx, 1]
                j_shifted = self.j + dy
                strength_y = torch.zeros_like(x1[0, 0])
                strength_y[mask_y] = x1[0, 0, j_shifted[mask_y], self.i[mask_y]]
                
                influence[0, 0, ..., dx_idx, dy_idx, 0] = strength_x
                influence[0, 0, ..., dx_idx, dy_idx, 1] = strength_y
        
        self.cache.append(influence)
        
        sum_influence_x = torch.zeros(1, 1, H, W, 2)
        sum_influence_y = torch.zeros(1, 1, H, W, 2)
        
        if len(self.cache) >= self.flowLayerCache:
            if len(self.cache) > self.flowLayerCache:
                self.cache.pop(0)
            
            imCache = torch.stack(self.cache, dim=-1)  # 新增时间维度
            
            # 水平方向累加逻辑
            for dx_idx in range(len(self.offsets_x)):
                for dy_idx in range(len(self.offsets_y)):
                    # 水平分量(X方向)的时间延迟累加
                    time_idx_x = self.flowLayerCache - (abs(dx_idx - self.lateralField) + abs(dy_idx))
                    if time_idx_x < 0: 
                        continue
                    sum_influence_x[..., 0] += imCache[..., dx_idx, dy_idx, 0, time_idx_x]  # 正向
                    sum_influence_x[..., 1] += imCache[..., dx_idx, dy_idx, 0, self.flowLayerCache - time_idx_x - 1]  # 反向
                    
                    # 垂直分量(Y方向)的时间延迟累加
                    time_idx_y = self.flowLayerCache - (abs(dx_idx) + abs(dy_idx - self.lateralField))
                    if time_idx_y < 0: 
                        continue
                    
                    print(imCache.shape)
                    print(dx_idx, dy_idx, time_idx_y)
                    
                    sum_influence_y[..., 0] += imCache[..., dx_idx, dy_idx, 1, time_idx_y]  # 正向
                    sum_influence_y[..., 1] += imCache[..., dx_idx, dy_idx, 1, self.flowLayerCache - time_idx_y - 1]  # 反向
            
            slicer = TensorSlicer(imCache[:, :, 256, :, :, 0, :])
            slicer.set_dimension_names(["batch", "channel", "x", "feature", "time"])
            slicer.display_slice_grid(
                coord_dims=(2, 4),       # 选择 height 和 width 作为坐标
                content_dims=[0, 1, 3],  # 选择 batch, channel, feature 作为内容
                coord_ranges=[slice(256-20, 256+20), slice(0, 5)], 
                max_elements_per_cell=40,  # 每个单元格最多显示4个元素
                max_cell_width=80,        # 单元格最大宽度
                precision=2               # 浮点数精度为2位
            )
            slicer = TensorSlicer(imCache[:, :, :, 256, :, 0, :])
            slicer.set_dimension_names(["batch", "channel", "y", "feature", "time"])
            slicer.display_slice_grid(
                coord_dims=(2, 4),       # 选择 height 和 width 作为坐标
                content_dims=[0, 1, 3],  # 选择 batch, channel, feature 作为内容
                coord_ranges=[slice(256-20, 256+20), slice(0, 5)], 
                max_elements_per_cell=40,  # 每个单元格最多显示4个元素
                max_cell_width=80,        # 单元格最大宽度
                precision=2               # 浮点数精度为2位
            )
                    
        else:
            print("Not enough frames in cache")
        
        # 应用衰减因子
        factor = torch.exp(-self.tau * torch.ones_like(influence))
        self.cache = [tensor * factor for tensor in self.cache]
        
        
                
        return (
            sum_influence_x[..., 0],  # X正向
            sum_influence_x[..., 1],  # X反向
            sum_influence_y[..., 0],  # Y正向
            sum_influence_y[..., 1]   # Y反向
        )

def smooth_moving(input1, input2, velocity):
    center_x0, center_y0 = input1
    center_x1, center_y1 = input2
    dx = center_x1 - center_x0
    dy = center_y1 - center_y0
    distance = np.sqrt(dx**2 + dy**2)
    steps = int(distance // velocity)
    x = np.linspace(center_x0, center_x1, steps)
    y = np.linspace(center_y0, center_y1, steps)
    return x, y

class RetinaModel(nn.Module):
    def __init__(self, cropped_size=1024, output_size=512, center_size=256, projectiontau=0.01, lateralField=2, flowLayerCache=5, flowLayertau=1.0):
        super().__init__()
        self.preprocess = PreprocessLayer(cropped_size)
        self.projection = ProjectionLayer(output_size, center_size, projectiontau)
        self.edge_detection = EdgeDetectionLayer()
        self.frameDiff = FrameDifferenceLayer()
        # self.flow = LateralInfulentialOpticalFlowLayer(lateralField, flowLayerCache, flowLayertau)
        self.flow = InfluenceSumLayerN(lateralField, flowLayerCache, flowLayertau)
        
    def forward(self, x, center_x, center_y):
        x = self.preprocess(x, center_x, center_y)
        x = self.projection(x)
        grad = self.edge_detection(x)
        diff = self.frameDiff(x)
        # flowvelocity = self.flow(x)
        flowvelocity = self.flow(diff)
        
        return x, flowvelocity, diff        

if __name__ == "__main__":

    
    fig, axes = plt.subplots(2,2, figsize=(10, 5))
    
    # 准备输入
    image_path = 'src/picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # [1, 3, 1280, 1920]
    tensor, oringinal_image = image_to_tensor(image_path)
    # axes[0,0].imshow(oringinal_image)
    # axes[0,0].set_title('')
    
    tensor = generate_graph_tensor(H=1280, W=1920, graph="Triangle", R=500)
    
    r = RetinaModel()
    
    axes[0,0].imshow(tensor_to_image(tensor))
    
    input_center0 = (710, 533)
    input_center1 = (720, 533)
    
    centers_x, centers_y = smooth_moving(input_center0, input_center1, 1)
   
    window = 20
     
    for x, y in zip(centers_x, centers_y):
        print(x, y)
        out, velocity, diff = r(tensor, x, y)
    
    axes[0,1].imshow(tensor_to_image(diff[:, :, 256-window : 256+window, 256-window : 256+window]))
    axes[1,0].imshow(tensor_to_image(out[:, :, 256-window : 256+window, 256-window : 256+window]))
    axes[1,1].imshow(tensor_to_image(out))
    plt.show()

'''
我希望在这个神经网络层中实现类似光流的效果,即sum_influence_x和sum_influence_y变量分别多每个位置上横向和纵向移动的物体敏感
有一个神经元接受self.lateralField * 2 + 1个神将元的输入,前一层的这些神经元呈水平排列,它们对后一层的激活时间不同,随着水平排布的距离增加而延迟时间激活.如果有一个边缘沿着水平方向移动,那么后一层的神经元接受到的刺激是多个较高的值相加,这个边缘对后一层的影响是最大的,而反向移动或垂直移动这个神经元输出则不够
infuence接收前一层差帧的结果,第-2个维度保存了这个(x,y)位置上- lateralField -> dx -> lateralField范围内前一层的影响,第-1个维度保存了横向和纵向的标记
在接受多次输入直到满足flowLayerCache次后,imCache变量的第-1个维度保存了时间信息,将imCache的时间沿着第-2个维度依次延迟相加并保存在sum_influence_x和sum_influence_y中,即可模拟上述神将元的行为
但这个代码有问题,对于输入一个水平向右移动的边缘,sum_influence_x[..., 0]和sum_influence_y[..., 0]完全相同,不知道是代码中有错误的部分还是数学关系如此,请你帮我修改并指出具体原因'''
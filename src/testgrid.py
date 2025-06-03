import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math

from trail.pic_tools import *
from trail.matrix_tools import *

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)
from typing import List, Optional, Tuple, Union, Any, Dict

lateralField=2
offsets = [dx for dx in range(-lateralField, lateralField+1)]

def gird_precompute( H, W):
    
    valid = torch.tensor([False] * H * W * (2 *  lateralField + 1) * 4, dtype=torch.bool).reshape(H, W, 2 *  lateralField + 1, 4)
    j ,i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    for dx in range(- lateralField,  lateralField+1):
        grid_in_x = i + dx # x
        grid_in_y = j + dx # y
        grid_out_x = i - dx # x
        grid_out_y = j - dx # y     
        
        valid[...,dx +  lateralField, 0] = (grid_in_x >= 0) & (grid_in_x < W)
        valid[...,dx +  lateralField, 1] = (grid_in_y >= 0) & (grid_in_y < H)
        valid[...,dx +  lateralField, 2] = (grid_out_x >= 0) & (grid_out_x < W)
        valid[...,dx +  lateralField, 3] = (grid_out_y >= 0) & (grid_out_y < H)
        
    return valid, j, i

H=W=20
valid, j, i = gird_precompute(H, W)
x = torch.ones(1, 1, H, W)

for a in range(H):
    for b in range(W):
        x[0, 0, a, b] = a+b*0.1
print(x)

influence = torch.zeros(1, 1, H, W,  lateralField*2+1, 2)
for dx_idx, dx in enumerate( offsets):
    print(dx_idx, dx)
    
    # 获取有效区域掩码
    mask_x =  valid[..., dx_idx, 0]  # x方向偏移dx的有效区域 (H,W)
    mask_y =  valid[..., dx_idx, 1]  # y方向偏移dx的有效区域 (H,W)
    
    # 计算平移后的坐标
    i_shifted =  i + dx  # x方向平移后的x坐标
    j_shifted =  j + dx  # y方向平移后的y坐标
    
    # 初始化强度矩阵
    strength_x = torch.zeros_like(x[0, 0])  # (H,W)
    strength_y = torch.zeros_like(x[0, 0])
    
    # 应用平移：将x向右平移dx，y向下平移dx
    strength_x[mask_x] = x[0, 0,  j[mask_x], i_shifted[mask_x]]
    strength_y[mask_y] = x[0, 0, j_shifted[mask_y],  i[mask_y]]
    
    # 将结果存入influence张量
    influence[..., dx_idx, 0] = strength_x.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    influence[..., dx_idx, 1] = strength_y.unsqueeze(0).unsqueeze(0)
    
    print(influence[...,dx_idx,1])
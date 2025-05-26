import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math

from trail.pic_tools import *
from trail.matrix_tools import *

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)

H = 10
W = 10
lateralField = 2
offsets = [dx for dx in range(-lateralField, lateralField+1)]

x = torch.ones(1, 1, H, W)
for q in range(H):
    for e in range(W):
        x[0, 0, q, e] = q + e*0.1
# x = torch.rand(1, 1, H, W)
print(x)

def gird_precompute(H, W):
    
    valid = torch.tensor([False] * H * W * 5 * 4, dtype=torch.bool).reshape(H, W, 5, 4)
    for dx in range(-2, 3):
        j ,i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_in_x = i + dx # x
        grid_in_y = j + dx # y
        grid_out_x = i - dx # x
        grid_out_y = j - dx # y     
        
        valid[...,dx + 2, 0] = (grid_in_x >= 0) & (grid_in_x < W)
        valid[...,dx + 2, 1] = (grid_in_y >= 0) & (grid_in_y < H)
        valid[...,dx + 2, 2] = (grid_out_x >= 0) & (grid_out_x < W)
        valid[...,dx + 2, 3] = (grid_out_y >= 0) & (grid_out_y < H)
        # print(valid[...,dx + 2, 1])
        # print(valid[...,dx + 2, 3])
        
    return valid

valid = gird_precompute(H, W)
j ,i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

'''
for dx in range(-2, 3):
    
    # grid_in_x = i + dx # x
    # grid_in_y = j + dx # y
    # grid_out_x = i - dx # x
    # grid_out_y = j - dx # y

    # valid_in_x = (grid_in_x >= 0) & (grid_in_x < W)
    # valid_in_y = (grid_in_y >= 0) & (grid_in_y < H)
    # valid_out_x = (grid_out_x >= 0) & (grid_out_x < W)
    # valid_out_y = (grid_out_y >= 0) & (grid_out_y < H)

    # print(j[valid_in_x], i[valid_in_x])
    
    # 对应x+dx和y+dx的强度
    strength_x = strength_y = torch.zeros(1, 1, H, W)
    strength_x[0, 0, j[valid[...,dx + 2, 2]], i[valid[...,dx + 2, 2]]] = x[0, 0, j[valid[...,dx + 2, 0]], i[valid[...,dx + 2, 0]]]
    strength_y[0, 0, j[valid[...,dx + 2, 3]], i[valid[...,dx + 2, 3]]] = x[0, 0, j[valid[...,dx + 2, 1]], i[valid[...,dx + 2, 1]]]
    print(valid[...,dx + 2, 0])
    print(j[valid[...,dx + 2, 0]], i[valid[...,dx + 2, 0]])
    print(valid[...,dx + 2, 1])
    print(j[valid[...,dx + 2, 1]], i[valid[...,dx + 2, 1]])
    print(valid[...,dx + 2, 2])
    print(j[valid[...,dx + 2, 2]], i[valid[...,dx + 2, 2]])
    print(valid[...,dx + 2, 3])
    print(j[valid[...,dx + 2, 3]], i[valid[...,dx + 2, 3]])
    # print(strength_x)
    # print(strength_y)
'''
for dx_idx, dx in enumerate(offsets):
    ################################ 修改部分 ################################
    # 获取有效区域掩码
    mask_x = valid[..., dx_idx, 0]  # x方向偏移dx的有效区域 (H,W)
    mask_y = valid[..., dx_idx, 1]  # y方向偏移dx的有效区域 (H,W)
    
    # 计算平移后的坐标
    i_shifted = i + dx  # x方向平移后的x坐标
    j_shifted = j + dx  # y方向平移后的y坐标
    
    # 初始化强度矩阵
    strength_x = torch.zeros_like(x[0, 0])  # (H,W)
    strength_y = torch.zeros_like(x[0, 0])
    
    # 应用平移：将x向右平移dx，y向下平移dx
    strength_x[mask_x] = x[0, 0, j[mask_x], i_shifted[mask_x]]
    strength_y[mask_y] = x[0, 0, j_shifted[mask_y], i[mask_y]]
    print(strength_x)
    print(strength_y)
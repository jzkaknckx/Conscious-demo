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

H = 5
W = 5


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

for dx in range(-2, 3):
    j ,i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
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
    print(j[valid[...,dx + 2, 2]], i[valid[...,dx + 2, 2]])
    print(j[valid[...,dx + 2, 1]], i[valid[...,dx + 2, 1]])
    # print(strength_x)
    print(strength_y)
    

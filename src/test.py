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


H = 512
W = 512
dx = dy = -5
i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
k = i + dx
l = j + dy


valid = (k >= 0) & (k < H) & (l >= 0) & (l < W)
valid4 = valid.view(1, 1, H, W)

strength_kl = torch.zeros(1, 1, H, W, 5) 
strength = torch.zeros(1, 1, H, W, 5)
strength_kl[valid4] = strength[0, 0, k[valid], l[valid]]

print(i.shape, j.shape)
print(k.shape, l.shape)
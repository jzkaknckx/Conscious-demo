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
    ,FrameDifferenceLayer
    ,OpticalFlowLayer
    # ,RetinaModel
)

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

fuction = RGB2H()
image_path = 'src/picture/trail/H.jpg'  # [1, 3, 1280, 1920]
tensor, oringinal_image = image_to_tensor(image_path)
image_path2 = 'src/picture/trail/HS.jpg'  # [1, 3, 1280, 1920]
tensor2, oringinal_image2 = image_to_tensor(image_path2)
fig, axes = plt.subplots(2,2, figsize=(10, 5))
h1 = fuction(tensor)
axes[0,0].imshow(tensor_to_image(tensor))
axes[1,0].imshow(tensor_to_image_1channel(h1))
h2 = fuction(tensor2)
axes[0,1].imshow(tensor_to_image(tensor2))
axes[1,1].imshow(tensor_to_image_1channel(h2))
plt.show()
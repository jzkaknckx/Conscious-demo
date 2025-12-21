import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from collections import deque

from trail.pic_tools import *
from trail.matrix_tools import *

torch.set_printoptions(profile="full",linewidth=512)

from nns.Retina import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
    ,RetinaModel
)

from nns.advanced_feature import * 


if __name__ == "__main__":
    fig, axes = plt.subplots(2,2, figsize=(10, 5))
    
    # 准备输入
    image_path = 'src/picture/trail/arcs_pattern.png'  # [1, 3, 1280, 1920]
    tensor, oringinal_image = image_to_tensor(image_path)
    
    # curvFilter = CurvatureFilter()
    # curve = curvFilter(tensor)
    curvFilters = MultiScaleCurvatureBank([1.0, 2.0, 4.0])
    curve = curvFilters(tensor)
    
    axes[0,0].imshow(tensor_to_image(tensor))
    axes[0,1].imshow(tensor_to_image(curve[:, 0, :, :]))
    axes[1,0].imshow(tensor_to_image(curve[:, 1, :, :]))
    axes[1,1].imshow(tensor_to_image(curve[:, 2, :, :]))
    plt.show()
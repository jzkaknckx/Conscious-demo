import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from collections import deque

from trail.pic_tools import *
from trail.matrix_tools import *

from nns.cnns.features import RetinaModel, MultiScaleFeatureBank

if __name__ == "__main__":
    fig, axes = plt.subplots(3,3, figsize=(10, 5))
    
    # 准备输入
    image_path = 'src/picture/OIP-C.jpg'  # [1, 3, 1280, 1920]
    tensor, oringinal_image = image_to_tensor(image_path)
    axes[0,0].imshow(tensor_to_image(tensor))
 
    retina = RetinaModel(edge_apply_gaussian=True, edge_gauss_kernel_size=5, edge_gauss_sigma=1.0)
    
    x_proj, grad, h, _, _, cropped = retina(tensor, center_x = -1, center_y = -1)
    msbank = MultiScaleFeatureBank(scales=[1.0, 2.0, 4.0], base_sigma=1.0)
    curvature_bank, aspect_bank, orient_bank = msbank(cropped, precomputed_derivs={'grad': grad})

    axes[0,1].imshow(tensor_to_image(x_proj))
    axes[0,2].imshow(edge_to_image(grad[..., 0]))
    axes[1,0].imshow(edge_to_image(grad[..., 1]))
    axes[1,1].imshow(edge_to_image(h))
    
    axes[1,2].imshow(tensor_to_image(curvature_bank[:, 0, ...].unsqueeze(1)))
    axes[2,0].imshow(tensor_to_image(aspect_bank[:, 0, ...].unsqueeze(1)))
    axes[2,2].imshow(tensor_to_image(orient_bank[:, 0, ...].unsqueeze(1)))
    
    
    plt.show()
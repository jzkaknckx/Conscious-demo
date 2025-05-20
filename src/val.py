import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from trail.pic_tools import *
from trail.matrix_tools import *

from nn.Retina_d_3rd import (
    RetinaModel
)


    
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
    
    '''trail'''
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
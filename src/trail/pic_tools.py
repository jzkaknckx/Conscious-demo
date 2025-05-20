"""
图片调试工具
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
import torchvision.transforms as transforms
import math

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_printoptions(profile="full",linewidth=512)

'''
图片输入
'''
def load_and_crop_image(image_path, crop_size=(128, 128)):
    """
    加载图像并裁剪到指定尺寸。
    
    参数:
        image_path (str): 图像文件路径。
        crop_size (tuple): 裁剪尺寸 (height, width)。
    
    返回:
        Tensor: 裁剪后的图像张量。
    """
    # 加载图像并转换为 RGB 格式
    img = Image.open(image_path).convert('RGB')
    
    # 获取原始图像尺寸
    width, height = img.size
    
    # 计算中心裁剪区域
    left = (width - crop_size[1]) // 2
    top = (height - crop_size[0]) // 2
    right = left + crop_size[1]
    bottom = top + crop_size[0]
    
    # 裁剪图像
    img_cropped = img.crop((left, top, right, bottom))
    
    # 转换为 PyTorch 张量并添加 batch 维度
    transform = transforms.ToTensor()
    img_tensor = transform(img_cropped).unsqueeze(0)
    
    return img_tensor, img_cropped

def image_to_tensor(image_path):
    """
    将 PIL 图像转换为 PyTorch 张量。
    
    参数:
        image (PIL.Image): 图像对象。
    
    返回:
        Tensor: 图像张量。
    """
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    
    return transform(img).unsqueeze(0), img

'''
张量输出
'''
def tensor_to_image(tensor):
    """
    将张量转换为 PIL 图像。
    
    参数:
        tensor (Tensor): 图像张量。
    
    返回:
        PIL.Image: 图像对象。
    """

    # 移除 batch 维度并确保值在 [0, 1] 范围内
    tensor = tensor.squeeze(0).clamp(0, 1)
    transform = transforms.ToPILImage()
    return transform(tensor)

def edge_to_image(tensor):
    """
    将channel=1的tensor转换为 PIL 图像。
    
    参数:
        tensor (Tensor): 图像张量。
    
    返回:
        PIL.Image: 图像对象。
    """
    
    B, C, H, W = tensor.shape
    scaled_tensor = torch.zeros(3, H, W)
    
    assert C == 1, "Dimension2 must be 1"
    
    min_vals = torch.min(tensor) 
    max_vals = torch.max(tensor)
    maxedge = torch.max((-1)*min_vals, max_vals)
    
    # 正向边缘与反向边缘归一化
    # posEdgeTensor = tensor.squeeze(0).clamp(0, 1) 
    # negEdgeTensor = tensor.squeeze(0).clamp(-1, 0) * (-1) 
    posEdgeTensor = tensor.squeeze(0).clamp(0, maxedge) *2 / maxedge
    negEdgeTensor = tensor.squeeze(0).clamp((-1)*maxedge, 0) *(-2) / maxedge
    scaled_tensor[0,:,:] = posEdgeTensor
    scaled_tensor[1,:,:] = negEdgeTensor
        
    transform = transforms.ToPILImage()
    return transform(scaled_tensor)

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
import numpy as np

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

def generate_graph_tensor(H: int, W: int, graph: str, R: float) -> torch.Tensor:
    """
    生成指定图形的Tensor,内部为白色(1.0),外部为黑色(0.0)
    
    示例
        circle_tensor = generate_graph_tensor(H=100, W=100, graph="Circle", R=40)
        rect_tensor = generate_graph_tensor(H=150, W=200, graph="Rectangle", R=60)
        tri_tensor = generate_graph_tensor(H=120, W=120, graph="Triangle", R=50)
    Args:
        H (int): 输出Tensor的高度
        W (int): 输出Tensor的宽度
        graph (str): 图形类型,支持"Circle","Triangle","Rectangle"
        R (float): 图形尺寸(圆形半径/矩形半边长/三角形外接圆半径)
    
    Returns:
        torch.Tensor: 形状为[1,3,H,W]的Tensor(NCHW格式)
    """
    # 计算中心点坐标(H和W的中间位置)
    cx, cy = (W-1)/2, (H-1)/2  # 使用浮点坐标避免整数截断
    
    # 生成坐标网格(HxW)
    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)  # x:列坐标, y:行坐标

    # 初始化全黑掩码(0.0)
    mask = np.zeros((H, W), dtype=np.float32)

    if graph.lower() == "circle":
        # 圆形：欧氏距离 <= 半径R
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        mask = (distance <= R).astype(np.float32)

    elif graph.lower() == "rectangle":
        # 矩形：中心向四周扩展R(半边长)
        x_min, x_max = cx - R, cx + R
        y_min, y_max = cy - R, cy + R
        mask = ((x_coords >= x_min) & (x_coords <= x_max) &
                (y_coords >= y_min) & (y_coords <= y_max)).astype(np.float32)

    elif graph.lower() == "triangle":
        # 等边三角形(外接圆半径R),三个顶点分布在圆周上(0°, 120°, 240°)
        angles = np.deg2rad([0, 120, 240])
        vertices = np.array([
            (cx + R * np.cos(angles[0]), cy + R * np.sin(angles[0])),
            (cx + R * np.cos(angles[1]), cy + R * np.sin(angles[1])),
            (cx + R * np.cos(angles[2]), cy + R * np.sin(angles[2]))
        ])

        # 计算各边的叉积判断点是否在三角形内部
        def edge_check(px, py, a, b):
            """判断点(px,py)是否在边ab的内侧(叉积法)"""
            return (b[0]-a[0])*(py - a[1]) - (b[1]-a[1])*(px - a[0])

        # 计算三个边的叉积值
        cross1 = edge_check(x_coords, y_coords, vertices[0], vertices[1])
        cross2 = edge_check(x_coords, y_coords, vertices[1], vertices[2])
        cross3 = edge_check(x_coords, y_coords, vertices[2], vertices[0])

        # 内部条件：所有叉积同号(或零)
        mask = ((cross1 >= -1e-6) & (cross2 >= -1e-6) & (cross3 >= -1e-6)) | \
               ((cross1 <= 1e-6) & (cross2 <= 1e-6) & (cross3 <= 1e-6))
        mask = mask.astype(np.float32)

    else:
        raise ValueError(f"不支持的图形类型: {graph},请选择['Circle','Triangle','Rectangle']")

    # 扩展为3通道并调整维度为[N,C,H,W]
    tensor = np.stack([mask]*3, axis=0)  # 形状[3,H,W]
    return torch.from_numpy(tensor).unsqueeze(0)  # 形状[1,3,H,W]
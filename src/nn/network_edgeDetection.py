import torch 
from torch import nn
import math

from trail.pic_tools import *
from trail.matrix_tools import *

def gaussian_kernel(size=8, sigma=1.5):
    kernel = torch.zeros(size, size)
    center = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel /= kernel.sum()  # 归一化总和为1
    return kernel

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel):
        super(FixedConvLayer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        with torch.no_grad():
            self.conv.weight.copy_(kernel)  # kernel 
        # 禁止梯度更新
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class EdgeDetectionLayer(nn.Module):
    def __init__(self, in_channels):
        super(EdgeDetectionLayer, self).__init__()
        
        # 定义卷积层：输入通道数为 in_channels，输出通道数为 out_channels，卷积核大小为 3x3
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 定义 Sobel 算子
        sobel_horizontal = torch.tensor([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]], dtype=torch.float32)
        sobel_vertical = torch.tensor([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]], dtype=torch.float32)
        kernel = torch.tensor([[1, 0, 1],
                              [1, 0, 1],
                              [1, 0, 1]], dtype=torch.float32)
        
        # 将 Sobel 算子扩展为 (1, 1, 3, 3) 的形状，并重复 in_channels 次
        sobel_horizontal = sobel_horizontal.view(1, 1, 3, 3).repeat(1, in_channels, 1, 1)
        sobel_vertical = sobel_vertical.view(1, 1, 3, 3).repeat(1, in_channels, 1, 1)
        kernel = kernel.view(1, 1, 3, 3).repeat(1, in_channels, 1, 1)
        
        # 设置卷积核权重
        self.conv.weight.data[0] = sobel_horizontal  # 第一个通道检测横向边缘
        self.conv.weight.data[1] = sobel_vertical    # 第二个通道检测纵向边缘
        self.conv.weight.data[2] = kernel
        
        # 固定卷积核权重，不参与训练
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class SmoothLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SmoothLayer, self).__init__()
        
        # 定义卷积层：输入通道数为 3，输出通道数为 3，卷积核大小为 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=3, bias=False)
        
        # 定义算子
        # kernel = torch.tensor([[0.025, 0.025, 0.025],
        #                       [0.025, 0.8, 0.025],
        #                       [0.025, 0.025, 0.025]], dtype=torch.float32)
        kernel = torch.full((kernel_size, kernel_size), 1/(kernel_size*kernel_size), dtype=torch.float32)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)
        
        # 设置卷积核权重
        for channel in range(3):
            self.conv.weight.data[channel] = kernel
        
        # 固定卷积核权重，不参与训练
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)
    
class SmoothLayerGauss(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sigma):
        super(SmoothLayerGauss, self).__init__()
        
        # 定义卷积层：输入通道数为 3，输出通道数为 3，卷积核大小为 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=3, bias=False)
        
        # 定义算子
        # kernel = torch.tensor([[0.025, 0.025, 0.025],
        #                       [0.025, 0.8, 0.025],
        #                       [0.025, 0.025, 0.025]], dtype=torch.float32)
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(1, 1, 1, 1)
        
        # 设置卷积核权重
        for channel in range(3):
            self.conv.weight.data[channel] = kernel
        
        # 固定卷积核权重，不参与训练
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

# # 前向传播
# if __name__ == "__main__":
#     # 输入数据：batch_size=1, channels=3, height=64, width=64
#     input_data = torch.randn(1, 3, 64, 64)
#     # 实例化卷积层
#     conv_layer = ConvLayer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
#     # 前向传播
#     output = conv_layer(input_data)
#     print("Output shape:", output.shape)


if __name__ == "__main__":
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    input_tensor, cropped_image = load_and_crop_image(image_path, crop_size=(128, 128))
    
    edge_layer_rgb = EdgeDetectionLayer(in_channels=3)
    output_rgb = edge_layer_rgb(input_tensor)
    
    smooth_layer_rgb = SmoothLayer(in_channels=3, out_channels=3, kernel_size=3)
    smooth_rgb = smooth_layer_rgb(input_tensor)
    smooth_edge = edge_layer_rgb(smooth_rgb)

    smooth_layer2_rgb = SmoothLayerGauss(in_channels=3, out_channels=3, kernel_size=3, sigma=1.5)
    smooth_rgb2 = smooth_layer2_rgb(input_tensor)
    smooth_edge2 = edge_layer_rgb(smooth_rgb2)    
    
    
    output_image0 = tensor_to_image(output_rgb[:, 0, :, :])
    output_image1 = tensor_to_image(output_rgb[:, 1, :, :])
    output_image2 = tensor_to_image(smooth_rgb)
    output_image3 = tensor_to_image(smooth_edge[:, 0, :, :])
    output_image4 = tensor_to_image(smooth_edge[:, 1, :, :])
    output_image5 = tensor_to_image(smooth_rgb2)
    output_image6 = tensor_to_image(smooth_edge2[:, 0, :, :])
    output_image7 = tensor_to_image(smooth_edge2[:, 1, :, :])
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 5))
    axes[0,0].imshow(cropped_image)
    axes[0,0].set_title('Cropped Input Image (128x128)')
    axes[0,1].imshow(output_image0)
    axes[0,1].set_title('Output 1 (128x128)')
    axes[0,2].imshow(output_image1)
    axes[0,2].set_title('Output 2 (128x128)')
    axes[1,0].imshow(output_image2)
    axes[1,0].set_title('Smoothed (128x128)')
    axes[1,1].imshow(output_image3)
    axes[1,1].set_title('Smoothed_edge1 (128x128)')
    axes[1,2].imshow(output_image4)
    axes[1,2].set_title('Smoothed_edge2 (128x128)')
    axes[2,0].imshow(output_image5)
    axes[2,0].set_title('Smoothed (128x128)')
    axes[2,1].imshow(output_image6)
    axes[2,1].set_title('Smoothed_edge1 (128x128)')
    axes[2,2].imshow(output_image7)
    axes[2,2].set_title('Smoothed_edge2 (128x128)')

    plt.show()
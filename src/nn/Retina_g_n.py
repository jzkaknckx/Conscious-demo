import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from trail.pic_trail import *

class RetinaLayer(nn.Module):
    def __init__(self, output_size, center_radius, min_kernel_size, max_kernel_size, alpha, beta):
        """
        参数:
        - output_size: 输出层大小 (e.g., 512)
        - center_radius: 中央圆形区域半径 (e.g., 128)
        - min_kernel_size: 最小池化核大小 (e.g., 1)
        - max_kernel_size: 最大池化核大小 (e.g., 9)
        - alpha: 池化核增长速度系数
        - beta: 池化核增长指数
        """
        super(RetinaLayer, self).__init__()
        self.output_size = output_size
        self.center_radius = center_radius
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.alpha = alpha
        self.beta = beta
        self.mapping_cache = {}  # 缓存映射关系

    def _compute_mapping(self, input_height, input_width):
        """
        计算输出到输入的映射关系并返回。
        返回值: 映射字典，键为(i, j)，值为(start_i, end_i, start_j, end_j, kernel_size)。
        """
        mapping = {}
        center_x, center_y = self.output_size / 2.0, self.output_size / 2.0
        scale_h = input_height / self.output_size
        scale_w = input_width / self.output_size

        for i in range(self.output_size):
            for j in range(self.output_size):
                # 计算到中心的欧几里得距离
                d = math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)

                # 计算池化核大小
                if d <= self.center_radius:
                    kernel_size = 1  # 中央圆形区域直接投射
                else:
                    kernel_size = self.min_kernel_size + self.alpha * ((d - self.center_radius) ** self.beta)
                    kernel_size = min(max(int(kernel_size), 1), self.max_kernel_size)
                    if kernel_size % 2 == 0:  # 确保池化核大小为奇数
                        kernel_size += 1

                # 计算输入层对应位置
                input_i = int(i * scale_h)
                input_j = int(j * scale_w)

                # 计算池化区域
                half_k = kernel_size // 2
                start_i = max(input_i - half_k, 0)
                end_i = min(input_i + half_k + 1, input_height)
                start_j = max(input_j - half_k, 0)
                end_j = min(input_j + half_k + 1, input_width)

                # 存储映射关系
                mapping[(i, j)] = (start_i, end_i, start_j, end_j, kernel_size)

        return mapping

    def forward(self, x):
        """
        输入:
        - x: 输入图片，形状为 [batch_size, channels, height, width]
        输出:
        - output: 输出特征图，形状为 [batch_size, channels, output_size, output_size]
        """
        batch_size, channels, input_height, input_width = x.size()

        # 检查缓存中是否有当前输入大小的映射
        input_size_key = (input_height, input_width)
        if input_size_key not in self.mapping_cache:
            self.mapping_cache[input_size_key] = self._compute_mapping(input_height, input_width)

        # 获取映射关系
        mapping = self.mapping_cache[input_size_key]

        # 初始化输出
        output = torch.zeros(batch_size, channels, self.output_size, self.output_size, device=x.device)

        for i in range(self.output_size):
            for j in range(self.output_size):
                start_i, end_i, start_j, end_j, kernel_size = mapping[(i, j)]

                if kernel_size == 1:
                    # 中央区域直接投射
                    input_i = int(i * (input_height / self.output_size))
                    input_j = int(j * (input_width / self.output_size))
                    output[:, :, i, j] = x[:, :, input_i, input_j]
                else:
                    # 周围区域平均池化
                    pooled_value = F.avg_pool2d(
                        x[:, :, start_i:end_i, start_j:end_j],
                        kernel_size=(end_i - start_i, end_j - start_j),
                        stride=1
                    )
                    output[:, :, i, j] = pooled_value.squeeze()

        return output

# 示例用法
if __name__ == "__main__":
    # 参数设置
    output_size = 512
    center_radius = 128  # 中央圆形区域半径
    min_kernel_size = 1
    max_kernel_size = 9
    alpha = 0.1  # 池化核增长速度
    beta = 1.0   # 线性增长

    # 创建层
    retina_layer = RetinaLayer(output_size, center_radius, min_kernel_size, max_kernel_size, alpha, beta)

    # 输入示例
    input_image = torch.randn(1, 3, 1024, 1024)  # 批量大小1，3通道，1024x1024
    output = retina_layer(input_image)
    print(f"outputshape: {output.shape}")  # 应为 [1, 3, 512, 512]
    
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    tensor, oringinal_image = image_to_tensor(image_path)
    output1 = tensor_to_image(retina_layer(tensor))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(oringinal_image)
    axes[0].set_title('Original')
    axes[1].imshow(output1)
    axes[1].set_title('Cropped')
    
    plt.show()
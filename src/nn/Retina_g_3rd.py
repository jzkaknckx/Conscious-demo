import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from trail.pic_trail import *
class RetinaLayer(nn.Module):
    def __init__(self, output_size, center_size, fixed_input_size, min_kernel_size, max_kernel_size, alpha, beta, visual_center=None):
        """
        参数:
        - output_size: 输出层大小 (例如 512)
        - center_size: 中央直接投射区域大小 (例如 256)
        - fixed_input_size: 固定的输入区域大小 (例如 4096)
        - min_kernel_size: 最小池化核大小 (例如 1)
        - max_kernel_size: 最大池化核大小 (例如 9)
        - alpha: 池化核增长速度系数
        - beta: 池化核增长指数
        - visual_center: 视觉中心坐标 (例如 (256, 256)), 默认为输出中心
        """
        super(RetinaLayer, self).__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.fixed_input_size = fixed_input_size
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.alpha = alpha
        self.beta = beta
        self.visual_center = visual_center if visual_center else (output_size // 2, output_size // 2)

    def forward(self, x):
        """
        输入:
        - x: 输入图片，形状为 [batch_size, channels, height, width]
        输出:
        - output: 输出特征图，形状为 [batch_size, channels, output_size, output_size]
        """
        batch_size, channels, input_height, input_width = x.size()

        # 处理输入大小
        if input_height > self.fixed_input_size or input_width > self.fixed_input_size:
            # 截断到 fixed_input_size
            x = x[:, :, :self.fixed_input_size, :self.fixed_input_size]
        elif input_height < self.fixed_input_size or input_width < self.fixed_input_size:
            # 填充到 fixed_input_size，使用边缘像素填充以实现平滑过渡
            pad_h = max(self.fixed_input_size - input_height, 0)
            pad_w = max(self.fixed_input_size - input_width, 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')  # 使用边缘像素填充

        # 初始化输出
        output = torch.zeros(batch_size, channels, self.output_size, self.output_size, device=x.device)

        # 计算中央区域的输出坐标
        center_start = (self.output_size - self.center_size) // 2
        center_end = center_start + self.center_size

        # 计算输入与输出的缩放比例（基于 fixed_input_size）
        scale_h = self.fixed_input_size / self.output_size
        scale_w = self.fixed_input_size / self.output_size

        for i in range(self.output_size):
            for j in range(self.output_size):
                # 计算到视觉中心的距离
                di = abs(i - self.visual_center[0])
                dj = abs(j - self.visual_center[1])
                distance = max(di, dj)

                # 根据距离计算池化核大小
                if i >= center_start and i < center_end and j >= center_start and j < center_end:
                    kernel_size = 1  # 中央区域直接投射
                else:
                    kernel_size = self.min_kernel_size + self.alpha * (distance ** self.beta)
                    kernel_size = min(max(int(kernel_size), 1), self.max_kernel_size)
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # 确保池化核大小为奇数

                # 计算输入层对应位置
                input_i = int(i * scale_h)
                input_j = int(j * scale_w)

                # 计算池化区域
                half_k = kernel_size // 2
                start_i = max(input_i - half_k, 0)
                end_i = min(input_i + half_k + 1, self.fixed_input_size)
                start_j = max(input_j - half_k, 0)
                end_j = min(input_j + half_k + 1, self.fixed_input_size)

                # 处理池化区域
                if start_i < end_i and start_j < end_j:
                    if kernel_size == 1:
                        output[:, :, i, j] = x[:, :, input_i, input_j]
                    else:
                        pooled_value = F.avg_pool2d(
                            x[:, :, start_i:end_i, start_j:end_j],
                            kernel_size=(end_i - start_i, end_j - start_j),
                            stride=1
                        )
                        output[:, :, i, j] = pooled_value.squeeze()
                else:
                    output[:, :, i, j] = x[:, :, input_i, input_j]  # 边界外使用对应位置值

        return output

# 示例用法
if __name__ == "__main__":
    # 参数设置
    output_size = 512
    center_size = 256
    fixed_input_size = 4096  # 固定的输入区域大小
    min_kernel_size = 1
    max_kernel_size = 9
    alpha = 0.1
    beta = 1.0
    visual_center = (256, 256)  # 自定义视觉中心

    # 创建层
    retina_layer = RetinaLayer(output_size, center_size, fixed_input_size, min_kernel_size, max_kernel_size, alpha, beta, visual_center)

    # 输入示例
    input_image = torch.randn(1, 3, 2048, 2048)  # 小于 fixed_input_size
    output = retina_layer(input_image)
    print(f"输出形状: {output.shape}")  # [1, 3, 512, 512]

    # 测试超大输入
    input_image_large = torch.randn(1, 3, 8192, 8192)  # 大于 fixed_input_size
    output_large = retina_layer(input_image_large)
    print(f"超大输入输出形状: {output_large.shape}")  # [1, 3, 512, 512]    
    
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    tensor, oringinal_image = image_to_tensor(image_path)
    output1 = tensor_to_image(retina_layer(tensor))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(oringinal_image)
    axes[0].set_title('Original')
    axes[1].imshow(output1)
    axes[1].set_title('Cropped')
    
    plt.show()
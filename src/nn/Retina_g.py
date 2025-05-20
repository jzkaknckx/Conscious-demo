import torch
import torch.nn as nn
import torch.nn.functional as F

from trail.pic_trail import *

class RetinaLayer(nn.Module):
    def __init__(self, output_size, center_size, min_kernel_size, max_kernel_size, alpha, beta):
        """
        参数:
        - output_size: 输出层大小 (e.g., 512)
        - center_size: 中央区域大小 (e.g., 256)
        - min_kernel_size: 最小池化核大小 (e.g., 1)
        - max_kernel_size: 最大池化核大小 (e.g., 9)
        - alpha: 池化核增长速度系数
        - beta: 池化核增长指数
        """
        super(RetinaLayer, self).__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        """
        输入:
        - x: 输入图片，形状为 [batch_size, channels, height, width]
        输出:
        - output: 输出特征图，形状为 [batch_size, channels, output_size, output_size]
        """
        batch_size, channels, input_height, input_width = x.size()
        output = torch.zeros(batch_size, channels, self.output_size, self.output_size, device=x.device)

        # 计算中央区域的输入和输出坐标
        center_start_input = (input_height - self.center_size) // 2
        center_end_input = center_start_input + self.center_size
        center_start_output = (self.output_size - self.center_size) // 2
        center_end_output = center_start_output + self.center_size

        # 中央区域直接投射
        output[:, :, center_start_output:center_end_output, center_start_output:center_end_output] = \
            x[:, :, center_start_input:center_end_input, center_start_input:center_end_input]

        # 边缘区域处理
        for i in range(self.output_size):
            for j in range(self.output_size):
                # 跳过中央区域
                if center_start_output <= i < center_end_output and center_start_output <= j < center_end_output:
                    continue

                # 计算到中心的距离
                di = abs(i - self.output_size // 2)
                dj = abs(j - self.output_size // 2)
                d = max(di, dj)

                # 计算池化核大小
                kernel_size = self.min_kernel_size + self.alpha * (d ** self.beta)
                kernel_size = min(max(int(kernel_size), 1), self.max_kernel_size)
                if kernel_size % 2 == 0:  # 确保池化核大小为奇数
                    kernel_size += 1

                # 计算输入层对应位置
                scale_h = input_height / self.output_size
                scale_w = input_width / self.output_size
                input_i = int(i * scale_h)
                input_j = int(j * scale_w)

                # 计算池化区域
                half_k = kernel_size // 2
                input_start_i = max(input_i - half_k, 0)
                input_end_i = min(input_i + half_k + 1, input_height)
                input_start_j = max(input_j - half_k, 0)
                input_end_j = min(input_j + half_k + 1, input_width)

                # 平均池化
                pooled_value = F.avg_pool2d(
                    x[:, :, input_start_i:input_end_i, input_start_j:input_end_j],
                    kernel_size=(input_end_i - input_start_i, input_end_j - input_start_j),
                    stride=1
                )
                output[:, :, i, j] = pooled_value.squeeze()

        return output

# 示例用法
if __name__ == "__main__":
    # 参数设置
    output_size = 512
    center_size = 256
    min_kernel_size = 1
    max_kernel_size = 9
    alpha = 0.1  # 池化核增长速度
    beta = 1.0   # 线性增长

    # 创建层
    retina_layer = RetinaLayer(output_size, center_size, min_kernel_size, max_kernel_size, alpha, beta)

    # 输入示例
    input_image = torch.randn(1, 3, 1024, 1024)  # 批量大小1，3通道，1024x1024
    output = retina_layer(input_image)
    print(f"输出形状: {output.shape}")  # 应为 [1, 3, 512, 512]
    
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    tensor, oringinal_image = image_to_tensor(image_path)
    output1 = tensor_to_image(retina_layer(tensor))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(oringinal_image)
    axes[0].set_title('Original')
    axes[1].imshow(output1)
    axes[1].set_title('Cropped')
    
    plt.show()
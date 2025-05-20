import torch
import torch.nn as nn

class FixedConvLayer(nn.Module):
    '''
    固定卷积层
    '''
    def __init__(self, in_channels, out_channels, kernel_size, kernel):
        super(FixedConvLayer, self).__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # 设置固定的卷积核
        with torch.no_grad():
            self.conv.weight.copy_(kernel)  # kernel 是您提供的张量
        # 禁止梯度更新
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)
    
    
    
class ProjectionLayer(nn.Module):
    '''
        投射层
    '''
    def forward(self, x1, x2):
        # 在通道维度上拼接第二层和第三层的特征图
        return torch.cat([x1, x2], dim=1)
    
    
    
class DynamicLayer(nn.Module):
    '''
        连接层
    '''
    def __init__(self, initial_neurons, input_size):
        super(DynamicLayer, self).__init__()
        self.neurons = initial_neurons  # 当前神经元数量
        self.input_size = input_size    # 输入特征的维度
        # 初始化权重矩阵
        self.weight = nn.Parameter(torch.randn(self.neurons, self.input_size))  # 输入到神经元的权重
        self.self_weight = nn.Parameter(torch.randn(self.neurons, self.neurons))  # 自连接权重
        self.feedback_weight = nn.Parameter(torch.randn(self.input_size, self.neurons))  # 反馈权重

    def forward(self, x, feedback=False):
        # x: (batch_size, input_size)
        # 计算初始输出
        output = torch.matmul(x, self.weight.t())
        # 自连接循环（模拟同一层神经元交互）
        for _ in range(5):  # 假设迭代5次，您可调整
            output = torch.matmul(output, self.self_weight.t())
            output = torch.relu(output)  # 添加激活函数
        if feedback:
            # 计算反馈到第四层的输入
            feedback_input = torch.matmul(output, self.feedback_weight.t())
            return output, feedback_input
        return output

    def add_neurons(self, num_new_neurons):
        # 动态增加神经元
        new_weight = torch.randn(num_new_neurons, self.input_size)
        new_self_weight = torch.randn(self.neurons + num_new_neurons, num_new_neurons)
        new_feedback_weight = torch.randn(self.input_size, num_new_neurons)
        # 更新权重矩阵
        self.weight = nn.Parameter(torch.cat([self.weight, new_weight], dim=0))
        self.self_weight = nn.Parameter(torch.cat([self.self_weight, new_self_weight[:, :self.neurons]], dim=0))
        self.self_weight = nn.Parameter(torch.cat([self.self_weight, new_self_weight], dim=1))
        self.feedback_weight = nn.Parameter(torch.cat([self.feedback_weight, new_feedback_weight], dim=1))
        self.neurons += num_new_neurons



class CustomNet(nn.Module):
    '''
        网络
    '''
    def __init__(self, kernel1, kernel2, initial_neurons):
        super(CustomNet, self).__init__()
        # 假设输入是3通道图像
        self.conv1 = FixedConvLayer(3, 16, 3, kernel1)  # 第二层：3 -> 16通道
        self.conv2 = FixedConvLayer(16, 32, 3, kernel2) # 第三层：16 -> 32通道
        self.projection = ProjectionLayer()             # 第四层：拼接
        # 第五层输入维度为第四层通道数 (16 + 32 = 48) 乘以展平后的空间维度
        self.dynamic_layer = DynamicLayer(initial_neurons, 48 * 32 * 32)  # 假设输入图像为32x32

    def forward(self, x):
        # 第一层输入 x: (batch_size, 3, H, W)
        x1 = self.conv1(x)          # 第二层输出
        x2 = self.conv2(x1)         # 第三层输出
        x_proj = self.projection(x1, x2)  # 第四层：拼接
        # 展平第四层特征图
        x_flat = x_proj.view(x_flat.size(0), -1)
        # 第五层计算，包含反馈
        output, feedback_input = self.dynamic_layer(x_flat, feedback=True)
        # 将反馈加到第四层
        x_proj += feedback_input.view_as(x_proj)
        return output
    
    
    
# 定义固定的卷积核（示例）
kernel1 = torch.randn(16, 3, 3, 3)  # 第二层卷积核
kernel2 = torch.randn(32, 16, 3, 3) # 第三层卷积核

# 初始化网络
model = CustomNet(kernel1, kernel2, initial_neurons=100)

# 输入示例（假设批大小为4，图像大小为32x32）
input_image = torch.randn(4, 3, 32, 32)
output = model(input_image)

# 动态增加神经元
model.dynamic_layer.add_neurons(50)  # 增加50个神经元

# 定义优化器，只训练第五层的参数
optimizer = torch.optim.SGD(model.dynamic_layer.parameters(), lr=0.01)
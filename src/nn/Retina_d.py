import torch
import torch.nn as nn
import torch.nn.functional as F

from trail.pic_trail import *

class EfficientRetinaLayer(nn.Module):
    """
        模拟视网膜映射
        将前一层输入按与中央凹距离调整投射精度,中央直接映射,外部池化
        
        Args:
            output_size: 输出尺寸
            center_size: 中心区域尺寸
            gamma: 池化核衰减速度
            tau: 指数衰减系数
            eps: 防止除0
    
    """
    def __init__(self, 
                 output_size=512, 
                 center_size=256, 
                 gamma=0.8, 
                 tau=0.5, 
                 eps=1e-6):
        super().__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.gamma = gamma  # 池化核衰减速度
        self.tau = tau      # 指数衰减系数
        self.eps = eps
        
        # 预计算坐标映射系统
        self.register_buffer('grid_map', self._precompute_grid_new())

    def _precompute_grid(self):
        """
            预计算输出特征图到输入特征图的坐标映射关系

        Returns:
            # (1, H, W, 2)
        """
        # 生成输出坐标系
        out_y = torch.linspace(-1, 1, self.output_size)
        out_x = torch.linspace(-1, 1, self.output_size)
        grid_y, grid_x = torch.meshgrid(out_y, out_x, indexing='ij')
        
        # 计算极坐标参数
        rho = torch.sqrt(grid_x**2 + grid_y**2)  # 归一化半径
        
        # 初始化输入坐标映射
        in_grid = torch.zeros(self.output_size, self.output_size, 2)
        
        # 中心区域直接映射
        center_mask = rho <= (self.center_size/self.output_size)
        in_grid[center_mask] = torch.stack([grid_x[center_mask], 
                                           grid_y[center_mask]], dim=-1)
        
        # 外围区域动态计算
        outer_mask = ~center_mask
        scaled_rho = rho[outer_mask]
        
        # 指数衰减映射函数
        scale_factor = self.tau ** (scaled_rho * self.output_size / self.center_size)
        scaled_rho = scaled_rho * scale_factor
        
        # 角度保持连续
        theta = torch.atan2(grid_y[outer_mask], grid_x[outer_mask])
        
        # 更新外围坐标
        in_grid[outer_mask] = torch.stack([
            scaled_rho * torch.cos(theta),
            scaled_rho * torch.sin(theta)
        ], dim=-1)
        
        return in_grid.unsqueeze(0)  # (1, H, W, 2)

    def _precompute_grid_new(self):
        # 生成输出坐标系
        out_y = torch.linspace(-1, 1, self.output_size)
        out_x = torch.linspace(-1, 1, self.output_size)
        grid_y, grid_x = torch.meshgrid(out_y, out_x, indexing='ij')
        
        # 计算极坐标参数
        rho = torch.sqrt(grid_x**2 + grid_y**2)  # 归一化半径
        
        # 初始化输入坐标映射
        in_grid = torch.zeros(self.output_size, self.output_size, 2)
        
        # 中心区域直接映射（保持原样）
        center_radius = self.center_size / self.output_size
        center_mask = rho <= center_radius
        
        in_grid[center_mask] = torch.stack([grid_x[center_mask], 
                                        grid_y[center_mask]], dim=-1)
        
        # 过渡区域平滑处理
        transition_width = 0.1  # 过渡带宽度比例
        transition_start = center_radius * (1 - transition_width)
        transition_end = center_radius * (1 + transition_width)
        
        # 计算平滑过渡权重（sigmoid函数）
        transition_ratio = (rho - transition_start) / (transition_end - transition_start)
        blend_weight = torch.sigmoid(10 * (transition_ratio - 0.5))  # 陡峭过渡
        
        # 外围区域坐标变换（保持原逻辑但应用平滑）
        outer_mask = rho > transition_start
        scaled_rho = rho[outer_mask]
        
        # 原始缩放因子
        original_scale = self.tau ** (scaled_rho * self.output_size / self.center_size)
        
        # 混合缩放因子
        blended_scale = 1 * (1 - blend_weight[outer_mask]) + original_scale * blend_weight[outer_mask]
        scaled_rho = scaled_rho * blended_scale
        
        # 极坐标转笛卡尔（保持角度连续）
        theta = torch.atan2(grid_y[outer_mask], grid_x[outer_mask])
        in_grid[outer_mask] = torch.stack([
            scaled_rho * torch.cos(theta),
            scaled_rho * torch.sin(theta)
        ], dim=-1)
        
        return in_grid.unsqueeze(0)

    def _dynamic_pooling(self, x):
        B, C, H, W = x.shape
        
        # 生成采样网格
        grid = self.grid_map.repeat(B, 1, 1, 1)
        
        # 自适应尺寸调整
        if H != W or H != self.output_size:
            grid = grid * torch.tensor([W/self.output_size, 
                                       H/self.output_size], 
                                      device=x.device)
        
        # 双线性采样+自适应池化
        sampled = F.grid_sample(x, grid, align_corners=False)
        
        # 动态模糊核
        # blur_kernel = self._get_blur_kernel(C, H, W)
        
        '''
        test
        '''
        straight_kernel = torch.tensor([1], dtype=torch.float32)
        straight_kernel = straight_kernel.view(1, 1, 1, 1).repeat(3, 1, 1, 1)
        '''
        end
        '''
                
        # return F.conv2d(sampled, blur_kernel, padding='same', groups=3), F.conv2d(sampled, straight_kernel, padding='same', groups=3)
        return sampled

    def _get_blur_kernel(self, C, H, W):
        # 基于相对位置的动态核尺寸
        max_radius = min(H, W) // 2
        kernel_size = int(max_radius ** self.gamma) | 1  # 保证奇数
        
        # 生成高斯核
        sigma = kernel_size / 3.0
        ax = torch.arange(-kernel_size//2, kernel_size//2, dtype=torch.float32)     #change1
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        kernel /= kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    def forward(self, x):
        # 尺寸预处理
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(x, size=self.output_size, mode='bilinear')
        
        # 动态池化处理
        return self._dynamic_pooling(x)

# 使用示例
if __name__ == "__main__":
    # 参数配置
    retina_layer = EfficientRetinaLayer(
        # output_size=512,
        # center_size=256,
        # gamma=0.8,   # 池化核增长指数
        # tau=0.5      # 精度衰减系数
    )
    retina_layer_03= EfficientRetinaLayer(tau=0.65)
    retina_layer_04= EfficientRetinaLayer(tau=0.70)
    retina_layer_075= EfficientRetinaLayer(tau=0.73)
    retina_layer_09= EfficientRetinaLayer(tau=0.75)
    
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    tensor, oringinal_image = image_to_tensor(image_path)
    
    input_tensor, image_cropped = load_and_crop_image(image_path, crop_size=(2048, 2048))     # 裁切
    
    #输入网络
    nopooling = retina_layer(tensor)

    nopooling_image = tensor_to_image(nopooling)
    nopooling_03 = tensor_to_image(retina_layer_03(tensor))
    nopooling_04 = tensor_to_image(retina_layer_04(tensor))
    nopooling_075 = tensor_to_image(retina_layer_075(tensor))
    nopooling_09 = tensor_to_image(retina_layer_09(tensor))
    
    #可视化
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes[0,0].imshow(oringinal_image)
    axes[0,0].set_title('Original')
    axes[0,1].imshow(image_cropped)
    axes[0,1].set_title('Cropped')
    # axes[0,2].imshow(output_image)
    # axes[0,2].set_title('Output')
    axes[0,3].imshow(nopooling_image)
    axes[0,3].set_title('test')
    axes[1,0].imshow(nopooling_03)
    axes[1,0].set_title('nopooling_03')
    axes[1,1].imshow(nopooling_04)
    axes[1,1].set_title('nopooling_04')
    axes[1,2].imshow(nopooling_075)
    axes[1,2].set_title('nopooling_075')
    axes[1,3].imshow(nopooling_09)
    axes[1,3].set_title('nopooling_09')
    
    plt.show()
    
    # 测试不同输入尺寸
    # input_tensor = torch.randn(1, 3, 256, 256)  # 小尺寸输入
    # output = retina_layer(input_tensor)
    # print(f"Input shape: {input_tensor.shape} -> Output shape: {output.shape}")

    # input_tensor = torch.randn(1, 3, 1024, 1024)  # 大尺寸输入
    # output = retina_layer(input_tensor)
    # print(f"Input shape: {input_tensor.shape} -> Output shape: {output.shape}")
    
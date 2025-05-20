import torch
import torch.nn as nn
import torch.nn.functional as F

from trail.pic_trail import *

class EfficientRetinaLayer(nn.Module):
    def __init__(self, 
                 output_size=512,
                 center_size=256,
                 gamma=0.8,
                 tau=0.5,
                 blend_width=0.15):
        super().__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.gamma = gamma
        self.tau = tau
        self.blend_width = blend_width  # 过渡带相对宽度
        
        # 预计算系统
        self.register_buffer('grid_map', self._precompute_grid())
        self.register_buffer('blend_weights', self._precompute_blending())

    def _precompute_grid(self):
        # 生成归一化极坐标系 (改进连续映射)
        out_y = torch.linspace(-1, 1, self.output_size)
        out_x = torch.linspace(-1, 1, self.output_size)
        grid_y, grid_x = torch.meshgrid(out_y, out_x, indexing='ij')
        rho = torch.sqrt(grid_x**2 + grid_y**2)
        
        # 连续缩放函数 (避免硬截断)
        scale_ratio = self.tau ** (rho * self.output_size / self.center_size)
        scaled_rho = rho * (1 + (scale_ratio - 1)) * F.sigmoid(10*(rho - 0.5*self.center_size/self.output_size))
        
        # 笛卡尔坐标重建
        theta = torch.atan2(grid_y, grid_x)
        new_x = scaled_rho * torch.cos(theta)
        new_y = scaled_rho * torch.sin(theta)
        
        return torch.stack([new_x, new_y], dim=-1).unsqueeze(0)

    def _precompute_blending(self):
        # 生成混合权重图 (高斯过渡)
        y = torch.linspace(-1, 1, self.output_size)
        x = torch.linspace(-1, 1, self.output_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        rho = torch.sqrt(xx**2 + yy**2)
        
        # 计算过渡带参数
        r_center = self.center_size / self.output_size
        transition_start = r_center * (1 - self.blend_width)
        transition_end = r_center * (1 + self.blend_width)
        
        # 高斯混合权重
        blend = torch.exp(-(rho - transition_start)**2 / (2*(self.blend_width/3)**2))
        blend = torch.clamp(1 - blend, 0, 1)
        
        return blend.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def _dynamic_processing(self, x):
        B, C, H, W = x.shape
        
        # 生成自适应网格
        grid = self.grid_map.repeat(B,1,1,1)
        if H != self.output_size or W != self.output_size:
            scale = torch.tensor([W/self.output_size, H/self.output_size], device=x.device)
            grid = grid * scale
            
        # 双阶段采样
        orig_sampled = F.grid_sample(x, grid, align_corners=False)
        # blurred = self._blur_with_varying_kernel(x, grid)
        
        '''
        change1
        '''
        straight_kernel = torch.tensor([1], dtype=torch.float32)
        straight_kernel = straight_kernel.view(1, 1, 1, 1).repeat(3, 1, 1, 1)
        conv_output = F.conv2d(x, straight_kernel, padding='same', groups=3)
        
        '''
        changeend
        '''
        
        # 混合输出
        # return orig_sampled * self.blend_weights + blurred * (1 - self.blend_weights)
        return orig_sampled * self.blend_weights + conv_output * (1 - self.blend_weights), orig_sampled * self.blend_weights, conv_output 
        

    def _blur_with_varying_kernel(self, x, grid):
        # 生成位置相关的模糊核
        B, C, H, W = x.shape
        y_coord = torch.linspace(-1, 1, H, device=x.device)
        x_coord = torch.linspace(-1, 1, W, device=x.device)
        gy, gx = torch.meshgrid(y_coord, x_coord, indexing='ij')
        rho = torch.sqrt(gx**2 + gy**2)
        
        # 动态核尺寸计算
        base_size = min(H, W) // 2
        kernel_sizes = (base_size * rho**self.gamma).int() | 1
        
        # 迭代处理每个独特核尺寸
        output = torch.zeros_like(x)
        for k_size in torch.unique(kernel_sizes):
            if k_size <= 1: continue
            mask = (kernel_sizes == k_size).float()
            sigma = k_size / 3.0
            kernel = self._gaussian_kernel(k_size, sigma)
            conv_output = F.conv2d(x, kernel, padding=k_size//2)
            output += conv_output * mask.unsqueeze(0)
            
        return output

    def _gaussian_kernel(self, size, sigma):
        ax = torch.arange(-size//2, size//2, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        return (kernel / kernel.sum()).view(1,1,size,size)

    def forward(self, x):
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(x, size=self.output_size, mode='bicubic', align_corners=False)
        return self._dynamic_processing(x)

# 使用示例
if __name__ == "__main__":
    # 参数配置
    retina_layer = EfficientRetinaLayer(
        # output_size=512,
        # center_size=256,
        # gamma=0.8,   # 池化核增长指数
        tau=0.6      # 精度衰减系数
    )
    retina_layer_03= EfficientRetinaLayer(tau=0.3)
    retina_layer_04= EfficientRetinaLayer(tau=0.4)
    retina_layer_075= EfficientRetinaLayer(tau=0.75)
    retina_layer_09= EfficientRetinaLayer(tau=0.9)
    
    image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
    tensor, oringinal_image = image_to_tensor(image_path)
    
    input_tensor, image_cropped = load_and_crop_image(image_path, crop_size=(512, 512))     # 裁切
    
    #输入网络
    output = retina_layer(tensor)

    output_image = tensor_to_image(output[0])
    
    
    #可视化
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes[0,0].imshow(oringinal_image)
    axes[0,0].set_title('Original')
    axes[0,1].imshow(output_image)
    axes[0,1].set_title('')
    axes[1,0].imshow(tensor_to_image(output[1]))
    axes[1,0].set_title('')
    axes[1,1].imshow(tensor_to_image(output[2]))
    axes[1,1].set_title('')
    
    plt.show()
    
    # 测试不同输入尺寸
    # input_tensor = torch.randn(1, 3, 256, 256)  # 小尺寸输入
    # output = retina_layer(input_tensor)
    # print(f"Input shape: {input_tensor.shape} -> Output shape: {output.shape}")

    # input_tensor = torch.randn(1, 3, 1024, 1024)  # 大尺寸输入
    # output = retina_layer(input_tensor)
    # print(f"Input shape: {input_tensor.shape} -> Output shape: {output.shape}")
    
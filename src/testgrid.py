import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from trail.pic_trail import *
torch.set_printoptions(profile="full",linewidth=1000)

image_path = 'picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg'  # 替换为你的图像路径'
tensor, oringinal_image = image_to_tensor(image_path)
x = tensor

fig, axes = plt.subplots(2, 2, figsize=(10, 5))

cropped_size = 1024
output_size = 512
center_size = 256
tau = 0.01
center_x = torch.tensor([128.0])
center_y = torch.tensor([128.0])



class PreprocessLayer(nn.Module):
    '''
        预处理
            切割为cropped_size*cropped_size,以(center_x, center_y)为中心,其余补零
            
        只能输入B = 1
        未限制(center_x, center_y)输入大小
    '''
    def __init__(self, cropped_size):
        super().__init__()
        self.cropped_size = cropped_size
        self.resize = T.Resize(
            cropped_size,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True
        )

    def forward(self, x, center_x, center_y):
        
        B, C, H, W = x.shape
        device = x.device
        
        if(center_x > H or center_y > W):
            print("Invalid Center")
            
        if(center_x == -1 ): center_x = W//2
        if(center_y == -1 ): center_y = H//2

        # 计算动态裁切窗口
        # crop_h = min(self.cropped_size, H)
        # crop_w = min(self.cropped_size, W)
        crop_h = crop_w = self.cropped_size
        
        # 计算裁切区域坐标
        # start_x = torch.clamp(torch.tensor(center_x - crop_w//2), 0, W - crop_w)
        # start_y = torch.clamp(torch.tensor(center_y - crop_h//2), 0, H - crop_h)
        start_x = center_x - crop_w//2
        start_y = center_y - crop_h//2

        if(center_x < crop_w//2 or W - center_x < crop_w//2 or center_y < crop_h//2 or H - center_y < crop_h//2 or H < crop_h or W < crop_w):
            x = F.pad(x, (crop_w//2, crop_w//2, crop_h//2, crop_h//2))
            start_x += crop_w//2
            start_y += crop_h//2
        
        # 执行裁切
        cropped = []
        
        crop = x[0, :, 
                    int(start_y):int(start_y+crop_h),
                    int(start_x):int(start_x+crop_w)]
        
        # 等比缩放并保持原始比例
        scale = min(self.cropped_size/crop_h, self.cropped_size/crop_w)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        
        resized = T.functional.resize(
            crop, 
            size=[new_h, new_w],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True
        )
        
        # 边缘填充
        pad_h = (self.cropped_size - new_h)
        pad_w = (self.cropped_size - new_w)
        padded = T.functional.pad(
            resized,
            padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2],
            padding_mode='edge'
        )
        cropped.append(padded)
        
        return torch.cat(cropped, dim=0).unsqueeze(0)    
    
preprocess = PreprocessLayer(cropped_size)
x = preprocess(x, -1, -1)

axes[0,0].imshow(oringinal_image)
axes[0,0].set_title('')
axes[1,0].imshow(tensor_to_image(x))
axes[1,0].set_title('')
    
'''
    trail for ProjectionLayer
'''
B, C, H_in, W_in = x.shape
device = x.device
assert H_in == W_in, "Input must be square"
cropped_size = H_in

# 生成输出网格坐标
i, j = torch.meshgrid(torch.arange(output_size, device=device, dtype=torch.float),
                        torch.arange(output_size, device=device, dtype=torch.float),
                        indexing='ij')  # (output_size, output_size)
center_out = (output_size - 1) / 2.0
dx = i - center_out
dy = j - center_out

# 计算极坐标
r_out = torch.sqrt(dx**2 + dy**2)
theta = torch.atan2(dy, dx)

# 计算输入半径r_in
mask = r_out > center_size
delta_r = torch.where(mask, r_out - center_size, torch.zeros_like(r_out))
r_in = torch.where(mask, tau * (delta_r) ** 3 / 6 + r_out, r_out)

# 转换为输入坐标
center_in = (cropped_size - 1) / 2.0
x_in = center_in + r_in * torch.cos(theta)
y_in = center_in + r_in * torch.sin(theta)
'''
trail
'''
grid_0 = torch.stack([y_in, x_in], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, o, o, 2)
# gridtensor_print(grid_0)

# 归一化到[-1, 1]
x_norm = (x_in / (cropped_size - 1)) * 2 - 1
y_norm = (y_in / (cropped_size - 1)) * 2 - 1

# 生成采样网格
grid = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, o, o, 2)

# gridtensor_print(grid)


# 应用网格采样
projected = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)


axes[1,1].imshow(tensor_to_image(projected))
axes[1,1].set_title('')
'''
    end
'''
plt.show()
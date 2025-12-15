import torch
import torchvision.transforms as transforms
from PIL import Image

# 设置图像尺寸和网格参数
image_size = 2048
grid_size = 40

# 创建一个全白的图像 (1通道，灰度图)
image = torch.ones((1, image_size, image_size))

# 计算网格线间距
cell_size = image_size // grid_size

# 绘制黑色网格线
for i in range(0, image_size + 1, cell_size):
    # 绘制垂直线
    if i < image_size:
        image[:, :, i] = 0  # 黑色
    # 绘制水平线
    if i < image_size:
        image[:, i, :] = 0  # 黑色

# 将张量转换为PIL图像
transform = transforms.ToPILImage()
pil_image = transform(image)

# 保存图像
pil_image.save('grid_image2048.png')

# 显示图像 (可选)
pil_image.show()

print("网格图像已生成并保存为 grid_image.png")
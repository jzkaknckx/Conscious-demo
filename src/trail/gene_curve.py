import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

class ArcGenerator:
    """
    生成左上四分之一圆弧的生成器
    从左到右：曲率逐渐增大
    从上到下：粗细逐渐增大
    """
    
    def __init__(self, 
                 canvas_size: Tuple[int, int] = (512, 512),
                 n_rows: int = 5,
                 n_cols: int = 5,
                 curvature_range: Tuple[float, float] = (0.3, 0.9),
                 thickness_range: Tuple[float, float] = (1.0, 10.0),
                 spacing: float = 0.1,
                 bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 arc_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        初始化参数
        
        参数:
        - canvas_size: 画布尺寸 (height, width)
        - n_rows: 行数 (粗细变化方向)
        - n_cols: 列数 (曲率变化方向)
        - curvature_range: 曲率范围 (min, max)，值越大曲率越大（半径越小）
        - thickness_range: 线条粗细范围 (min, max)
        - spacing: 圆弧之间的间距比例 (0-1之间)
        - bg_color: 背景颜色 (RGB, 0-1范围)
        - arc_color: 圆弧颜色 (RGB, 0-1范围)
        """
        self.canvas_size = canvas_size
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.curvature_range = curvature_range
        self.thickness_range = thickness_range
        self.spacing = spacing
        self.bg_color = torch.tensor(bg_color).view(3, 1, 1)
        self.arc_color = torch.tensor(arc_color).view(3, 1, 1)
        
        # 计算每个圆弧的网格位置
        self.cell_width = canvas_size[1] // n_cols
        self.cell_height = canvas_size[0] // n_rows
        
    def generate_arc_mask(self, 
                         center: Tuple[float, float],
                         radius: float,
                         thickness: float) -> torch.Tensor:
        """
        生成单个圆弧的mask
        
        参数:
        - center: 圆心坐标 (x, y)
        - radius: 半径
        - thickness: 圆弧粗细
        
        返回:
        - mask: 圆弧的mask (1, H, W)
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                             torch.arange(W, device=device), 
                             indexing='ij')
        x = x.float()
        y = y.float()
        
        # 计算每个点到圆心的距离
        cx, cy = center
        distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        
        # 计算角度（左上四分之一圆弧，角度范围在90-180度之间）
        # 注意：图像坐标中，y轴向下为正，所以角度需要调整
        angle = torch.atan2(cy - y, cx - x) * 180 / torch.pi  # 转换为角度
        
        # 创建圆弧mask
        # 条件1: 距离在半径±厚度/2范围内
        inner_radius = radius - thickness / 2
        outer_radius = radius + thickness / 2
        
        # 条件2: 角度在90-180度范围内（左上四分之一）
        # 调整角度范围以确保在左上角
        angle_mask = (angle >= 90) & (angle <= 180)
        
        # 距离mask
        distance_mask = (distance >= inner_radius) & (distance <= outer_radius)
        
        # 最终mask
        mask = angle_mask & distance_mask
        
        return mask.float()
    
    def calculate_arc_position(self, row: int, col: int, radius: float) -> dict:
        """
        计算圆弧的几何参数，确保不重叠
        
        参数:
        - row: 行索引
        - col: 列索引
        - radius: 半径
        
        返回:
        - 包含圆弧参数的字典
        """
        # 计算单元格内的可用区域（考虑间距）
        cell_margin_x = self.cell_width * self.spacing / 2
        cell_margin_y = self.cell_height * self.spacing / 2
        
        # 可用的绘图区域
        available_width = self.cell_width * (1 - self.spacing)
        available_height = self.cell_height * (1 - self.spacing)
        
        # 圆弧在单元格内的位置（左上角对齐）
        cell_x = col * self.cell_width + cell_margin_x
        cell_y = row * self.cell_height + cell_margin_y
        
        # 圆心位置：放在单元格的左上角
        center_x = cell_x
        center_y = cell_y
        
        # 确保半径不会超出可用区域
        max_radius = min(available_width, available_height)
        radius = min(radius, max_radius * 0.9)  # 留一点边距
        
        return {
            'center': (center_x, center_y),
            'radius': radius,
            'cell_bbox': (cell_x, cell_y, available_width, available_height)
        }
    
    def generate_all_arcs(self) -> torch.Tensor:
        """
        生成所有圆弧的图像
        
        返回:
        - image: 生成的图像 (3, H, W)
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化背景
        image = self.bg_color.expand(3, H, W).to(device)
        
        # 计算曲率和粗细的渐变
        curvatures = torch.linspace(self.curvature_range[0], 
                                   self.curvature_range[1], 
                                   self.n_cols)
        thicknesses = torch.linspace(self.thickness_range[0],
                                    self.thickness_range[1],
                                    self.n_rows)
        
        # 遍历所有网格位置
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # 计算当前圆弧的曲率（半径的倒数）
                curvature = curvatures[j]
                # 将曲率转换为半径（曲率越大，半径越小）
                # 使用指数函数确保半径变化更加自然
                base_radius = min(self.cell_width, self.cell_height) * 0.8
                radius = base_radius / (curvature * 2)
                
                # 获取当前圆弧的粗细
                thickness = thicknesses[i]
                
                # 计算圆弧位置
                arc_params = self.calculate_arc_position(i, j, radius)
                center = arc_params['center']
                radius = arc_params['radius']
                
                # 生成圆弧mask
                arc_mask = self.generate_arc_mask(center, radius, thickness)
                
                # 将圆弧绘制到图像上
                image = image * (1 - arc_mask) + self.arc_color.to(device) * arc_mask
        
        return image.cpu()
    
    def visualize(self, image: torch.Tensor = None, save_path: str = None):
        """
        可视化生成的图像
        
        参数:
        - image: 要可视化的图像，如果为None则重新生成
        - save_path: 保存路径，如果为None则不保存
        """
        if image is None:
            image = self.generate_all_arcs()
        
        # 转换为matplotlib可用的格式 (H, W, C)
        img_np = image.permute(1, 2, 0).numpy()
        
        # 创建图形
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 7))
        
        # 显示图像
        ax1.imshow(img_np)
        ax1.set_title('Generated Arcs Pattern')
        ax1.axis('off')
        
        # 显示参数信息
        info_text = f"""
        Parameters:
        - Canvas size: {self.canvas_size}
        - Grid: {self.n_rows} × {self.n_cols}
        - Curvature range: {self.curvature_range}
        - Thickness range: {self.thickness_range}
        - Spacing: {self.spacing}
        - Background color: {self.bg_color.flatten().tolist()}
        - Arc color: {self.arc_color.flatten().tolist()}
        """
        
        # ax2.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
        #         transform=ax2.transAxes, family='monospace')
        # ax2.axis('off')
        # ax2.set_title('Generation Parameters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Image saved to {save_path}")
        
        plt.show()
        
        return img_np
    
    def generate_animation_frames(self, n_frames: int = 60) -> List[torch.Tensor]:
        """
        生成动画帧，展示参数变化过程
        
        参数:
        - n_frames: 帧数
        
        返回:
        - frames: 帧列表
        """
        frames = []
        
        # 创建曲率变化的动画
        for frame_idx in range(n_frames):
            # 动态调整曲率范围
            t = frame_idx / n_frames
            dynamic_curvature = (
                self.curvature_range[0] + 
                (self.curvature_range[1] - self.curvature_range[0]) * t
            )
            
            # 创建临时生成器
            temp_gen = ArcGenerator(
                canvas_size=self.canvas_size,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                curvature_range=(0.1, dynamic_curvature),
                thickness_range=self.thickness_range,
                spacing=self.spacing,
                bg_color=self.bg_color.flatten().tolist(),
                arc_color=self.arc_color.flatten().tolist()
            )
            
            frame = temp_gen.generate_all_arcs()
            frames.append(frame)
        
        return frames


# 使用示例
def main():
    # 示例1：基本使用
    print("生成基本图案...")
    generator = ArcGenerator(
        canvas_size=(512, 512),
        n_rows=4,
        n_cols=10,
        curvature_range=(0.1, 1),
        thickness_range=(2.0, 8.0),
        spacing=0.001,
        bg_color=(1.0, 1.0, 1.0),  # 白色
        arc_color=(0.0, 0.0, 0.0)   # 黑色
    )
    
    # 生成并显示图像
    image = generator.generate_all_arcs()
    generator.visualize(image, save_path="pattern_arcs.png")
    
    # # 示例2：不同的参数设置
    # print("\n生成高分辨率图案...")
    # generator2 = ArcGenerator(
    #     canvas_size=(1024, 1024),
    #     n_rows=6,
    #     n_cols=6,
    #     curvature_range=(0.1, 0.9),
    #     thickness_range=(3.0, 15.0),
    #     spacing=0.1,
    #     bg_color=(1.0, 1.0, 1.0),
    #     arc_color=(0.0, 0.0, 0.0)
    # )
    
    # image2 = generator2.generate_all_arcs()
    # generator2.visualize(image2, save_path="arcs_pattern_hd.png")
    
    # # 示例3：创建动画（可选）
    # print("\n生成动画帧...")
    # frames = generator.generate_animation_frames(n_frames=30)
    # print(f"生成了 {len(frames)} 帧动画")
    
    # return image, image2


if __name__ == "__main__":
    # 运行主程序
    main()
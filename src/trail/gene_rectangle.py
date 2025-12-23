import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional

class RectangleGenerator:
    """
    生成不同长宽比和尺寸的长方形
    从左到右：长宽比逐渐增大（中间为正方形）
    从上到下：尺寸逐渐增大
    """
    
    def __init__(self,
                 canvas_size: Tuple[int, int] = (512, 512),
                 n_rows: int = 5,
                 n_cols: int = 5,
                 aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
                 size_range: Tuple[float, float] = (0.1, 0.4),
                 spacing: float = 0.15,
                 bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 rect_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 center_square: bool = True):
        """
        初始化参数
        
        参数:
        - canvas_size: 画布尺寸 (height, width)
        - n_rows: 行数 (尺寸变化方向)
        - n_cols: 列数 (长宽比变化方向)
        - aspect_ratio_range: 长宽比范围 (width/height)
        - size_range: 尺寸范围 (相对于单元格的尺寸比例)
        - spacing: 长方形之间的间距比例 (0-1之间)
        - bg_color: 背景颜色 (RGB, 0-1范围)
        - rect_color: 长方形颜色 (RGB, 0-1范围)
        - center_square: 中间是否为正方形
        """
        self.canvas_size = canvas_size
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.aspect_ratio_range = aspect_ratio_range
        self.size_range = size_range
        self.spacing = spacing
        self.bg_color = torch.tensor(bg_color).view(3, 1, 1)
        self.rect_color = torch.tensor(rect_color).view(3, 1, 1)
        self.center_square = center_square
        
        # 计算中间列索引（用于放置正方形）
        self.mid_col = n_cols // 2
        if n_cols % 2 == 0:
            self.mid_col = n_cols // 2 - 0.5  # 如果列数为偶数，中间在两列之间
        
        # 计算每个长方形的网格位置
        self.cell_width = canvas_size[1] // n_cols
        self.cell_height = canvas_size[0] // n_rows
        
    def create_rectangle_mask(self,
                            top_left: Tuple[float, float],
                            width: float,
                            height: float) -> torch.Tensor:
        """
        创建矩形mask
        
        参数:
        - top_left: 左上角坐标 (x, y)
        - width: 矩形宽度
        - height: 矩形高度
        
        返回:
        - mask: 矩形区域的mask (1, H, W)
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device),
                             indexing='ij')
        x = x.float()
        y = y.float()
        
        # 矩形边界
        x0, y0 = top_left
        x1 = x0 + width
        y1 = y0 + height
        
        # 创建矩形mask
        mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        
        return mask.float()
    
    def calculate_aspect_ratio(self, col: int) -> float:
        """
        计算指定列的长宽比
        
        参数:
        - col: 列索引
        
        返回:
        - 长宽比 (width/height)
        """
        # 线性插值计算长宽比
        if self.n_cols == 1:
            # 只有一列，设为中间值或正方形
            if self.center_square:
                return 1.0
            else:
                return (self.aspect_ratio_range[0] + self.aspect_ratio_range[1]) / 2
        
        # 计算长宽比序列
        aspect_ratios = []
        
        if self.center_square:
            # 中间为正方形，两侧对称
            left_half = np.linspace(self.aspect_ratio_range[0], 1.0, 
                                   self.mid_col + 1 if isinstance(self.mid_col, int) else int(np.ceil(self.n_cols/2)))
            right_half = np.linspace(1.0, self.aspect_ratio_range[1], 
                                    self.n_cols - len(left_half) + 1)
            
            # 合并并去除重复的1.0
            aspect_ratios = list(left_half[:-1]) + list(right_half)
            
            # 如果总数列数不符，进行调整
            if len(aspect_ratios) > self.n_cols:
                aspect_ratios = aspect_ratios[:self.n_cols]
            elif len(aspect_ratios) < self.n_cols:
                # 线性插值补充
                aspect_ratios = np.linspace(self.aspect_ratio_range[0], 
                                           self.aspect_ratio_range[1], 
                                           self.n_cols)
        else:
            # 线性渐变
            aspect_ratios = np.linspace(self.aspect_ratio_range[0],
                                       self.aspect_ratio_range[1],
                                       self.n_cols)
        
        return float(aspect_ratios[col])
    
    def calculate_size_factor(self, row: int) -> float:
        """
        计算指定行的尺寸因子
        
        参数:
        - row: 行索引
        
        返回:
        - 尺寸因子 (0-1之间)
        """
        # 从上到下尺寸逐渐增大
        if self.n_rows == 1:
            return (self.size_range[0] + self.size_range[1]) / 2
        
        # 线性插值
        size_factors = np.linspace(self.size_range[0], 
                                  self.size_range[1], 
                                  self.n_rows)
        
        return float(size_factors[row])
    
    def calculate_rectangle_parameters(self, row: int, col: int) -> dict:
        """
        计算单个长方形的参数
        
        参数:
        - row: 行索引
        - col: 列索引
        
        返回:
        - 包含长方形参数的字典
        """
        # 计算单元格内的可用区域（考虑间距）
        cell_margin_x = self.cell_width * self.spacing / 2
        cell_margin_y = self.cell_height * self.spacing / 2
        
        # 可用的绘图区域
        available_width = self.cell_width * (1 - self.spacing)
        available_height = self.cell_height * (1 - self.spacing)
        
        # 获取长宽比
        aspect_ratio = self.calculate_aspect_ratio(col)
        
        # 获取尺寸因子
        size_factor = self.calculate_size_factor(row)
        
        # 计算基础尺寸（基于可用区域和尺寸因子）
        base_size = min(available_width, available_height) * size_factor
        
        # 根据长宽比计算宽度和高度
        if aspect_ratio >= 1.0:
            # 宽 >= 高
            width = base_size * np.sqrt(aspect_ratio)
            height = base_size / np.sqrt(aspect_ratio)
        else:
            # 宽 < 高
            width = base_size * np.sqrt(aspect_ratio)
            height = base_size / np.sqrt(aspect_ratio)
        
        # 确保不超过可用区域
        width = min(width, available_width * 0.95)
        height = min(height, available_height * 0.95)
        
        # 计算在单元格内的位置（居中）
        cell_x = col * self.cell_width + cell_margin_x
        cell_y = row * self.cell_height + cell_margin_y
        
        # 居中放置
        rect_x = cell_x + (available_width - width) / 2
        rect_y = cell_y + (available_height - height) / 2
        
        return {
            'top_left': (rect_x, rect_y),
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'size_factor': size_factor,
            'cell_bbox': (cell_x, cell_y, available_width, available_height)
        }
    
    def generate_all_rectangles(self) -> torch.Tensor:
        """
        生成所有长方形的图像
        
        返回:
        - image: 生成的图像 (3, H, W)
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化背景
        image = self.bg_color.expand(3, H, W).to(device)
        
        # 收集所有参数用于调试
        all_params = []
        
        # 遍历所有网格位置
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # 计算长方形参数
                rect_params = self.calculate_rectangle_parameters(i, j)
                all_params.append(rect_params)
                
                # 创建长方形mask
                rect_mask = self.create_rectangle_mask(
                    rect_params['top_left'],
                    rect_params['width'],
                    rect_params['height']
                )
                
                # 将长方形绘制到图像上
                image = image * (1 - rect_mask) + self.rect_color.to(device) * rect_mask
        
        # 保存参数供调试使用
        self.all_params = all_params
        
        return image.cpu()
    
    def generate_with_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成带网格线的图像，用于调试
        
        返回:
        - image_with_grid: 带网格的图像
        - image_clean: 不带网格的图像
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 先生成长方形
        image_clean = self.generate_all_rectangles()
        
        # 创建网格线图像
        image_grid = self.bg_color.expand(3, H, W).to(device)
        
        # 添加网格线
        grid_color = torch.tensor([0.7, 0.7, 0.7]).view(3, 1, 1).to(device)
        line_thickness = 1
        
        # 绘制垂直线
        for col in range(1, self.n_cols):
            x = col * self.cell_width
            x_mask = self.create_rectangle_mask((x - line_thickness//2, 0), 
                                               line_thickness, H)
            image_grid = image_grid * (1 - x_mask) + grid_color * x_mask
        
        # 绘制水平线
        for row in range(1, self.n_rows):
            y = row * self.cell_height
            y_mask = self.create_rectangle_mask((0, y - line_thickness//2), 
                                               W, line_thickness)
            image_grid = image_grid * (1 - y_mask) + grid_color * y_mask
        
        # 合并图像
        image_with_grid = image_clean.clone()
        
        # 在空白区域添加网格线
        blank_mask = (image_clean == self.bg_color).all(dim=0).float()
        for c in range(3):
            image_with_grid[c] = image_with_grid[c] * (1 - blank_mask) + image_grid[c] * blank_mask
        
        return image_with_grid.cpu(), image_clean.cpu()
    
    def visualize(self, 
                  image: torch.Tensor = None,
                  show_grid: bool = False,
                  save_path: str = None,
                  show_params: bool = False):
        """
        可视化生成的图像
        
        参数:
        - image: 要可视化的图像，如果为None则重新生成
        - show_grid: 是否显示网格线
        - save_path: 保存路径，如果为None则不保存
        - show_params: 是否显示参数信息
        """
        if image is None:
            if show_grid:
                image, _ = self.generate_with_grid()
            else:
                image = self.generate_all_rectangles()
        
        # 转换为matplotlib可用的格式 (H, W, C)
        img_np = image.permute(1, 2, 0).numpy()
        
        # 创建图形
        if show_params:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax_img, ax_info, ax_debug = axes
        else:
            fig, ax_img = plt.subplots(1, 1, figsize=(8, 8))
            axes = [ax_img]
        
        # 显示图像
        ax_img.imshow(img_np)
        title = 'Generated Rectangles Pattern'
        if show_grid:
            title += ' (with grid)'
        ax_img.set_title(title)
        ax_img.axis('off')
        
        if show_params:
            # 显示参数信息
            info_text = f"""
            Generation Parameters:
            - Canvas size: {self.canvas_size}
            - Grid: {self.n_rows} × {self.n_cols}
            - Aspect ratio range: {self.aspect_ratio_range}
            - Size range: {self.size_range}
            - Spacing: {self.spacing}
            - Background color: {self.bg_color.flatten().tolist()}
            - Rectangle color: {self.rect_color.flatten().tolist()}
            - Center square: {self.center_square}
            """
            
            ax_info.text(0.1, 0.5, info_text, fontsize=10, 
                        verticalalignment='center',
                        transform=ax_info.transAxes, family='monospace')
            ax_info.axis('off')
            ax_info.set_title('Parameters')
            
            # 显示调试信息
            if hasattr(self, 'all_params') and self.all_params:
                # 计算每行的平均尺寸和每列的平均长宽比
                row_sizes = {}
                col_aspects = {}
                
                for i, params in enumerate(self.all_params):
                    row = i // self.n_cols
                    col = i % self.n_cols
                    
                    if row not in row_sizes:
                        row_sizes[row] = []
                    if col not in col_aspects:
                        col_aspects[col] = []
                    
                    row_sizes[row].append(params['size_factor'])
                    col_aspects[col].append(params['aspect_ratio'])
                
                debug_text = f"""
                Debug Information:
                - Total rectangles: {len(self.all_params)}
                - Row size factors (top to bottom):
                """
                
                for row in sorted(row_sizes.keys()):
                    avg_size = np.mean(row_sizes[row])
                    debug_text += f"  Row {row}: {avg_size:.3f}\n"
                
                debug_text += "\n- Column aspect ratios (left to right):\n"
                for col in sorted(col_aspects.keys()):
                    avg_aspect = np.mean(col_aspects[col])
                    shape_type = "Square" if abs(avg_aspect - 1.0) < 0.01 else "Rectangle"
                    debug_text += f"  Col {col}: {avg_aspect:.3f} ({shape_type})\n"
                
                ax_debug.text(0.1, 0.5, debug_text, fontsize=9,
                             verticalalignment='center',
                             transform=ax_debug.transAxes, family='monospace')
                ax_debug.axis('off')
                ax_debug.set_title('Debug Info')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Image saved to {save_path}")
        
        plt.show()
        
        return img_np
    
    def create_parameter_sweep(self,
                              n_variations: int = 9,
                              save_prefix: str = "rect_sweep"):
        """
        创建参数扫描，生成多个不同参数的图像
        
        参数:
        - n_variations: 变化数量
        - save_prefix: 保存文件前缀
        """
        # 定义要扫描的参数
        aspect_ratios = [(0.3, 3.0), (0.5, 2.0), (0.7, 1.5)]
        size_ranges = [(0.1, 0.3), (0.15, 0.4), (0.2, 0.5)]
        spacings = [0.1, 0.15, 0.2]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(min(n_variations, 9)):
            # 选择参数
            aspect_idx = idx % len(aspect_ratios)
            size_idx = (idx // 3) % len(size_ranges)
            spacing_idx = (idx // 6) % len(spacings)
            
            # 创建生成器
            gen = RectangleGenerator(
                canvas_size=(400, 400),
                n_rows=5,
                n_cols=5,
                aspect_ratio_range=aspect_ratios[aspect_idx],
                size_range=size_ranges[size_idx],
                spacing=spacings[spacing_idx],
                bg_color=(1.0, 1.0, 1.0),
                rect_color=(0.0, 0.0, 0.0),
                center_square=True
            )
            
            # 生成图像
            image = gen.generate_all_rectangles()
            img_np = image.permute(1, 2, 0).numpy()
            
            # 显示
            axes[idx].imshow(img_np)
            axes[idx].set_title(f"AR:{aspect_ratios[aspect_idx]}, "
                              f"Size:{size_ranges[size_idx]}, "
                              f"Spacing:{spacings[spacing_idx]}")
            axes[idx].axis('off')
            
            # 保存单独的文件
            plt.figure(figsize=(6, 6))
            plt.imshow(img_np)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_{idx}.png", dpi=120, bbox_inches='tight')
            plt.close()
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_grid.png", dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
def main():
    # 示例1：基本使用（中间有正方形）
    print("生成基本图案（中间有正方形）...")
    generator1 = RectangleGenerator(
        canvas_size=(512, 512),
        n_rows=5,
        n_cols=5,
        aspect_ratio_range=(0.05, 20.0),  # 宽高比范围
        size_range=(0.15, 0.35),        # 尺寸范围（相对于单元格）
        spacing=0.15,
        bg_color=(1.0, 1.0, 1.0),       # 白色背景
        rect_color=(0.0, 0.0, 0.0),     # 黑色长方形
        center_square=True              # 中间为正方形
    )
    
    # 生成并显示图像
    image1 = generator1.generate_all_rectangles()
    generator1.visualize(image1, save_path="trail/pattern_rect.png")
    
    # 示例2：没有中间正方形
    # print("\n生成没有中间正方形的图案...")
    # generator2 = RectangleGenerator(
    #     canvas_size=(600, 600),
    #     n_rows=7,
    #     n_cols=7,
    #     aspect_ratio_range=(0.3, 3.0),
    #     size_range=(0.1, 0.3),
    #     spacing=0.1,
    #     center_square=False
    # )
    
    # image2 = generator2.generate_all_rectangles()
    # generator2.visualize(image2, save_path="rectangles_no_center_square.png")
    
    # 示例3：宽屏布局
    # print("\n生成宽屏布局...")
    # generator3 = RectangleGenerator(
    #     canvas_size=(400, 800),
    #     n_rows=4,
    #     n_cols=8,
    #     aspect_ratio_range=(0.4, 2.5),
    #     size_range=(0.12, 0.3),
    #     spacing=0.12
    # )
    
    # 带网格线的可视化
    # image3_with_grid, image3_clean = generator3.generate_with_grid()
    # generator3.visualize(image3_with_grid, show_grid=True,
    #                     save_path="rectangles_wide_grid.png")
    
    # # 示例4：创建参数扫描
    # print("\n创建参数扫描...")
    # generator1.create_parameter_sweep(n_variations=9,
    #                                  save_prefix="parameter_sweep")
    
    # return image1, image2, image3_clean


if __name__ == "__main__":
    # 运行主程序
    main()
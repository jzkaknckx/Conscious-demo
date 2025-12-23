import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional
import math

class TriangleGenerator:
    """
    生成等腰三角形图案
    从左到右：顶角大小逐渐增大
    从上到下：中线逐渐旋转，第一行竖直，最后一行旋转180°后也竖直
    """
    
    def __init__(self,
                 canvas_size: Tuple[int, int] = (512, 512),
                 n_rows: int = 5,
                 n_cols: int = 5,
                 vertex_angle_range: Tuple[float, float] = (30.0, 120.0),
                 side_length: float = 80.0,
                 spacing: float = 0.15,
                 bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 triangle_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        初始化参数
        
        参数:
        - canvas_size: 画布尺寸 (height, width)
        - n_rows: 行数 (中线旋转变化方向)
        - n_cols: 列数 (顶角变化方向)
        - vertex_angle_range: 顶角范围 (度)
        - side_length: 等腰三角形的腰长
        - spacing: 三角形之间的间距比例 (0-1之间)
        - bg_color: 背景颜色 (RGB, 0-1范围)
        - triangle_color: 三角形颜色 (RGB, 0-1范围)
        """
        self.canvas_size = canvas_size
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.vertex_angle_range = vertex_angle_range
        self.side_length = side_length
        self.spacing = spacing
        self.bg_color = torch.tensor(bg_color).view(3, 1, 1)
        self.triangle_color = torch.tensor(triangle_color).view(3, 1, 1)
        
        # 计算每个三角形的网格位置
        self.cell_width = canvas_size[1] // n_cols
        self.cell_height = canvas_size[0] // n_rows
        
        # 角度转换为弧度
        self.vertex_angle_range_rad = (math.radians(vertex_angle_range[0]), 
                                      math.radians(vertex_angle_range[1]))
        
    def create_triangle_mask(self,
                           vertices: List[Tuple[float, float]]) -> torch.Tensor:
        """
        创建三角形mask
        
        参数:
        - vertices: 三角形三个顶点的坐标列表 [(x1,y1), (x2,y2), (x3,y3)]
        
        返回:
        - mask: 三角形区域的mask (1, H, W)
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device),
                             indexing='ij')
        x = x.float()
        y = y.float()
        
        # 获取三角形顶点
        v0_x, v0_y = vertices[0]
        v1_x, v1_y = vertices[1]
        v2_x, v2_y = vertices[2]
        
        # 计算向量
        v0_x_tensor = torch.tensor(v0_x, device=device)
        v0_y_tensor = torch.tensor(v0_y, device=device)
        v1_x_tensor = torch.tensor(v1_x, device=device)
        v1_y_tensor = torch.tensor(v1_y, device=device)
        v2_x_tensor = torch.tensor(v2_x, device=device)
        v2_y_tensor = torch.tensor(v2_y, device=device)
        
        # 计算重心坐标
        denom = ((v1_y_tensor - v2_y_tensor) * (v0_x_tensor - v2_x_tensor) + 
                (v2_x_tensor - v1_x_tensor) * (v0_y_tensor - v2_y_tensor))
        
        # 避免除零错误
        denom = torch.where(denom == 0, 1e-10, denom)
        
        a = ((v1_y_tensor - v2_y_tensor) * (x - v2_x_tensor) + 
             (v2_x_tensor - v1_x_tensor) * (y - v2_y_tensor)) / denom
        b = ((v2_y_tensor - v0_y_tensor) * (x - v2_x_tensor) + 
             (v0_x_tensor - v2_x_tensor) * (y - v2_y_tensor)) / denom
        c = 1 - a - b
        
        # 判断点是否在三角形内
        mask = (a >= 0) & (b >= 0) & (c >= 0) & (a <= 1) & (b <= 1) & (c <= 1)
        
        return mask.float()
    
    def calculate_vertex_angle(self, col: int) -> float:
        """
        计算指定列的顶角大小
        
        参数:
        - col: 列索引
        
        返回:
        - 顶角 (弧度)
        """
        # 线性插值计算顶角
        if self.n_cols == 1:
            mid_angle = (self.vertex_angle_range_rad[0] + 
                        self.vertex_angle_range_rad[1]) / 2
            return mid_angle
        
        t = col / (self.n_cols - 1)
        angle = self.vertex_angle_range_rad[0] + t * (
            self.vertex_angle_range_rad[1] - self.vertex_angle_range_rad[0])
        
        return angle
    
    def calculate_rotation_angle(self, row: int) -> float:
        """
        计算指定行的旋转角度
        
        参数:
        - row: 行索引
        
        返回:
        - 旋转角度 (弧度)，从0到π (0°到180°)
        """
        # 第一行竖直(0°)，最后一行旋转180°后也竖直(π)
        if self.n_rows == 1:
            return 0.0
        
        # 计算旋转角度，从0到π线性变化
        t = row / (self.n_rows - 1)
        rotation_angle = t * math.pi  # 从0到π
        
        return rotation_angle
    
    def calculate_triangle_vertices(self, 
                                  center: Tuple[float, float],
                                  vertex_angle: float,
                                  rotation_angle: float) -> List[Tuple[float, float]]:
        """
        计算等腰三角形的三个顶点坐标
        
        参数:
        - center: 三角形底边中点坐标 (x, y)
        - vertex_angle: 顶角大小 (弧度)
        - rotation_angle: 旋转角度 (弧度)，中线旋转角
        
        返回:
        - vertices: 三角形三个顶点的坐标列表
        """
        center_x, center_y = center
        
        # 计算等腰三角形的几何参数
        # 腰长 = self.side_length
        # 顶角 = vertex_angle
        # 底角 = (π - vertex_angle) / 2
        
        # 计算底边的一半长度
        half_base = self.side_length * math.sin(vertex_angle / 2)
        
        # 计算中线长度（从顶点到底边中点）
        median_length = self.side_length * math.cos(vertex_angle / 2)
        
        # 初始状态（未旋转）的顶点坐标
        # 顶点在上方，底边在下方
        v0_x = center_x  # 顶点
        v0_y = center_y - median_length
        
        v1_x = center_x - half_base  # 底边左端点
        v1_y = center_y
        
        v2_x = center_x + half_base  # 底边右端点
        v2_y = center_y
        
        # 旋转三角形（绕底边中点旋转）
        cos_theta = math.cos(rotation_angle)
        sin_theta = math.sin(rotation_angle)
        
        # 将坐标转换为相对于旋转中心的向量
        vertices = []
        for (vx, vy) in [(v0_x, v0_y), (v1_x, v1_y), (v2_x, v2_y)]:
            # 转换为相对坐标
            dx = vx - center_x
            dy = vy - center_y
            
            # 旋转
            dx_rot = dx * cos_theta - dy * sin_theta
            dy_rot = dx * sin_theta + dy * cos_theta
            
            # 转换回绝对坐标
            vx_rot = center_x + dx_rot
            vy_rot = center_y + dy_rot
            
            vertices.append((vx_rot, vy_rot))
        
        return vertices
    
    def calculate_triangle_position(self, row: int, col: int) -> dict:
        """
        计算单个三角形的位置参数
        
        参数:
        - row: 行索引
        - col: 列索引
        
        返回:
        - 包含三角形参数的字典
        """
        # 计算单元格内的可用区域（考虑间距）
        cell_margin_x = self.cell_width * self.spacing / 2
        cell_margin_y = self.cell_height * self.spacing / 2
        
        # 可用的绘图区域
        available_width = self.cell_width * (1 - self.spacing)
        available_height = self.cell_height * (1 - self.spacing)
        
        # 三角形在单元格内的中心位置
        cell_x = col * self.cell_width + cell_margin_x
        cell_y = row * self.cell_height + cell_margin_y
        
        center_x = cell_x + available_width / 2
        center_y = cell_y + available_height / 2
        
        return {
            'center': (center_x, center_y),
            'cell_bbox': (cell_x, cell_y, available_width, available_height)
        }
    
    def generate_all_triangles(self) -> torch.Tensor:
        """
        生成所有三角形的图像
        
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
                # 计算顶角大小
                vertex_angle = self.calculate_vertex_angle(j)
                
                # 计算旋转角度
                rotation_angle = self.calculate_rotation_angle(i)
                
                # 计算三角形位置
                position = self.calculate_triangle_position(i, j)
                center = position['center']
                
                # 计算三角形顶点
                vertices = self.calculate_triangle_vertices(
                    center, vertex_angle, rotation_angle)
                
                # 保存参数
                all_params.append({
                    'row': i,
                    'col': j,
                    'vertex_angle_deg': math.degrees(vertex_angle),
                    'rotation_angle_deg': math.degrees(rotation_angle),
                    'center': center,
                    'vertices': vertices
                })
                
                # 创建三角形mask
                triangle_mask = self.create_triangle_mask(vertices)
                
                # 将三角形绘制到图像上
                image = image * (1 - triangle_mask) + self.triangle_color.to(device) * triangle_mask
        
        # 保存参数供调试使用
        self.all_params = all_params
        
        return image.cpu()
    
    def generate_with_annotations(self) -> torch.Tensor:
        """
        生成带标注的图像，显示中线和角度
        
        返回:
        - image: 带标注的图像
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 先生成基本图像
        image = self.bg_color.expand(3, H, W).to(device)
        
        # 遍历所有网格位置
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # 计算顶角大小
                vertex_angle = self.calculate_vertex_angle(j)
                
                # 计算旋转角度
                rotation_angle = self.calculate_rotation_angle(i)
                
                # 计算三角形位置
                position = self.calculate_triangle_position(i, j)
                center = position['center']
                
                # 计算三角形顶点
                vertices = self.calculate_triangle_vertices(
                    center, vertex_angle, rotation_angle)
                
                # 创建三角形mask
                triangle_mask = self.create_triangle_mask(vertices)
                
                # 将三角形绘制到图像上
                image = image * (1 - triangle_mask) + self.triangle_color.to(device) * triangle_mask
                
                # 绘制中线和顶点标记
                if i % max(1, self.n_rows // 3) == 0 or i == self.n_rows - 1:  # 每间隔几行标注一次
                    # 获取顶点和底边中点
                    vertex = vertices[0]
                    base_center = center
                    
                    # 计算中线方向
                    line_length = 60
                    end_x = base_center[0] + line_length * math.sin(rotation_angle)
                    end_y = base_center[1] - line_length * math.cos(rotation_angle)
                    
                    # 绘制中线
                    line_mask = self.create_line_mask(
                        base_center, (end_x, end_y), thickness=2)
                    line_color = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1).to(device)  # 红色
                    image = image * (1 - line_mask) + line_color * line_mask
                    
                    # 绘制顶点标记
                    vertex_radius = 4
                    vertex_mask = self.create_circle_mask(vertex, vertex_radius)
                    vertex_color = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1).to(device)  # 蓝色
                    image = image * (1 - vertex_mask) + vertex_color * vertex_mask
        
        return image.cpu()
    
    def create_line_mask(self, 
                        start: Tuple[float, float],
                        end: Tuple[float, float],
                        thickness: float = 2) -> torch.Tensor:
        """
        创建线段mask
        
        参数:
        - start: 起点坐标
        - end: 终点坐标
        - thickness: 线条粗细
        
        返回:
        - mask: 线段的mask
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device),
                             indexing='ij')
        x = x.float()
        y = y.float()
        
        x0, y0 = start
        x1, y1 = end
        
        # 计算点到线段的距离
        # 线段向量
        dx = x1 - x0
        dy = y1 - y0
        
        # 线段长度的平方
        segment_length_sq = dx * dx + dy * dy
        
        # 避免除零
        segment_length_sq = torch.where(segment_length_sq == 0, 1e-10, segment_length_sq)
        
        # 计算投影参数
        t = ((x - x0) * dx + (y - y0) * dy) / segment_length_sq
        t = torch.clamp(t, 0, 1)
        
        # 最近点坐标
        closest_x = x0 + t * dx
        closest_y = y0 + t * dy
        
        # 距离的平方
        distance_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
        
        # 创建mask
        mask = distance_sq <= (thickness / 2) ** 2
        
        return mask.float()
    
    def create_circle_mask(self,
                          center: Tuple[float, float],
                          radius: float) -> torch.Tensor:
        """
        创建圆形mask
        
        参数:
        - center: 圆心坐标
        - radius: 半径
        
        返回:
        - mask: 圆形的mask
        """
        H, W = self.canvas_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device),
                             indexing='ij')
        x = x.float()
        y = y.float()
        
        cx, cy = center
        distance_sq = (x - cx) ** 2 + (y - cy) ** 2
        mask = distance_sq <= radius ** 2
        
        return mask.float()
    
    def visualize(self, 
                  image: torch.Tensor = None,
                  show_annotations: bool = False,
                  save_path: str = None,
                  show_params: bool = False):
        """
        可视化生成的图像
        
        参数:
        - image: 要可视化的图像，如果为None则重新生成
        - show_annotations: 是否显示标注（中线）
        - save_path: 保存路径，如果为None则不保存
        - show_params: 是否显示参数信息
        """
        if image is None:
            if show_annotations:
                image = self.generate_with_annotations()
            else:
                image = self.generate_all_triangles()
        
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
        title = ' '
        if show_annotations:
            title += ' (带中线标注)'
        ax_img.set_title(title)
        ax_img.axis('off')
        
        if show_params:
            # 显示参数信息
            info_text = f"""
            生成参数:
            - 画布尺寸: {self.canvas_size}
            - 网格: {self.n_rows} × {self.n_cols}
            - 顶角范围: {self.vertex_angle_range}°
            - 腰长: {self.side_length}像素
            - 间距: {self.spacing}
            - 背景颜色: {self.bg_color.flatten().tolist()}
            - 三角形颜色: {self.triangle_color.flatten().tolist()}
            - 旋转范围: 0°到180°
            """
            
            ax_info.text(0.1, 0.5, info_text, fontsize=10, 
                        verticalalignment='center',
                        transform=ax_info.transAxes, family='monospace')
            ax_info.axis('off')
            ax_info.set_title('参数信息')
            
            # 显示调试信息
            if hasattr(self, 'all_params') and self.all_params:
                # 统计每列的顶角和每行的旋转角
                col_angles = {}
                row_rotations = {}
                
                for params in self.all_params:
                    col = params['col']
                    row = params['row']
                    
                    if col not in col_angles:
                        col_angles[col] = []
                    if row not in row_rotations:
                        row_rotations[row] = []
                    
                    col_angles[col].append(params['vertex_angle_deg'])
                    row_rotations[row].append(params['rotation_angle_deg'])
                
                debug_text = f"""
                调试信息:
                - 总三角形数: {len(self.all_params)}
                - 列顶角变化 (从左到右):
                """
                
                for col in sorted(col_angles.keys()):
                    avg_angle = np.mean(col_angles[col])
                    debug_text += f"  列 {col}: {avg_angle:.1f}°\n"
                
                debug_text += "\n- 行旋转角变化 (从上到下):\n"
                for row in sorted(row_rotations.keys()):
                    avg_rotation = np.mean(row_rotations[row])
                    # 判断是否为竖直方向
                    is_vertical = abs(avg_rotation % 180) < 1 or abs(avg_rotation % 180 - 180) < 1
                    direction = "竖直" if is_vertical else f"旋转 {avg_rotation:.1f}°"
                    debug_text += f"  行 {row}: {direction}\n"
                
                # 显示顶部和底部的三角形对比
                if self.n_rows > 1:
                    top_idx = 0
                    bottom_idx = (self.n_rows - 1) * self.n_cols
                    if bottom_idx < len(self.all_params):
                        top_params = self.all_params[top_idx]
                        bottom_params = self.all_params[bottom_idx]
                        debug_text += f"""
                对比信息:
                - 顶部三角形 (行0, 列0):
                  顶角: {top_params['vertex_angle_deg']:.1f}°
                  旋转角: {top_params['rotation_angle_deg']:.1f}°
                  顶点朝上
                - 底部三角形 (行{self.n_rows-1}, 列0):
                  顶角: {bottom_params['vertex_angle_deg']:.1f}°
                  旋转角: {bottom_params['rotation_angle_deg']:.1f}°
                  顶点朝下 (旋转180°)
                """
                
                ax_debug.text(0.1, 0.5, debug_text, fontsize=8,
                             verticalalignment='center',
                             transform=ax_debug.transAxes, family='monospace')
                ax_debug.axis('off')
                ax_debug.set_title('调试信息')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到 {save_path}")
        
        plt.show()
        
        return img_np
    
    def visualize_rotation_sequence(self):
        """
        可视化旋转序列，显示每行的旋转角度
        """
        # 生成所有参数
        if not hasattr(self, 'all_params') or not self.all_params:
            self.generate_all_triangles()
        
        # 提取每行的旋转角度
        row_angles = {}
        for params in self.all_params:
            row = params['row']
            if row not in row_angles:
                row_angles[row] = []
            row_angles[row].append(params['rotation_angle_deg'])
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 旋转角度随行的变化
        ax1 = axes[0]
        rows = sorted(row_angles.keys())
        avg_angles = [np.mean(row_angles[row]) for row in rows]
        
        # 计算理论值（0到180°线性）
        theoretical_angles = [row / (self.n_rows - 1) * 180 for row in rows]
        
        ax1.plot(rows, avg_angles, 'bo-', linewidth=2, markersize=8, label='实际值')
        ax1.plot(rows, theoretical_angles, 'r--', linewidth=2, label='理论值 (0-180°线性)')
        ax1.set_xlabel('行索引')
        ax1.set_ylabel('旋转角度 (°)')
        ax1.set_title('中线旋转角度随行的变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记竖直位置
        for angle in [0, 90, 180]:
            ax1.axhline(y=angle, color='g', linestyle=':', alpha=0.5)
            ax1.text(rows[-1] + 0.1, angle, f'{angle}°', 
                    verticalalignment='center', color='g')
        
        # 2. 旋转方向示意图
        ax2 = axes[1]
        
        # 创建角度示意图
        angles = np.linspace(0, np.pi, self.n_rows)
        radii = np.ones(self.n_rows) * 0.8
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            # 计算箭头方向
            dx = radius * np.sin(angle)
            dy = -radius * np.cos(angle)  # 负号因为y轴向下为正
            
            # 绘制箭头
            ax2.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.1, 
                     fc='red', ec='red', alpha=0.7)
            
            # 标记角度
            label_angle = np.degrees(angle)
            label_radius = radius + 0.1
            label_dx = label_radius * np.sin(angle)
            label_dy = -label_radius * np.cos(angle)
            ax2.text(label_dx, label_dy, f'{label_angle:.0f}°', 
                    ha='center', va='center', fontsize=9)
            
            # 标记行号
            row_label_radius = radius + 0.25
            row_dx = row_label_radius * np.sin(angle)
            row_dy = -row_label_radius * np.cos(angle)
            ax2.text(row_dx, row_dy, f'行{i}', 
                    ha='center', va='center', fontsize=8, color='blue')
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_aspect('equal')
        ax2.set_title('中线旋转方向示意图')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', alpha=0.3)
        ax2.axvline(x=0, color='k', alpha=0.3)
        
        # 添加说明
        ax2.text(0, -1.1, '↑ 顶点朝上 (0°)\n↓ 顶点朝下 (180°)', 
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('rotation_sequence.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_vertical_comparison(self, save_path: str = "vertical_comparison.png"):
        """
        创建顶部和底部三角形的对比图
        """
        if not hasattr(self, 'all_params') or not self.all_params:
            self.generate_all_triangles()
        
        # 获取顶部和底部的三角形
        top_row = 0
        bottom_row = self.n_rows - 1
        mid_col = self.n_cols // 2
        
        top_idx = top_row * self.n_cols + mid_col
        bottom_idx = bottom_row * self.n_cols + mid_col
        
        if top_idx >= len(self.all_params) or bottom_idx >= len(self.all_params):
            print("无法获取对比三角形")
            return
        
        top_params = self.all_params[top_idx]
        bottom_params = self.all_params[bottom_idx]
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 创建两个独立的图像
        for i, (params, ax) in enumerate(zip([top_params, bottom_params], axes)):
            # 创建空白画布
            H, W = 200, 200
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image = self.bg_color.expand(3, H, W).to(device)
            
            # 调整顶点坐标到新画布中心
            center_x, center_y = W/2, H/2
            vertices = params['vertices']
            original_center = params['center']
            
            # 重新计算三角形在新画布中的位置
            vertex_angle = math.radians(params['vertex_angle_deg'])
            rotation_angle = math.radians(params['rotation_angle_deg'])
            
            new_vertices = self.calculate_triangle_vertices(
                (center_x, center_y), vertex_angle, rotation_angle)
            
            # 创建三角形mask
            triangle_mask = self.create_triangle_mask(new_vertices)
            
            # 绘制三角形
            image = image * (1 - triangle_mask) + self.triangle_color.to(device) * triangle_mask
            
            # 绘制中线和顶点
            line_length = 40
            end_x = center_x + line_length * math.sin(rotation_angle)
            end_y = center_y - line_length * math.cos(rotation_angle)
            
            line_mask = self.create_line_mask(
                (center_x, center_y), (end_x, end_y), thickness=2)
            line_color = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1).to(device)
            image = image * (1 - line_mask) + line_color * line_mask
            
            # 绘制顶点标记
            vertex = new_vertices[0]
            vertex_radius = 4
            vertex_mask = self.create_circle_mask(vertex, vertex_radius)
            vertex_color = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1).to(device)
            image = image * (1 - vertex_mask) + vertex_color * vertex_mask
            
            # 转换并显示
            img_np = image.cpu().permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            
            # 添加标题
            row_text = "顶部" if i == 0 else "底部"
            direction_text = "顶点朝上" if i == 0 else "顶点朝下"
            ax.set_title(f"{row_text}三角形\n"
                        f"顶角: {params['vertex_angle_deg']:.1f}°\n"
                        f"旋转角: {params['rotation_angle_deg']:.1f}°\n"
                        f"{direction_text}")
            ax.axis('off')
        
        plt.suptitle('顶部与底部三角形对比 (相同列，中线都竖直)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
def main():
    # 示例1：基本使用
    # print("生成等腰三角形图案：中线旋转180°...")
    generator1 = TriangleGenerator(
        canvas_size=(720, 720),
        n_rows=6,  # 6行，从0°旋转到180°
        n_cols=5,
        vertex_angle_range=(40.0, 140.0),  # 顶角从40°到140°
        side_length=80.0,                  # 腰长70像素
        spacing=0.12,
        bg_color=(1.0, 1.0, 1.0),         # 白色背景
        triangle_color=(0.0, 0.0, 0.0)    # 黑色三角形
    )
    
    # 生成并显示图像
    image1 = generator1.generate_all_triangles()
    generator1.visualize(image1, save_path="pattern_.png")
    
    # # 示例2：带标注的版本
    # print("\n生成带中线标注的图案...")
    # generator2 = TriangleGenerator(
    #     canvas_size=(600, 600),
    #     n_rows=7,  # 7行，从0°旋转到180°
    #     n_cols=7,
    #     vertex_angle_range=(30.0, 150.0),
    #     side_length=65.0,
    #     spacing=0.1
    # )
    
    # image2 = generator2.generate_with_annotations()
    # generator2.visualize(image2, show_annotations=True,
    #                     save_path="triangles_with_medians_180.png")
    
    # # 示例3：可视化旋转序列
    # print("\n生成旋转序列可视化...")
    # generator1.visualize_rotation_sequence()
    
    # # 示例4：顶部和底部三角形对比
    # print("\n生成顶部和底部三角形对比...")
    # generator1.create_vertical_comparison("vertical_comparison.png")
    
    # # 示例5：奇数行和偶数行的区别
    # print("\n测试奇数行和偶数行...")
    # generators = []
    
    # for n_rows in [3, 4, 5, 6]:
    #     gen = TriangleGenerator(
    #         canvas_size=(300, 300),
    #         n_rows=n_rows,
    #         n_cols=4,
    #         vertex_angle_range=(50.0, 130.0),
    #         side_length=50.0,
    #         spacing=0.15
    #     )
    #     generators.append(gen)
        
    #     image = gen.generate_all_triangles()
    #     gen.visualize(image, show_params=False, 
    #                  save_path=f"triangles_{n_rows}_rows.png")
    
    # return generator1, generator2


if __name__ == "__main__":
    # 运行主程序
    main()
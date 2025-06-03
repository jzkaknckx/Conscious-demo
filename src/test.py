import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math

from trail.pic_tools import *
from trail.matrix_tools import *

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)

from typing import List, Optional, Tuple, Union

import torch
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np

class TensorSlicer:
    def __init__(self, tensor: torch.Tensor):
        """
        初始化TensorSlicer
        :param tensor: 要处理的多维张量
        """
        self.tensor = tensor
        self.original_shape = tensor.shape
        self.ndim = tensor.ndim
        self.dim_names = [f"dim{i}" for i in range(self.ndim)]  # 默认维度名称
        
    def set_dimension_names(self, names: List[str]):
        """
        设置维度名称
        :param names: 各维度的名称列表，长度必须与张量维度数相同
        """
        if len(names) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimension names, got {len(names)}")
        self.dim_names = names
        
    def slice_tensor(
        self,
        coord_dims: Tuple[int, int],
        content_dims: List[int],
        coord_ranges: Optional[Tuple[Union[slice, Tuple[int, int, int]], ...]] = None,
        content_ranges: Optional[List[Union[slice, Tuple[int, int, int]]]] = None
    ) -> torch.Tensor:
        """
        执行切片操作
        
        :param coord_dims: 作为坐标的两个维度索引 (dim_x, dim_y)
        :param content_dims: 作为内容展示的维度索引列表
        :param coord_ranges: 坐标维度的切片范围，格式为(slice_x, slice_y)
        :param content_ranges: 内容维度的切片范围列表
        :return: 切片后的张量
        """
        # 验证输入维度
        self._validate_dims(coord_dims, content_dims)
        
        # 处理默认切片范围
        coord_slices = self._process_slices(coord_ranges, 2)
        content_slices = self._process_slices(content_ranges, len(content_dims))
        
        # 构建完整切片
        full_slices = self._build_full_slices(coord_dims, content_dims, coord_slices, content_slices)
        
        # 应用切片
        sliced_tensor = self.tensor[full_slices]
        
        # 重排维度：坐标维度在前，内容维度在后
        final_dims = list(coord_dims) + content_dims
        return sliced_tensor.permute(final_dims)
    
    def display_slice_grid(
        self,
        coord_dims: Tuple[int, int],
        content_dims: List[int],
        coord_ranges: Optional[Tuple[Union[slice, Tuple[int, int, int]], ...]] = None,
        content_ranges: Optional[List[Union[slice, Tuple[int, int, int]]]] = None,
        max_elements_per_cell: int = 50,
        max_cell_width: int = 30,
        precision: int = 4,
        show_coordinates: bool = True
    ) -> None:
        """
        以网格形式展示切片结果
        
        :param coord_dims: 作为坐标的两个维度索引
        :param content_dims: 作为内容展示的维度索引列表
        :param coord_ranges: 坐标维度的切片范围
        :param content_ranges: 内容维度的切片范围
        :param max_elements_per_cell: 每个单元格最多显示的元素数量
        :param max_cell_width: 每个单元格的最大宽度（字符数）
        :param precision: 浮点数显示精度
        :param show_coordinates: 是否显示坐标值
        """
        # 执行切片
        sliced_tensor = self.slice_tensor(
            coord_dims, content_dims, coord_ranges, content_ranges
        )
        
        # 获取坐标维度信息
        coord_dim_names = [self.dim_names[d] for d in coord_dims]
        coord_dim_sizes = sliced_tensor.shape[:2]
        
        # 获取内容维度信息
        content_dim_names = [self.dim_names[d] for d in content_dims]
        content_dim_sizes = sliced_tensor.shape[2:]
        total_content_elements = np.prod(content_dim_sizes)
        
        # 将张量转换为numpy数组便于处理
        array = sliced_tensor.detach().cpu().numpy()
        
        # 打印摘要信息
        print("=" * 80)
        print("Tensor Slice Grid View")
        print("=" * 80)
        print(f"Coordinate Dimensions: {coord_dim_names} (shape: {coord_dim_sizes})")
        print(f"Content Dimensions: {content_dim_names} (shape: {content_dim_sizes})")
        print(f"Total Cells: {coord_dim_sizes[0] * coord_dim_sizes[1]}")
        print(f"Elements per Cell: {total_content_elements}")
        print("-" * 80)
        
        # 准备网格显示
        grid = []
        max_row_label_len = 0
        
        # 收集所有单元格内容
        for i in range(coord_dim_sizes[0]):
            row = []
            for j in range(coord_dim_sizes[1]):
                # 获取当前坐标位置的内容
                content = array[i, j]
                
                # 格式化单元格内容
                cell_content = self._format_cell_content(
                    content, 
                    max_elements_per_cell,
                    precision
                )
                
                # 如果内容太长则截断
                if len(cell_content) > max_cell_width:
                    cell_content = cell_content[:max_cell_width-3] + "..."
                
                row.append(cell_content)
            
            # 添加行标签
            row_label = f"{coord_dim_names[0]}={i}"
            max_row_label_len = max(max_row_label_len, len(row_label))
            grid.append((row_label, row))
        
        # 准备列标题
        col_labels = [f"{coord_dim_names[1]}={j}" for j in range(coord_dim_sizes[1])]
        
        # 打印列标题
        if show_coordinates:
            # 第一行：列标题
            header = " " * (max_row_label_len + 2)
            for label in col_labels:
                header += f" {label.center(max_cell_width)}"
            print(header)
            
            # 分隔线
            separator = "-" * (max_row_label_len + 2)
            for _ in col_labels:
                separator += "+" + "-" * max_cell_width
            print(separator)
        
        # 打印网格内容
        for row_label, row_cells in grid:
            # 行标签
            row_str = row_label.rjust(max_row_label_len) + " | "
            
            # 单元格内容
            for cell in row_cells:
                row_str += cell.center(max_cell_width) + " | "
            
            print(row_str)
        
        print("=" * 80)
    
    def _format_cell_content(
        self, 
        content: np.ndarray, 
        max_elements: int, 
        precision: int
    ) -> str:
        """
        格式化单元格内容
        :param content: 单元格数据（可能为标量或多维数组）
        :param max_elements: 最多显示的元素数量
        :param precision: 浮点数精度
        :return: 格式化后的字符串
        """
        # 如果是标量
        if content.size == 1:
            return self._format_value(content.item(), precision)
        
        # 展平数组以便处理
        flat_content = content.ravel()
        total_elements = flat_content.size
        
        # 如果元素数量少，全部显示
        if total_elements <= max_elements:
            elements = [self._format_value(x, precision) for x in flat_content]
            return "[" + ", ".join(elements) + "]"
        
        # 元素数量多，只显示部分
        first_half = [self._format_value(x, precision) for x in flat_content[:max_elements//2]]
        second_half = [self._format_value(x, precision) for x in flat_content[-max_elements//2:]]
        return "[" + ", ".join(first_half) + ", ..., " + ", ".join(second_half) + "]"
    
    def _format_value(self, value: float, precision: int) -> str:
        """
        格式化单个值
        :param value: 数值
        :param precision: 浮点数精度
        :return: 格式化后的字符串
        """
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        elif isinstance(value, complex):
            real = value.real
            imag = value.imag
            return f"{real:.{precision}f}{imag:+.{precision}f}j"
        else:
            return str(value)
    
    def _validate_dims(self, coord_dims: Tuple[int, int], content_dims: List[int]):
        """验证维度参数是否有效"""
        if len(coord_dims) != 2:
            raise ValueError("coord_dims must contain exactly two dimensions")
        
        if len(set(coord_dims)) != 2:
            raise ValueError("Coordinate dimensions must be distinct")
        
        if len(set(content_dims)) != len(content_dims):
            raise ValueError("Content dimensions contain duplicates")
        
        if set(coord_dims) & set(content_dims):
            raise ValueError("Coordinate and content dimensions overlap")
        
        all_dims = set(coord_dims) | set(content_dims)
        if all_dims != set(range(self.ndim)):
            raise ValueError(f"Dimensions must cover all axes. Missing: {set(range(self.ndim)) - all_dims}")

    def _process_slices(self, ranges: Optional[List], num_dims: int) -> List[slice]:
        """处理切片参数，转换为标准slice对象"""
        if ranges is None:
            return [slice(None)] * num_dims
        
        if len(ranges) != num_dims:
            raise ValueError(f"Expected {num_dims} slice ranges, got {len(ranges)}")
        
        processed = []
        for r in ranges:
            if isinstance(r, slice):
                processed.append(r)
            elif isinstance(r, tuple) and len(r) == 3:
                processed.append(slice(*r))
            elif isinstance(r, tuple) and len(r) == 2:
                processed.append(slice(r[0], r[1]))
            else:
                raise TypeError("Slice range must be slice object or tuple (start, stop[, step])")
        return processed

    def _build_full_slices(
        self,
        coord_dims: Tuple[int, int],
        content_dims: List[int],
        coord_slices: List[slice],
        content_slices: List[slice]
    ) -> Tuple[Union[slice, int], ...]:
        """构建应用于原始张量的完整切片元组"""
        full_slices = [slice(None)] * self.ndim
        
        # 设置坐标维度切片
        full_slices[coord_dims[0]] = coord_slices[0]
        full_slices[coord_dims[1]] = coord_slices[1]
        
        # 设置内容维度切片
        for dim, slc in zip(content_dims, content_slices):
            full_slices[dim] = slc
        
        return tuple(full_slices)

    def get_dimension_info(self, dim: int) -> str:
        """获取指定维度的信息"""
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Dimension {dim} out of range")
        
        return (f"Dimension {dim} ({self.dim_names[dim]}): Size={self.original_shape[dim]}, "
                f"Range=[0, {self.original_shape[dim]-1}]")

# 创建示例张量 (形状[2, 3, 4, 5, 6])
tensor = torch.randn(2, 3, 4, 5, 6) * 10

# 创建切片器并设置维度名称
slicer = TensorSlicer(tensor)
slicer.set_dimension_names(["batch", "channel", "height", "width", "feature"])

# 以网格形式展示切片结果
slicer.display_slice_grid(
    coord_dims=(2, 3),       # 选择 height 和 width 作为坐标
    content_dims=[0, 1, 4],  # 选择 batch, channel, feature 作为内容
    coord_ranges=[slice(0, 3), slice(0, 4)],  # 高度取0-2，宽度取0-3
    max_elements_per_cell=40,  # 每个单元格最多显示4个元素
    max_cell_width=300,        # 单元格最大宽度
    precision=2               # 浮点数精度为2位
)
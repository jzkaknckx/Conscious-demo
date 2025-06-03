"""
矩阵工具
"""

import numpy as np 
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
import math
from typing import List, Optional, Tuple, Union, Any, Dict


torch.set_printoptions(profile="full",linewidth=512)

def gridtensor_print(tensor):
    B, H, W, N = tensor.shape
    
    assert N == 2, "The 4th dim must be 2"
    tensor_p =  tensor.reshape(B, H, W*2)
    print(tensor_p)
    
def stream_draw(flow):
    '''
    流速场绘画
        输入形状必须为[1, H, W, 2]
    '''
    B, H, W, N = flow.shape
    
    assert B == 1, "Only 1 batch once"
    assert N == 2, "The 4th dim must be 2"
    
    # [1, H, W, 2] -> [H, W] * 2
    flow = flow.squeeze(0)
    u = flow[:,:,0].squeeze(-1)
    v = flow[:,:,1].squeeze(-1)
    sp = torch.sqrt(u**2 + v**2)
    
    y, x = torch.meshgrid(torch.arange(start=0, end=H, step=1, dtype=torch.float),
                        torch.arange(start=0, end=W, step=1, dtype=torch.float),
                        indexing='ij')
    
    fig = plt.figure() 
    gs = gridspec.GridSpec(nrows = 1, ncols = 1) 
    ax = fig.add_subplot(gs[0]) 
    strm = ax.streamplot(x.numpy(), y.numpy(), u.numpy(), v.numpy(), color = sp.numpy(),
                        linewidth = 2, cmap ='autumn') 

    fig.colorbar(strm.lines) 
    ax.set_title('Flow') 
    ax.invert_yaxis()  #y轴反向  

    # show plot 
    plt.tight_layout() 
    plt.show()
    
class TensorSlicer:
    '''
    使用示例
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
    '''
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

def matrix_disp(tensor):
    # 改进的预处理函数，支持更多维度
    def preprocess(t):
        t = t.detach().cpu().float()
        
        # 自动展开批次维度
        if t.dim() == 4 and t.shape[0] > 1:  # BCHW
            return 'batch', t.unbind(0), t.shape[0]
        if t.dim() == 5 and t.shape[0] > 1:  # BTHWC
            return 'batch', t.view(-1, *t.shape[2:]), t.shape[0]
        
        # 处理不同通道位置
        if t.dim() == 4:
            if t.shape[1] in [2, 3]:  # CHW
                return 'matrix', t.squeeze(0).permute(1,2,0), None
            return 'matrix', t.squeeze(0), None  # HWDC
        
        # 处理3D张量
        if t.dim() == 3:
            return 'matrix', t.permute(1,2,0) if t.shape[0]<4 else t, None
            
        return 'unknown', t, None

    # 动态渲染核心类
    class DynamicRenderer:
        def __init__(self, fig, ax, data, viewer_type, batch_size):
            self.fig = fig
            self.ax = ax
            self.full_data = data
            self.viewer_type = viewer_type
            self.batch_size = batch_size
            self.current_batch = 0
            self.cell_size = 1.0
            self.view_offset = (0, 0)
            self.rendered_cells = set()
            
            # 初始化UI
            self.init_controls()
            self.update_viewport(force=True)
            
            # 新增初始化参数
            self.initial_render_size = 20  # 初始显示区域边长（单位：单元格数）
            self.max_initial_cells = 400   # 最大初始渲染单元格数
            
            # 智能初始化视窗参数
            self.init_viewport_config(data.shape)
            self.update_viewport(force=True)
            
            # 事件绑定
            self.fig.canvas.mpl_connect('scroll_event', self.zooming)
            self.fig.canvas.mpl_connect('button_press_event', self.start_pan)
            self.fig.canvas.mpl_connect('motion_notify_event', self.panning)

        def init_viewport_config(self, shape):
            """智能计算初始显示参数"""
            h, w, _ = shape
            
            # 计算初始单元格大小
            base_size = max(h, w)
            self.cell_size = max(1, math.ceil(base_size / self.initial_render_size))
            
            # 计算初始视窗位置
            center_x = w * self.cell_size / 2
            center_y = h * self.cell_size / 2
            view_span = self.initial_render_size * self.cell_size / 2
            
            # 设置初始视窗范围
            self.ax.set_xlim(center_x - view_span, center_x + view_span)
            self.ax.set_ylim(center_y - view_span, center_y + view_span)

            def init_controls(self):
                """初始化控制按钮布局"""
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                
                # 控制面板布局
                control_y = 0.02
                self.btn_center = self.add_button('Center', (0.7, control_y), 
                                                self.jump_to_center)
                
                if self.viewer_type == 'batch':
                    self.add_button('Prev', (0.4, control_y), self.prev_batch)
                    self.add_button('Next', (0.5, control_y), self.next_batch)
                    self.batch_label = self.fig.text(
                        0.3, control_y, 
                        f'Batch: {self.current_batch+1}/{self.batch_size}',
                        ha='right'
                    )

        def add_button(self, label, pos, callback):
            """创建标准化按钮"""
            ax = self.fig.add_axes([pos[0], pos[1], 0.1, 0.04])
            btn = Button(ax, label)
            btn.on_clicked(callback)
            return btn

        def get_visible_area(self):
            """计算当前可见区域"""
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            return (x0, x1, y0, y1)

        def calculate_visible_cells(self, data_shape):
            """动态计算需要渲染的单元格范围"""
            h, w, _ = data_shape
            x0, x1, y0, y1 = self.get_visible_area()
            
            # 计算列范围（考虑缩放）
            start_col = max(0, int(x0 // self.cell_size - 1))
            end_col = min(w, int(x1 // self.cell_size + 1))
            
            # 计算行范围（考虑坐标系反转）
            start_row = max(0, int(y0 // self.cell_size - 1))
            end_row = min(h, int(y1 // self.cell_size + 1))
            
            return (start_row, end_row), (start_col, end_col)

        def render_cells(self, data, row_range, col_range):
            """高效渲染指定区域的单元格"""
            current_cells = set()
            for y in range(row_range[0], row_range[1]):
                for x in range(col_range[0], col_range[1]):
                    # 跳过已渲染的单元格
                    if (y, x) in self.rendered_cells:
                        current_cells.add((y, x))
                        continue
                    
                    # 动态绘制单元格
                    values = data[y, x].tolist()
                    self.ax.add_patch(Rectangle(
                        (x*self.cell_size, y*self.cell_size),
                        self.cell_size, self.cell_size,
                        facecolor='white', edgecolor='black'
                    ))
                    self.draw_text(x, y, values)
                    current_cells.add((y, x))
            
            # 清理不可见的单元格
            for cell in self.rendered_cells - current_cells:
                for artist in self.ax.artists + self.ax.texts:
                    if (artist.get_x() == cell[1]*self.cell_size and 
                        artist.get_y() == cell[0]*self.cell_size):
                        artist.remove()
            
            self.rendered_cells = current_cells

        def draw_text(self, x, y, values):
            """优化文本绘制方法"""
            y_pos = y * self.cell_size
            for i, val in enumerate(values):
                self.ax.text(
                    x + self.cell_size/2, 
                    y_pos + self.cell_size*(0.3 + 0.3*i),
                    f"{val:.2f}", 
                    ha='center', va='center',
                    fontsize=8*(1/math.log(self.cell_size+1)))
        
        def update_viewport(self, force=False):
            """核心更新方法"""
            data = self.get_current_data()
            h, w, _ = data.shape
        
            # 动态调整单元格大小
            if force or not hasattr(self, 'prev_cell_size'):
                self.ax.set_xlim(0, w*self.cell_size)
                self.ax.set_ylim(0, h*self.cell_size)
            
            # 计算可见区域
            row_range, col_range = self.calculate_visible_cells(data.shape)
            
            # 渲染可见单元格
            self.render_cells(data, row_range, col_range)
            
            # 更新显示
            self.ax.figure.canvas.draw_idle()
            self.prev_cell_size = self.cell_size

        # 事件处理改进
        def zooming(self, event):
            """平滑缩放处理"""
            base_scale = 1.5 if event.button == 'up' else 1/1.5
            self.cell_size *= base_scale ** (1 if event.key!='shift' else 0.5)
            self.update_viewport()

        def start_pan(self, event):
            """记录拖拽起始位置"""
            self.pan_start = (event.xdata, event.ydata) if event.inaxes else None

        def panning(self, event):
            """实时平移视图"""
            if not hasattr(self, 'pan_start') or self.pan_start is None:
                return
            
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            self.ax.set_xlim(self.ax.get_xlim() - dx)
            self.ax.set_ylim(self.ax.get_ylim() - dy)
            self.pan_start = (event.xdata, event.ydata)
            self.update_viewport()

        # 导航功能
        def jump_to_center(self, event):
            data = self.get_current_data()
            h, w, _ = data.shape
            self.ax.set_xlim(w*self.cell_size/2 - 5, w*self.cell_size/2 + 5)
            self.ax.set_ylim(h*self.cell_size/2 - 5, h*self.cell_size/2 + 5)
            self.update_viewport()

        def prev_batch(self, event):
            self.current_batch = max(0, self.current_batch-1)
            self.batch_label.set_text(f'Batch: {self.current_batch+1}/{self.batch_size}')
            self.update_viewport(force=True)

        def next_batch(self, event):
            self.current_batch = min(self.batch_size-1, self.current_batch+1)
            self.batch_label.set_text(f'Batch: {self.current_batch+1}/{self.batch_size}')
            self.update_viewport(force=True)

        def get_current_data(self):
            return self.full_data if self.viewer_type=='matrix' else \
                   self.full_data[self.current_batch]

    # 主执行流程
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    view_type, data, batch_size = preprocess(tensor)
    
    if view_type == 'unknown':
        print(f"Unsupported tensor shape: {tensor.shape}")
        return
    
    renderer = DynamicRenderer(fig, ax, data, view_type, batch_size)
    plt.show()
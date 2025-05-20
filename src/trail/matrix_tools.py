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
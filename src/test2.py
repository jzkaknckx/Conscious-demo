import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from IPython.display import display, clear_output

class HighDimTensorVisualizer:
    def __init__(self, tensor, spatial_dims=(2, 3), cache_size=1000):
        """
        参数：
        tensor: 输入的高维张量（numpy数组或PyTorch/TensorFlow张量）
        spatial_dims: 指定空间维度的索引元组（H, W），默认假设为第2和第3维（B, C, H, W, ...）
        cache_size: 缓存的最大数据量（单位：元素数）
        """
        self.tensor = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)  # 统一转换为numpy数组
        self.shape = self.tensor.shape
        self.H_dim, self.W_dim = spatial_dims
        self.H_size = self.shape[self.H_dim]
        self.W_size = self.shape[self.W_dim]
        
        # 提取非空间维度信息（过滤size=1的维度）
        self.other_dims = [i for i in range(len(self.shape)) if i not in {self.H_dim, self.W_dim}]
        self.valid_other_dims = [i for i in self.other_dims if self.shape[i] > 1]
        
        # 初始化缓存系统
        self.cache_size = cache_size
        self.current_cache = {}
        self.viewport = (slice(0, self.H_size), slice(0, self.W_size))  # 当前视口范围
        
        # 初始化交互组件
        self.dim_selector_x = widgets.Dropdown(description='X轴维度:', options=self._get_dim_options())
        self.dim_selector_y = widgets.Dropdown(description='Y轴维度:', options=self._get_dim_options())
        self.info_text = widgets.Text(description='维度信息:', value=self._get_dim_info())
        self.output = widgets.Output()
        
        # 初始化图形
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = GridSpec(self.H_size, self.W_size, figure=self.fig)
        self.axes = [[self.fig.add_subplot(self.gs[i, j]) for j in range(self.W_size)] for i in range(self.H_size)]
        self._update_viewport()
        
    def _get_dim_options(self):
        """生成可选维度的名称列表"""
        return [f'dim{i} ({self.shape[i]})' for i in self.valid_other_dims]
    
    def _get_dim_info(self):
        """生成当前显示的维度信息字符串"""
        if len(self.valid_other_dims) == 0:
            return "标量数据"
        return '[' + ','.join(f'dim{i}' for i in self.valid_other_dims) + ']'
    
    def _load_into_cache(self, h_slice, w_slice):
        """根据视口加载数据到缓存"""
        current_elements = 0
        new_cache = {}
        for h in h_slice:
            for w in w_slice:
                indices = [slice(None)] * len(self.shape)
                indices[self.H_dim] = h
                indices[self.W_dim] = w
                data_slice = tuple(indices)
                element = self.tensor[data_slice]
                element_size = np.prod(element.shape)
                
                # 控制缓存大小
                if current_elements + element_size > self.cache_size:
                    break
                new_cache[(h, w)] = element
                current_elements += element_size
        self.current_cache = new_cache
    
    def _update_viewport(self):
        """更新视口显示"""
        h_start, h_end = self.viewport[0].start or 0, self.viewport[0].stop or self.H_size
        w_start, w_end = self.viewport[1].start or 0, self.viewport[1].stop or self.W_size
        
        # 加载新缓存
        self._load_into_cache(range(h_start, h_end), range(w_start, w_end))
        
        # 绘制视口内的子图
        for i in range(h_start, h_end):
            for j in range(w_start, w_end):
                ax = self.axes[i][j]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                
                # 处理四维情况（只有C和空间维度）
                if len(self.valid_other_dims) == 1:
                    data = self.current_cache.get((i, j), np.array([0]))
                    ax.text(0.5, 0.5, f'{data.flatten()}', ha='center', va='center')
                    ax.set_title(f'({i},{j})')
                else:
                    ax.text(0.5, 0.5, f'({i},{j})', ha='center', va='center')
                    ax.set_facecolor('lightgray')
        
        self.fig.canvas.draw()
    
    def _on_click(self, event):
        """处理子图点击事件"""
        with self.output:
            clear_output(wait=True)
            if event.inaxes is None:
                return
            
            # 获取点击的坐标
            for i in range(self.H_size):
                for j in range(self.W_size):
                    if event.inaxes == self.axes[i][j]:
                        selected_pos = (i, j)
                        break
            
            # 显示详细数据
            print(f'点击位置：H={i}, W={j}')
            print('维度数据：')
            data = self.current_cache.get((i, j), np.array([0]))
            print(data)
            
            # 显示维度选择器（当维度>4时）
            if len(self.valid_other_dims) > 1:
                display(self.dim_selector_x, self.dim_selector_y)
    
    def show(self):
        """显示可视化界面"""
        display(self.info_text, self.output)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 生成测试数据（1,5,32,32,4） -> [B,C,H,W,dim4]
    test_tensor = np.random.rand(1, 5, 32, 32, 4)
    
    # 初始化可视化工具（指定空间维度为第2和第3维）
    visualizer = HighDimTensorVisualizer(test_tensor, spatial_dims=(2, 3))
    
    # 显示界面
    visualizer.show()
    
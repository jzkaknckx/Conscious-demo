import matplotlib.pyplot as plt
import numpy as np

# 假设RetinaModel已定义
class RetinaModel:
    def __call__(self, x, center_x, center_y):
        # 示例处理逻辑：返回以点击坐标为中心的5x5区域
        h, w = x.shape
        y = np.zeros_like(x)
        y0 = max(0, int(center_y)-2)
        y1 = min(h, int(center_y)+3)
        x0 = max(0, int(center_x)-2)
        x1 = min(w, int(center_x)+3)
        y[y0:y1, x0:x1] = x[y0:y1, x0:x1]
        return y

class ClickHandler:
    def __init__(self, ax, model, input_img, output_ax):
        self.ax = ax
        self.model = model
        self.input_img = input_img
        self.output_ax = output_ax
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # 获取点击坐标
        x, y = event.xdata, event.ydata
        print(f"\n点击坐标: ({x:.1f}, {y:.1f})")
        
        # 运行神经网络
        output = self.model(self.input_img, x, y)
        
        # 更新输出显示
        self.output_ax.clear()
        self.output_ax.imshow(output)
        self.output_ax.set_title("Model Output")
        event.canvas.draw()

# 主程序
if __name__ == "__main__":
    # 初始化模型和测试数据
    model = RetinaModel()
    input_img = np.random.rand(20, 20)  # 示例输入
    
    # 创建界面
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(input_img)
    ax1.set_title("Input Image")
    
    # 初始显示输出区域
    output_plot = ax2.imshow(np.zeros_like(input_img))
    ax2.set_title("Model Output")
    
    # 绑定点击处理器
    handler = ClickHandler(ax1, model, input_img, ax2)
    
    # 添加退出提示
    print("点击图像进行预测，关闭窗口退出程序...")
    plt.show()
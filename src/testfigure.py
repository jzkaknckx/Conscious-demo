import matplotlib.pyplot as plt

class ClickCoordinates:
    def __init__(self, ax):
        self.ax = ax
        self.coords = None
        self._cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
    
    def on_click(self, event):
        if event.inaxes == self.ax:
            # 获取数据坐标
            x = event.xdata
            y = event.ydata
            # 转换为数组索引（假设origin为默认的'upper'）
            # 注意：imshow的extent和origin会影响坐标转换，此处假设为默认设置
            x_int = int(round(x))
            y_int = int(round(y))
            self.coords = (x, y, x_int, y_int)
            print(f"坐标 (x, y): ({x:.2f}, {y:.2f})，数组索引 [行, 列]: [{y_int}, {x_int}]")
            # 可选：标记点击位置
            self.ax.plot(x, y, 'r+')
            self.ax.figure.canvas.draw()

'''
# 示例用法
if __name__ == "__main__":
    # 创建测试数据
    import numpy as np
    data = np.random.rand(5, 5)
    
    # 创建包含子图的图像
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # 在第一个子图显示图像并绑定点击事件
    ax1.imshow(data)
    
    
    # 在第二个子图显示另一幅图像
    ax2.imshow(data.T)

    plt.show()

    
    while True:
        click_handler1 = ClickCoordinates(ax1)
        click_handler2 = ClickCoordinates(ax2)
        a = click_handler1.coords
        b = click_handler2.coords
        if a != None or b != None :
            print(a, b)
'''

if __name__ == "__main__":
    # 创建测试数据
    import numpy as np
    data = np.random.rand(5, 5)
    
    # 创建包含子图的图像
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # 在第一个子图显示图像并绑定点击事件
    ax1.imshow(data)
    click_handler1 = ClickCoordinates(ax1)
    
    # 在第二个子图显示另一幅图像
    ax2.imshow(data.T)
    click_handler2 = ClickCoordinates(ax2)
    
    print(0)
    plt.show()
    
    # 窗口关闭后查看捕获的坐标（最后一个点击）
    print("1:", click_handler1.coords)
    print("2:", click_handler2.coords)
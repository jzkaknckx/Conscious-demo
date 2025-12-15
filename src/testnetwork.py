import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from collections import deque

from trail.pic_tools import *
from trail.matrix_tools import *

torch.set_printoptions(profile="full",linewidth=512)

from nns.Retina import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
    ,RetinaModel
)



class ClickHandlerO:
    def __init__(self, ax, model, input_img, centerVelocity, output_ax0, output_ax1, output_ax2):
        self.ax = ax
        self.model = model
        self.input_img = input_img
        self.centerVelocity = centerVelocity
        self.output_ax0 = output_ax0
        self.output_ax1 = output_ax1
        self.output_ax2 = output_ax2
        B, C, H, W = input_img.shape
        self.x_old = W // 2
        self.y_old = H // 2
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # 获取点击坐标
        x, y = int(event.xdata), int(event.ydata)
        print(f"New Center: ({x:.1f}, {y:.1f})")
        
        delta_x = x - self.x_old
        delta_y = y - self.y_old
        step = int(delta_x // self.centerVelocity)
        
        for s in range(0, step + 1):
            x_running = self.x_old + s * self.centerVelocity
            y_running = self.y_old + s * self.centerVelocity * delta_y // delta_x 
            
            print(f"Running: {s} / {step}. Curruent Center: ({x_running:.1f}, {y_running:.1f})")
            
            output, flowvelocity = self.model(self.input_img, x_running, y_running)
            # output, flowvelocity = self.model(self.input_img, x, y)
            
            
            # 更新输出显示
            self.output_ax0.clear()
            self.output_ax0.imshow(tensor_to_image(output))
            self.output_ax0.set_title("output")
            self.output_ax1.clear()
            self.output_ax1.imshow(edge_to_image(flowvelocity[0]))
            self.output_ax1.set_title("flowvelocity0")
            self.output_ax2.clear()
            self.output_ax2.imshow(edge_to_image(flowvelocity[1]))
            self.output_ax2.set_title("flowvelocity1")
            
            event.canvas.draw()
        
        self.x_old = x
        self.y_old = y

class ClickHandler:
    '''
    创建中心点移动序列
        to be optimized
    '''
    # 使用方法
    '''
    def update_display(coordinates):
        for x, y in coordinates:
            print(f"Stepping = ({x}, {y})")
            output, flowvelocity, diff = r(tensor, x, y)
            print(torch.max(flowvelocity[0]))
            print(torch.max(flowvelocity[1]))
            print(torch.max(flowvelocity[2]))
            print(torch.max(flowvelocity[3]))
            
            # flowv = torch.stack([flowvelocity[0] - flowvelocity[1], flowvelocity[2] - flowvelocity[3]], dim = -1).squeeze(0) 
            # print(flowv.shape)
            # stream_draw(flowv)
            # 更新输出显示
            axes[0,1].clear()
            axes[0,1].imshow(tensor_to_image(output))
            axes[0,1].set_title("Output")
            
            axes[1,0].clear()
            axes[1,0].imshow(edge_to_image(flowvelocity[0]))
            axes[1,0].set_title("Flow X+")
            
            axes[1,1].clear()
            axes[1,1].imshow(edge_to_image(flowvelocity[1]))
            axes[1,1].set_title("Flow X-")
            
            axes[2,0].clear()
            axes[2,0].imshow(edge_to_image(flowvelocity[2]))
            axes[2,0].set_title("Flow Y+")
            
            axes[2,1].clear()
            axes[2,1].imshow(edge_to_image(flowvelocity[3]))
            axes[2,1].set_title("Flow Y-")
            
            axes[0,2].clear()
            axes[0,2].imshow(tensor_to_image(diff))
            axes[0,2].set_title("Flow Y-")
            # 强制刷新画布
            plt.gcf().canvas.draw()
            plt.pause(0.001)  # 允许GUI处理事件
    
    # 创建点击处理器
    B, C, H, W = tensor.shape
    click_handler = ClickHandler(
        ax=axes[0,0],
        input_shape=(H, W),
        center_velocity=1,  
        callback=update_display
    )
    '''
    def __init__(self, ax, input_shape, center_velocity, callback):
        self.ax = ax
        self.input_shape = input_shape  # (H, W) 格式
        self.center_velocity = center_velocity
        self.callback = callback
        self.x_old = input_shape[1] // 2  # W//2
        self.y_old = input_shape[0] // 2  # H//2
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        """处理点击事件并生成坐标序列"""
        if event.inaxes != self.ax:
            return

        # 获取新坐标
        x_new = int(event.xdata)
        y_new = int(event.ydata)
        print(f"\nNew Center: ({x_new}, {y_new})")

        # 计算坐标变化量
        delta_x = x_new - self.x_old
        delta_y = y_new - self.y_old
        
        # 计算步数并生成坐标序列
        coordinates = []
        if delta_x != 0:  # 水平方向移动
            step = int(delta_x // self.center_velocity)
            for s in range(0, step + 1):
                x = self.x_old + s * self.center_velocity
                y = self.y_old + s * self.center_velocity * delta_y // delta_x
                coordinates.append((x, y))
                # print(f"Step {s}/{step}: ({x}, {y})")
        else:  # 垂直方向移动特殊处理
            step = abs(int(delta_y // self.center_velocity))
            step = step if delta_y >= 0 else -step
            for s in range(0, step + 1):
                x = self.x_old
                y = self.y_old + s * self.center_velocity * (1 if delta_y >=0 else -1)
                coordinates.append((x, y))
                # print(f"Step {s}/{step}: ({x}, {y})")

        # 执行回调函数传递坐标序列
        if self.callback:
            self.callback(coordinates)

        # 更新坐标记录
        self.x_old = x_new
        self.y_old = y_new

def smooth_moving(input1, input2, velocity):
    center_x0, center_y0 = input1
    center_x1, center_y1 = input2
    dx = center_x1 - center_x0
    dy = center_y1 - center_y0
    distance = np.sqrt(dx**2 + dy**2)
    steps = int(distance // velocity)
    x = np.linspace(center_x0, center_x1, steps+1)
    y = np.linspace(center_y0, center_y1, steps+1)
    return x, y


# 验证测试
if __name__ == "__main__":

    
    fig, axes = plt.subplots(3, 3, figsize=(10, 5))
    
    # 准备输入
    image_path = 'src/picture/v2-a2184227abddb98b3b7405e6033651ff_r.jpg' 
    tensor, oringinal_image = image_to_tensor(image_path)
    # axes[0,0].imshow(oringinal_image)
    # axes[0,0].set_title('')
    
    # tensor = generate_graph_tensor(H=40, W=40, graph="Circle", R=10)

    r = RetinaModel()
    
    axes[0,0].imshow(tensor_to_image(tensor))
    
    input_center0 = (920, 890)
    input_center1 = (950, 920)
    
    centers_x, centers_y = smooth_moving(input_center0, input_center1, 1)
   
    window = 40
    windowx = 190
    windowy = 310
     
    for x, y in zip(centers_x, centers_y):
        print(x, y)
        
        out, grad, h, diff, velocity, cropped = r(tensor, x, y)
        
    
    
    min_vals = torch.min(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 0]) 
    max_vals = torch.max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 0])
    maxedge = torch.max((-1)*min_vals, max_vals)
    print(maxedge)
    
    min_vals = torch.min(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 1]) 
    max_vals = torch.max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 1])
    maxedge0 = torch.max((-1)*min_vals, max_vals)
    maxedge = torch.max(maxedge, maxedge0)
    print(maxedge)
    
    min_vals = torch.min(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 2]) 
    max_vals = torch.max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 2])
    maxedge0 = torch.max((-1)*min_vals, max_vals)
    maxedge = torch.max(maxedge, maxedge0)
    print(maxedge)
    
    min_vals = torch.min(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 3]) 
    max_vals = torch.max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 3])
    maxedge0 = torch.max((-1)*min_vals, max_vals)
    maxedge = torch.max(maxedge, maxedge0)
    print(maxedge)
    
    # axes[0,0].imshow(tensor_to_image(tensor))
    # axes[0,1].imshow(tensor_to_image(out))
    # axes[0,2].imshow(tensor_to_image(diff[:, :, windowy-window : windowy+window, windowx-window : windowx+window]))
    # axes[1,0].imshow(tensor_to_image(out[:, :, windowy-window : windowy+window, windowx-window : windowx+window]))
    # axes[1,1].imshow(tensor_to_image(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 1].unsqueeze(0)))
    # axes[1,2].imshow(edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 0].unsqueeze(0), maxedge))
    # axes[2,0].imshow(edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 1].unsqueeze(0), maxedge))
    # axes[2,1].imshow(edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 2].unsqueeze(0), maxedge))
    # axes[2,2].imshow(edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 3].unsqueeze(0), maxedge))
    
    tensor_to_image(out, save=True, filename="retina.png")
    tensor_to_image(out[:, :, windowy-window : windowy+window, windowx-window : windowx+window], save=True, filename="retina0.png")
    tensor_to_image(cropped, save=True, filename="cropped.png")
    tensor_to_image(h[:, :, windowy-window : windowy+window, windowx-window : windowx+window], save=True, filename="h.png")
    tensor_to_image(diff[:, :, windowy-window : windowy+window, windowx-window : windowx+window], save=True, filename="diff.png")
    
    edge_to_image(grad[:, :, windowy-window : windowy+window, windowx-window : windowx+window, 0], save=True, filename="grad0.png")
    edge_to_image(grad[:, :, windowy-window : windowy+window, windowx-window : windowx+window, 1], save=True, filename="grad1.png")
    
    edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 0].unsqueeze(0), maxedge, save=True, filename="v0.png")
    edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 1].unsqueeze(0), maxedge, save=True, filename="v1.png")
    edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 2].unsqueeze(0), maxedge, save=True, filename="v2.png")
    edge_to_image_with_max(velocity[:, 0, windowy-window : windowy+window, windowx-window : windowx+window, 3].unsqueeze(0), maxedge, save=True, filename="v3.png")
    
    # print(velocity[0,0,windowy,windowx,0])
    # print(velocity[0,0,windowy,windowx,1])
    # print(velocity[0,0, windowy-window : windowy+window, windowx-window : windowx+window,0])
    # print(velocity[0,0, windowy-window : windowy+window, windowx-window : windowx+window,1])

    # maxv, index = torch.max(velocity[..., 1])
    # print(index.item())
    # maxv, index = torch.max(velocity[..., 2])
    # print(index.item())
    # maxv, index = torch.max(velocity[..., 3])
    # print(index.item())
    plt.show()
    
    # model = RetinaModel()
    # output, _, _, _ = model(tensor, 960, 640)
    # axes[0,1].imshow(tensor_to_image(output))
    # axes[0,1].set_title('')
    
    # output, grad, h, diff, flowvelocity, cropped= model(tensor, -1, -1)
    # axes[1,0].imshow(tensor_to_image(output))

    
    # axes[0,1].imshow(tensor_to_image(cropped))
    # axes[0,2].imshow(tensor_to_image(output))
    
    # plt.show()
    # axes[1,1].imshow(edge_to_image(velocity[3]))

    # axes[0,1].imshow()
    # axes[0,1].set_title('')
    # axes[1,0].imshow()
    # axes[1,0].set_title('')
    # axes[1,1].imshow()
    # axes[1,1].set_title('')

    
    
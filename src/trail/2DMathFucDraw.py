import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
tau = 0.5
# 定义二元函数
def simple_function(x, y):
    return np.exp(tau * (x ** 2 - y ** 2) / np.sqrt(x**2 + y**2) / (x**2 + y**2)) 
 
# 生成坐标点
x = np.linspace(-5, 5, 11)
y = np.linspace(-5, 5, 11)
X, Y = np.meshgrid(x, y)
 
# 计算函数值
Z = simple_function(X, Y)
print(x)
 
# 绘制图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('')
plt.show()
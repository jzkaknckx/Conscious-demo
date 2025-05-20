import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist

figure1=plt.figure("",figsize=(8,8))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(figure1, 111)
#将绘图区对象添加到画布中
figure1.add_axes(ax)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("->", size = 1.0)
#设置x、y轴上刻度显示方向
ax.axis["x"].set_axis_direction("bottom")
ax.axis["y"].set_axis_direction("right")

tau = 0.5

# x0 = np.linspace(0,128,num=128)
x = np.linspace(0,180,num=180)

# y0 = x0
# y1 = 0.01*(x-256)**3/6+x
y2 = np.exp(tau / 2 * np.sin(2 * x * (np.pi / 180)))

# plt.plot(x0,y0,"b",label="")
# plt.plot(x,y1,"b",label="")
plt.plot(x,y2,"b",label="")


plt.legend()
plt.show()


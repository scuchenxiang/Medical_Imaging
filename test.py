import scipy.io as sio
import matplotlib.pyplot as plt
a=sio.loadmat("testres.mat")
b=a["predimg"]
c=b[0,0,0,:,:]

plt.figure("Image") # 图像窗口名称
plt.imshow(c,cmap='gray')
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()
d=1;
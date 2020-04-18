import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = np.loadtxt('data.txt')
new_data = data.reshape(256,256)
#kde = stats.gaussian_kde(new_data)
print(new_data)
x = np.linspace(0,255,256)
y = np.linspace(0,255,256)
X,Y = np.meshgrid(x,y)
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, new_data, 255, cmap='viridis')
#plt.plot(new_data)
plt.show()

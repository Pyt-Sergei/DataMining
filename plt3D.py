from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data2.txt',sep=' ', header=None)
xx1 = list(data[0])
xx2 = list(data[1])
yy = list(data[2])
m = len(data)

max_x1, min_x1 = max(data[0]), min(data[0])
max_x2, min_x2 = max(data[1]), min(data[1])

rx1 = sum(data[0])/m
rx2 = sum(data[1])/m

xx1 = list(map(lambda x: (x-rx1)/(max_x1-min_x1), xx1 ))
xx2 = list(map(lambda x: (x-rx2)/(max_x2-min_x2), xx2 ))

max_x1, min_x1 = max(xx1), min(xx1)
max_x2, min_x2 = max(xx2), min(xx2)

c0 = 340412.6595744676
c1 =  504777.38444559765
c2 =  -34951.40871912097

fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
ax1.scatter(xx1, xx2, yy)

x, y = np.meshgrid(np.arange(min_x1,max_x1,0.001), np.arange(min_x2,max_x2,0.001))
z = x*c1 + c2*y + c0

ax1.plot_surface(x, y, z, alpha=0.2)
plt.show()

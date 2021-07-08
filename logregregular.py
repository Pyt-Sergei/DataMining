import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('data4train.txt',sep=' ', header=None)
xx0 = np.array([1 for i in range(len(data))])
xx1 = np.array(data[0])
xx2 = np.array(data[1])
yy = np.transpose( [np.array(data[2])] )
m = len(data)


def h(x, c):
    return 1/(1 + np.exp(-(np.mat(c) * np.mat(x))))

def lg(h, y):
    return -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + lmb/(2*m)*np.sum((lmb1 * c) ** 2)

def j0(h, x, y):
    return ((h - y.transpose()) * x.transpose()) + lmb * (lmb1 * c)


eps = 1e-10
c = np.zeros((1, 28))
alpha = 1
lmb = 1
lmb1 = np.ones((1, 28))
lmb1[0][0] = 0

M = []
[M.append(xx1**i * xx2**j) for i in range(0,7) for j in range(0,7-i) ]

# вычисляем начальное значение
J0 = lg(h(M,c), yy)
J = J0

while True:
    J0 = J
    c = c - alpha * j0(h(M, c), np.array(M), yy) / m
    c = np.array(c)[0]
    J = lg(h(M,c), yy)

    if abs(J - J0) <= eps: break

print(c)

# plots
Z = 0
k = 0
delta = 1e-2
y0 = np.array(data[2])

x = np.arange(np.min(xx1), np.max(xx1), delta)
y = np.arange(np.min(xx2),np.max(xx2), delta)
X, Y = np.meshgrid(x, y)

for i in range(0, 7):
    for j in range(0, 7-i):
        Z += ((X**i * Y**j) * c[k])
        k += 1

plt.contour(X, Y, Z, levels=[0], colors='black')
plt.scatter(xx1[y0==0], xx2[y0==0], color='r', marker = '^')
plt.scatter(xx1[y0==1], xx2[y0==1], color='b', marker='^')
plt.show()
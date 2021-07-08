import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data3.txt',sep=' ', header=None)
xx0 = [1 for i in range(len(data))]
xx1 = list(data[0])
xx2 = list(data[1])
yy = list(data[2])
m = len(data)

def h(x1,x2):
    return 1/(1+np.exp(-(c0 + c1*x1 + c2*x2)))

def lg(x1,x2,y):
    return -y*np.log(h(x1,x2)) - (1-y)*np.log(1-h(x1,x2))

def dJ0(x1,x2,y):
    return h(x1,x2) - y

def dJ1(x1,x2,y):
    return x1*( h(x1,x2) - y )

def dJ2(x1,x2,y):
    return x2*( h(x1,x2) - y )

max_x1, min_x1 = max(xx1), min(xx1)
max_x2, min_x2 = max(xx2), min(xx2)
var1 = max_x1 - min_x1
var2 = max_x2 - min_x2

rx1 = sum(xx1)/m
rx2 = sum(xx2)/m

xx1 = list(map(lambda x: (x-rx1)/(max_x1-min_x1), xx1 ))
xx2 = list(map(lambda x: (x-rx2)/(max_x2-min_x2), xx2 ))

alpha = 1e-10 # 1e-10
eps = 1e-10    # 1e-10
c0, c1, c2 = 0, 0.75, 0.6

J0, dj0, dj1, dj2 = 0,0,0,0
for i in range(m):
    J0 += lg(xx1[i], xx2[i], yy[i])
    dj0 += dJ0(xx1[i], xx2[i], yy[i])
    dj1 += dJ1(xx1[i], xx2[i], yy[i])
    dj2 += dJ2(xx1[i], xx2[i], yy[i])

J0 /= m
dj0 /= m
dj1 /= m
dj2 /= m

c0 -= alpha * dj0
c1 -= alpha * dj1
c2 -= alpha * dj2

J = J0
while True:
    J0 = J
    J, dj0, dj1, dj2 = 0, 0, 0, 0
    for i in range(m):
        J += lg(xx1[i], xx2[i], yy[i])
        dj0 += dJ0(xx1[i], xx2[i], yy[i])
        dj1 += dJ1(xx1[i], xx2[i], yy[i])
        dj2 += dJ2(xx1[i], xx2[i], yy[i])

    J /= m
    dj0 /= m
    dj1 /= m
    dj2 /= m

    c0 -= alpha * dj0
    c1 -= alpha * dj1
    c2 -= alpha * dj2

    if abs(J-J0) <= eps: break

admitted_x,admitted_y, not_admitted_x, not_admitted_y = [],[],[],[]
for i in range(m):
    if h(xx1[i],xx2[i]) >= 0.5:
        admitted_x.append(xx1[i])
        admitted_y.append(xx2[i])
    else :
        not_admitted_x.append(xx1[i])
        not_admitted_y.append(xx2[i])



plt.scatter(admitted_x,admitted_y, s=30,c='b', label='Admitted')
plt.scatter(not_admitted_x,not_admitted_y, marker='x', s=30, c='r', label='Notadmitted')
x1_values = [ -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4 ]
x2_values = [ -(c0+c1*a)/c2 for a in x1_values ]
plt.plot(x1_values, x2_values, label='Decision Boundary')

plt.legend()
plt.show()

print('c0 = ', c0)
print('c1 = ', c1)
print('c2 = ', c2)


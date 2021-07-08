import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data2.txt',sep=' ', header=None)
xx0 = [1 for i in range(len(data))]
xx1 = list(data[0])
xx2 = list(data[1])
yy = list(data[2])
m = len(data)


max_x1, min_x1 = max(data[0]), min(data[0])
max_x2, min_x2 = max(data[1]), min(data[1])
var1 = max_x1 - min_x1
var2 = max_x2 - min_x2

rx1 = sum(data[0])/m
rx2 = sum(data[1])/m

plt.subplot(2, 2, 1)
plt.hist(xx1)
plt.subplot(2, 2, 2)
plt.hist(xx2)

xx1 = list(map(lambda x: (x-rx1)/(max_x1-min_x1), xx1 ))
xx2 = list(map(lambda x: (x-rx2)/(max_x2-min_x2), xx2 ))

plt.subplot(2, 2, 3)
plt.hist(xx1, color="red")
plt.subplot(2, 2, 4)
plt.hist(xx2, color="red")
plt.show()

def h(x1,x2):
    return c0 + c1*x1 + c2*x2

def MSE2(x1,x2,y):
    return ( h(x1,x2) - y )**2

def dJ0(x1,x2,y):
    return h(x1,x2) - y

def dJ1(x1,x2,y):
    return x1*( h(x1,x2) - y )

def dJ2(x1,x2,y):
    return x2*( h(x1,x2) - y )

eps = 1e-5
alpha = 0.05
c0, c1, c2 = 0,0,0

# вычисляется начальное значение
J0, dj0, dj1, dj2 = 0,0,0,0
for i in range(m):
    J0 += MSE2(xx1[i], xx2[i], yy[i])
    dj0 += dJ0(xx1[i], xx2[i], yy[i])
    dj1 += dJ1(xx1[i], xx2[i], yy[i])
    dj2 += dJ2(xx1[i], xx2[i], yy[i])

J0 /= 2*m
dj0 /= m
dj1 /= m
dj2 /= m

c0 -= alpha * dj0
c1 -= alpha * dj1
c2 -= alpha * dj2

J = J0
while True:
    J0 = J
    J, dj0, dj1, dj2 = 0,0,0,0
    for i in range(m):
        J += MSE2(xx1[i], xx2[i], yy[i])
        dj0 += dJ0(xx1[i], xx2[i], yy[i])
        dj1 += dJ1(xx1[i], xx2[i], yy[i])
        dj2 += dJ2(xx1[i], xx2[i], yy[i])

    J /= 2*m
    dj0 /= m
    dj1 /= m
    dj2 /= m

    c0 -= alpha * dj0
    c1 -= alpha * dj1
    c2 -= alpha * dj2

    #print(c1,c2)
    if abs(J-J0) <= eps : break
# Программа работает на моем компьютере за 5сек
print('c0 = ', c0)
print('c1 = ', c1)
print('c2 = ', c2)
print('прогноз при х1 = 1500, x2 = 3 ',h((1500-rx1)/var1, (3-rx2)/var2) )



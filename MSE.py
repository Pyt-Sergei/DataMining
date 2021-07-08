from matplotlib import pyplot as plt
import matplotlib.animation as animation

# collecting data into lists from msedatatxt file
data = []
xx = []
yy = []
with open('data1.txt', 'r') as f:
    for line in f:
        s = line.rstrip()
        x,y = '',''
        for i in s:
            if i == ' ':
                x = s[0:s.find(i)]
                y = s[s.find(i)+1: ]
                break
        data.append([float(x),float(y)])
        xx.append(float(x))
        yy.append(float(y))

def h(x):
    return c0+c1*x

def MSE(x,y):
    c = ( h(x) - y )**2
    return c

def dJ0(x,y):
    return ( h(x) - y )

def dJ1(x,y):
    return x*( h(x) - y )


alpha = 1e-2
eps = 1e-10
m = len(data)
c0, c1 = 0,6

min_x, max_x = min(xx), max(xx)
y_min, y_max = [], []

# вычисляется начальное значение
J0, dj0, dj1 = 0,0,0
for pair in data:
    J0 += MSE(pair[0], pair[1])
    dj0 += dJ0(pair[0], pair[1])
    dj1 += dJ1(pair[0], pair[1])

    J0 /= 2*m
    dj0 /= m
    dj1 /= m

c0 -= alpha * dj0
c1 -= alpha * dj1

y_min.append(h(min_x))
y_max.append(h(max_x))

J = J0
while True:
    J0 = J
    J, dj0, dj1 = 0,0,0
    for pair in data:
        J += MSE(pair[0],pair[1])
        dj0 += dJ0(pair[0],pair[1])
        dj1 += dJ1(pair[0],pair[1])

    J /= 2*m
    dj0 /= m
    dj1 /= m

    c0 -= alpha * dj0
    c1 -= alpha * dj1

    y_min.append(h(min_x))
    y_max.append(h(max_x))

    #print(c0,c1)
    if abs(J-J0) <= eps : break

print('c0 = ', c0)
print('c1 = ', c1)
print('прогноз при х = 10 ',h(10))

fig = plt.figure()
ax = plt.axes(xlim=(4, 25), ylim=(-3, 25))
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data([min_x, max_x], [y_min[i], y_max[i]])
    return line,
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=10, blit=True)
plt.plot(xx,yy,'or', markersize=3)
plt.show()




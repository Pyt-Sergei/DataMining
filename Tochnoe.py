import numpy as np

data = []
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

m = 97
sum_x = 0
sum_y = 0
sum_xx = 0
sum_xy = 0

for pair in data:
    sum_x += pair[0]
    sum_y += pair[1]
    sum_xx += pair[0]**2
    sum_xy += pair[0]*pair[1]

A = np.array([[m,sum_x],[sum_x,sum_xx]])
b = np.array([[sum_y],[sum_xy]])
etta = np.linalg.solve(A,b)

# Проверка корректности
if np.allclose(np.dot(A, etta), b):
    print(etta)





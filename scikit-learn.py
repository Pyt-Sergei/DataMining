import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data2.txt',sep=' ', header=None)
x = data.values[:,:-1]
y = data.values[:,-1:].ravel()

scaler = StandardScaler()
scaler.fit(x)
scaled_x = scaler.transform(x)

reg = LinearRegression().fit(scaled_x,y)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(scaler.transform([[1500, 3]]))[0])


import pandas as pd
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as Grid
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# collecting data
data = pd.read_csv('spambase.data', sep=',', header=None)
x = data.values[:,:-1]
y = data.values[:,-1:].ravel()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = SVC()
model.fit(x_train, y_train)
print('Вычисления до масштабирования ', accuracy_score(model.predict(x_test), y_test))

scaler = Scaler()
scaler.fit(x_train)

x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

model.fit(x_train,y_train)
print('Вычисления после масштабирования ', accuracy_score(model.predict(x_test), y_test))

params = \
    {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
     }
grid = Grid(model, params)
grid.fit(x_train, y_train)
print('Параметр с наилучшим результатом ', grid.best_params_)
print(grid.score(x_test, y_test))

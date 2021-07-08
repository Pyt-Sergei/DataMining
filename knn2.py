import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV as Grid

data = pd.read_csv('spambase.data', sep=',', header=None)
x = data.values[:,:-1]
y = data.values[:,-1:].ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn = KNN()

params = \
  {
  'n_neighbors': range(7, 11, 2),
  'weights': ['uniform', 'distance'],
  'p': [1, 2]
  }

grid = Grid(knn, params)
grid.fit(x_train, y_train)

print(grid.best_params_)
print(accuracy_score(grid.predict(x_test), y_test))
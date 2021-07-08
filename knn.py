import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score


data = pd.read_csv('spambase.data', sep=',', header=None)
x = data.values[:,:-1]
y = data.values[:,-1:].ravel()

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

knn = KNN()
knn.fit(x_train, y_train)
print('Вычисления до масштабирования ', accuracy_score(knn.predict(x_test), y_test))

scaler = SS()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNN()
knn.fit(x_train, y_train)
print('Вычисления после масштабирования ', accuracy_score(knn.predict(x_test), y_test))

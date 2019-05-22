
from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
iris_dataset = load_iris()

print(iris_dataset.keys())
print(iris_dataset['DESCR'][:198])
print('target names:', iris_dataset['target_names'])
print('feature names:', iris_dataset['feature_names'])
print('data types:', type(iris_dataset['data']))
print('data size:', iris_dataset['data'].shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state=0)

print('size of X_train:', X_train.shape)
print('size of y_train:', y_train.shape)


'''
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(
    iris_dataframe, c=y_train, figsize=(6, 6),
    marker='o',hist_kwds={'bins': 20}, s=30, alpha=.8, cmap=mglearn.cm3)
plt.show()
'''

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n", y_pred)
print("정확도: {:.2f}".format(np.mean(y_pred==y_test)))

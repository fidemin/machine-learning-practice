from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import mglearn
import numpy as np

X, y = make_blobs(random_state=42)
linear_svm = LinearSVC().fit(X, y)
print('coef size:', linear_svm.coef_.shape)
print('intercept size: ', linear_svm.intercept_.shape)


mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept)  / coef[1], c=color)

plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend(
    ['class 0', 'class 1', 'class 2',
     'class 0 line', 'class 1 line', 'classs 2 line'],
    loc=(1.01, 0.3))

plt.show()

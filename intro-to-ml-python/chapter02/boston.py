import mglearn
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print('lr train set score: {:.2f}'.format(lr.score(X_train, y_train)))
print('lr test set score: {:.2f}'.format(lr.score(X_test, y_test)))

ridge = Ridge().fit(X_train, y_train)
print('ridge train set score: {:.2f}'.format(ridge.score(X_train, y_train)))
print('ridge test set score: {:.2f}'.format(ridge.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print('ridge01 train set score: {:.2f}'.format(ridge01.score(X_train, y_train)))
print('ridge01 test set score: {:.2f}'.format(ridge01.score(X_test, y_test)))

lasso = Lasso().fit(X_train, y_train)
print('lasso train set score: {:.2f}'.format(lasso.score(X_train, y_train)))
print('lasso test set score: {:.2f}'.format(lasso.score(X_test, y_test)))
print('num of feature used:', np.sum(lasso.coef_ != 0))

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print('lasso001 train set score: {:.2f}'.format(lasso001.score(X_train, y_train)))
print('lasso001 test set score: {:.2f}'.format(lasso001.score(X_test, y_test)))
print('num of feature used:', np.sum(lasso001.coef_ != 0))


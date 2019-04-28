import numpy as np
from sklearn.metrics import mean_squared_error

from data_preprocess import R

def compute_ALS(R, n_iter, lambda_, k):
    m, n = R.shape

    X = 5 * np.random.rand(m, k)
    Y = 5 * np.random.rand(k, n)

    errors = []

    for i in range(0, n_iter):
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k), np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k), np.dot(X.T, R))

        errors.append(mean_squared_error(R, np.dot(X, Y)))
        print(mean_squared_error(R, np.dot(X, Y)))

    R_hat = np.dot(X, Y)
    print('Error of rated movies: %f ' % mean_squared_error(R, np.dot(X, Y)))
    print(R_hat, errors)

def compute_wALS(R, n_iter, lambda_, k):
    m, n = R.shape

    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)

    W = R > 0.0
    W[W == True] = 1
    W[W == False] = 0

    weighted_errors = []

    for j in range(n_iter):
        for u, Wu in enumerate(W):
            X[u, :] = np.linalg.solve(
                np.dot(Y, np.dot(np.diag(Wu), Y.T)) + \
                lambda_ * np.eye(k), np.dot(Y, np.dot(np.diag(Wu), R[u,:].T)).T
            )

        for i, Wi in enumerate(W.T):
            Y[:, i] = np.linalg.solve(
                np.dot(X.T, np.dot(np.diag(Wi), X)) + \
                lambda_ * np.eye(k), np.dot(X.T, np.dot(np.diag(Wi), R[:,i]))
            )

        weighted_errors.append(mean_squared_error(R, np.dot(X, Y), sample_weight=W))
        print('iteration %d is completed' % j)

    R_hat = np.dot(X, Y)
    print('Error of rated movies: %f ' % mean_squared_error(R, np.dot(X, Y)))
    print(R_hat, weighted_errors)





if __name__ == "__main__":
    m, n = R.shape

    k = 100 # 요인 행렬 크기. 100을 보통 사용한다.
    #compute_ALS(R, 20, 0.1, k)
    compute_wALS(R, 10, 0.1, k)

import numpy as np
from sklearn.metrics import mean_squared_error

from data_preprocess import R, W, train_test_split, movie_info_li

def get_test_mse(true, pred):
    pred = pred[true.nonzero()].flatten()
    true = true[true.nonzero()].flatten()
    return mean_squared_error(true, pred)


def compute_ALS(R, test, n_iter, lambda_, k):
    m, n = R.shape

    X = 5 * np.random.rand(m, k)
    Y = 5 * np.random.rand(k, n)

    errors = []

    for i in range(0, n_iter):
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k), np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k), np.dot(X.T, R))
        #errors.append(mean_squared_error(R, np.dot(X, Y)))
        errors.append(get_test_mse(np.dot(X, Y), test))

    R_hat = np.dot(X, Y)
    #print('Error of rated movies: %f ' % mean_squared_error(R, np.dot(X, Y)))
    print('Error of rated movies: %f ' % get_test_mse(np.dot(X,Y), test))
    return R_hat, errors

def compute_wALS(R, W, n_iter, lambda_, k):
    m, n = R.shape

    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)

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


def compute_GD(R, n_iter, lambda_, learning_rate, k):
    m, n = R.shape
    errors = []

    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)

    for ii in range(n_iter):
        for u in range(m):
            for i in range(n):
                if R[u, i] > 0:
                    e_ui = R[u, i] - np.dot(X[u, :], Y[:, i])
                    X[u, :] += learning_rate * (e_ui * Y[:, i].T - lambda_ * X[u, :])
                    Y[:, i] += learning_rate * (e_ui * X[u, :].T - lambda_ * Y[:, i])


        errors.append(mean_squared_error(R, np.dot(X, Y)))
        if ii % 10 == 0:
            print('iteration %d is completed' % ii)

    R_hat = np.dot(X, Y)
    print('Error of rated movies: %d' % (mean_squared_error(R, R_hat)))
    return (R_hat, errors)


def recomment_by_user(user, R_hat, W):
    R_hat -= np.min(R_hat)
    R_hat *= float(5) / np.max(R_hat)
    user_index = user - 1
    user_seen_movies = sorted(list(enumerate(R_hat[user_index])), key = lambda x:x[1], reverse=True)
    print(user_seen_movies[:10])
    recommended = 1
    print("------recommendation for user %d ------" % user)
    for movie_info in user_seen_movies:
        if W[user_index][movie_info[0]]==0:
            movie_title = movie_info_li[movie_info[0]]
            movie_score = movie_info[1]
            print("rank %d recommendation:%s(%.3f)" % (recommended, movie_title[0], movie_score))
            recommended += 1
            if recommended == 6:
                break


if __name__ == "__main__":
    k = 100 # 요인 행렬 크기. 100을 보통 사용한다.
    #compute_ALS(R, 20, 0.1, k)
    #compute_wALS(R,W, 10, 0.1, k)
    #R_hat, errors = compute_GD(R, 40, 1, 0.001, k)
    train, test = train_test_split(R, 10)
    #R_hat, train_errors = compute_ALS(train, train, 20, 0.1, 100)
    #print(train_errors)
    R_hat, train_errors = compute_ALS(train, test, 20, 500, 100)
    recomment_by_user(1, R_hat, W)

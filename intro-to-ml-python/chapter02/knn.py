import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def show_graph(X, y):
    # 기본 scatter 그래프 그리기
    mglearn.discrete_scatter(X[:,0], X[:,1], y)

    #mglearn.plots.plot_knn_classification(n_neighbors=3)
    plt.legend(['class 0', 'class 1'], loc=4)
    plt.xlabel('first attribute')
    plt.ylabel('second attribute')
    plt.show()


def show_by_n_neighbors(X, y, n_neighbors_lst):
    fig, axes = plt.subplots(1, len(n_neighbors_lst), figsize=(10, 3))

    for n_neighbors, ax in zip(n_neighbors_lst, axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title('{} 이웃'.format(n_neighbors))
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')

    axes[0].legend(loc=3)
    plt.show()


if __name__ == '__main__':
    X, y = mglearn.datasets.make_forge()
    #show_graph(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    print('테스트 세트 예측:', clf.predict(X_test))
    print('테스트 세트 정확도: {:.2f}'.format(clf.score(X_test, y_test)))

    show_by_n_neighbors(X, y, [1, 3, 6, 9])

    
    

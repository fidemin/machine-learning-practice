
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('feature importance')
    plt.ylabel('feature')
    plt.ylim(-1, n_features)
    plt.show()

if __name__ == '__main__':
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)
    print('train set precision: {:.3f}'.format(tree.score(X_train, y_train)))
    print('test set precesion: {:.3f}'.format(tree.score(X_test, y_test)))

    export_graphviz(tree, out_file='tree.dot', class_names=['malignant', 'benign'],
                    feature_names=cancer.feature_names, impurity=False, filled=True)

    print('feature importance: \n', tree.feature_importances_)
    plot_feature_importances_cancer(tree)



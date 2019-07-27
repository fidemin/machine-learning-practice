
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import RandomForestClassifier
import mglearn

cancer = load_breast_cancer()
X = cancer.data[:,:2]
y = cancer.target


#X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

forest = RandomForestClassifier(n_estimators=5, random_state=0)
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(10, 5))
for i , (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title('tree {}'.format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(
    forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title('random forest')
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()

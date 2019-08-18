from sklearn.svm import SVC
import mglearn
from matplotlib import pylab as plt

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)

mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
sv = svm.support_vectors_
print("suport vectors: ", sv)
sv_labels = svm.dual_coef_.ravel() > 0
print("support vector labels: ", sv_labels)

mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

plt.show()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

logreg001 = LogisticRegression(C=0.001).fit(X_train, y_train)
logreg = LogisticRegression(C=1).fit(X_train, y_train)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print('train score: {:.3f}'.format(logreg.score(X_train, y_train)))
print('test score: {:.3f}'.format(logreg.score(X_test, y_test)))

plt.plot(logreg100.coef_.T, '^', label='c=100')
plt.plot(logreg.coef_.T, 'o', label='c=1')
plt.plot(logreg001.coef_.T, 'v', label='c=0.001')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel('feature')
plt.ylabel('w size')
plt.legend()
plt.show()

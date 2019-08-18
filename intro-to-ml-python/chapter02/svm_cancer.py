from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

import matplotlib.pyplot as plt


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)


print("train: {:.2f}".format(svc.score(X_train, y_train)))
print("test: {:.2f}".format(svc.score(X_test, y_test)))


'''
plt.boxplot(X_train, manage_xticks=False)
plt.yscale('symlog')
plt.xlabel('features')
plt.ylabel('feature size')
plt.show()
'''

min_on_training = X_train.min(axis=0)
max_on_training = X_train.max(axis=0)
range_on_training = max_on_training - min_on_training

X_train_scaled = (X_train - min_on_training) / range_on_training
print('feature min\n', X_train_scaled.min(axis=0))
print('feature max\n', X_train_scaled.max(axis=0))

X_test_scaled = (X_test - min_on_training) / range_on_training


svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)


print("train with scaled: {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("test with scaled: {:.2f}".format(svc.score(X_test_scaled, y_test)))

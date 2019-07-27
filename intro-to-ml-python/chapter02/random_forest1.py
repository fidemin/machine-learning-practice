from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import RandomForestClassifier


cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

print('train set correctness: {:.3f}'.format(forest.score(X_train, y_train)))
print('test set correctness: {:.3f}'.format(forest.score(X_test, y_test)))


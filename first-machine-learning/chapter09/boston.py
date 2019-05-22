
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()
#print(boston.data[2].reshape(1, -1))

lr = LinearRegression()
lr.fit(boston.data, boston.target)

#print(lr.predict(boston.data[2].reshape(1, -1)))

[print(x) for x in zip(boston.feature_names, lr.coef_)]

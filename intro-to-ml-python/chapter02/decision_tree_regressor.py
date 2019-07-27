import os
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, 'ram_price.csv'))

    data_train = ram_prices[ram_prices.date < 2000]
    data_test = ram_prices[ram_prices.date >= 2000]

    X_train = data_train.date[:, np.newaxis]
    # 가격을 로그 스케일로 변경한다. => linear 관계를 만들기 위해서
    y_train = np.log(data_train.price)

    tree = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)

    X_all = ram_prices.date[:, np.newaxis]

    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)

    # log scale을 원래 값으로 되돌린다. (그래프 만들때, semilogy를 쓰므로)
    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)

    plt.yticks(fontname = 'Arial')
    plt.semilogy(data_train.date, data_train.price, label='training data')
    plt.semilogy(data_test.date, data_test.price, label='test data')
    plt.semilogy(ram_prices.date, price_tree, label='tree expectation')
    plt.semilogy(ram_prices.date, price_lr, label='lr expectation')
    plt.xlabel('year')
    plt.ylabel('prict ($/MByte)')
    plt.legend()

    plt.show()

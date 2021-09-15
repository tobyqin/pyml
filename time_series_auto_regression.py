# encoding: utf-8
# @Time: 2021/9/7 20:36
# @Author: zou wei
# @module: 8.time_series_auto_regression.py
# @Contact: visio@163.com
# @Software: PyCharm
# @User: 86138

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def ts_model(t, future_times=30, window=5, ratio=0.5):
    N = len(t)
    print(N)
    # window = 5
    x = np.empty(shape=(N-window, window))
    y = np.empty(shape=(N-window,))
    j, step = 0, 1
    for i in range(N-window):
        x[i, :] = t[j:j+window]
        y[i] = t[j+window]
        j += step
    print(x)
    print(y)
    model_rf = RandomForestRegressor(n_estimators=50, max_depth=6)
    model_lineregression = LinearRegression()
    model_rf.fit(x, y)
    model_lineregression.fit(x, y)
    y_pred = ratio*model_lineregression.predict(x) + (1-ratio)*model_rf.predict(x)

    # future_times = 30
    x_future = t[-window:]
    y_future = np.empty(shape=future_times)
    for i in range(future_times):
        y_rf = model_rf.predict(x_future.reshape(1, -1))
        y_linereg = model_lineregression.predict(x_future.reshape(1, -1))
        y_future[i] = (1-ratio)*y_rf + ratio*y_linereg
        x_future = np.concatenate((x_future[1:], [y_future[i]]), axis=0)
    return y_pred, y_future


if __name__ == '__main__':
    N = 1000
    x = np.random.uniform(-20, 20, N)
    x.sort()
    y = np.sin(x) + (x/15)**2 + np.random.normal(0, 0.1, N)

    window = 5
    y_pred, y_future = ts_model(y, 30, window, 0.9)
    plt.plot(np.arange(len(y)), y, label='Fact')
    plt.plot(np.arange(len(y))[window:], y_pred, label='Predict')
    plt.plot(np.arange(len(y_future))+len(y), y_future, label='Future')
    plt.legend(loc='best')
    print(y_future)
    plt.show()

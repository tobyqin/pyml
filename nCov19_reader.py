# encoding: utf-8
# @Time: 2021/7/25 13:29
# @Author: zou wei
# @module: nCov19_reader.py
# @Contact: visio@163.com
# @Software: PyCharm
# @User: 86138

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from PIL import Image
import matplotlib.pyplot as plt


def ts_average(a, half_window):
    b = np.empty_like(a)
    for i in range(len(a)):
        start = i - half_window if i > half_window else 0
        end = i + half_window + 1
        b[i] = (np.sum(a[start:end]) - a[i]) / (end-start-1)
    return b


def ts_smooth(a, b, cycle):
    for _ in range(cycle):
        err = np.abs(a - b)
        i = np.argmax(err)
        a[i] = b[i]


def ts_model(t, future_times=30):
    N = len(t)
    print(N)
    window = 5
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

    # future_times = 30
    x1 = t[-window:]
    y1 = np.empty(shape=future_times)
    for i in range(future_times):
        y_rf = model_rf.predict(x1.reshape(1, -1))
        y_linereg = model_lineregression.predict(x1.reshape(1, -1))
        y1[i] = 0.3*y_rf + 0.7*y_linereg
        x1 = np.concatenate((x1[1:], [y1[i]]), axis=0)
    return model_lineregression.predict(x), y1


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.min_row', 20)
    np.set_printoptions(edgeitems=10, suppress=True)
    data = pd.read_csv('alltime_China_2020.csv')
    print(data)
    t = data['today_confirm']
    print('data[\'today_confirm\'] = \n', t)
    t = t.values
    at = ts_average(t, 3)
    ts_smooth(t, at, 1)
    at = ts_average(t, 3)
    ts_smooth(t, at, 1)
    t_pred, y = ts_model(t, 30)
    print(y)
    plt.plot(t, 'r-', lw=2, label='Fact')
    plt.plot(t_pred, 'b--', t_pred, 'bo', lw=2, ms=3, label='Predict')
    plt.plot(np.arange(len(y))+len(t), y, ls='-', color='#8080FF', lw=2)
    plt.show()

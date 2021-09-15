# encoding: utf-8
# @Time: 2021/9/9 14:35
# @Author: zou wei
# @module: signal_clf.py
# @Contact: visio@163.com
# @Software: PyCharm
# @User: 86138

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import time
import joblib
import matplotlib.pyplot as plt


def create_class_weights():
    a = b = c = d = np.arange(1, 6)
    aa, bb, cc, dd = np.meshgrid(a, b, c, d)
    df = pd.DataFrame({0: aa.flatten(), 1: bb.flatten(), 2: cc.flatten(), 3: dd.flatten()})
    df = df.T
    return [dict(df[col]) for col in df.columns]


if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 4000)
    np.set_printoptions(suppress=True)
    data_path = './信号种类预测_8月'
    data_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    # print(data_train)
    data_test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    data_test['label'] = -1
    SAVE_IMAGE = False
    # print(data_test)
    # data = pd.concat((data_train, data_test), axis=0)
    # data.reset_index(drop=True, inplace=True)
    data = pd.concat((data_train, data_test), axis=0, ignore_index=True)
    print(data)

    # print(data_train['label'])
    result = pd.value_counts(data_train['label'], sort=False)   # normalize=True
    print(result)
    # plt.bar(result.index, result.values, width=0.4)
    # plt.show()
    description = data.describe()
    print(description)
    # for col in data.columns:
    #     print(col)
    #     t = data[col]
    #     plt.plot(t, '.', ms=1)
    #     plt.title(col)
    #     plt.savefig(os.path.join(data_path, 'Features', col+'.png'))
    #     plt.close()
    # description = data.drop(labels='label', axis=1).describe()
    # print(description)
    # t = np.percentile(data['3'], q=99.99)
    # data.loc[data['3'] > t, '3'] = t
    # y, thr, _ = plt.hist(data['3'], bins=30, edgecolor='k')
    # print(thr)
    # print(thr.shape)
    # print(y)
    # print(y.shape)
    # plt.show()
    print('数据异常检验和校正：')
    mean = data.mean()
    std = data.std()
    for col in data.columns:
        if col == 'label':
            continue
        min_thr, min_thr2 = mean[col] - 5*std[col], mean[col] - 3*std[col]
        max_thr, max_thr2 = mean[col] + 5*std[col], mean[col] + 3*std[col]
        if SAVE_IMAGE:
            plt.figure(figsize=(9, 5))
            plt.subplot(121)
            plt.hist(data[col], bins=30, edgecolor='k')
            plt.title('COL: ' + col + ' Prime Data')
        sel = data[col] < min_thr
        s1 = np.sum(sel)
        data.loc[sel, col] = np.random.normal(loc=min_thr2, scale=std[col], size=s1)
        sel = data[col] > max_thr
        s2 = np.sum(sel)
        data.loc[sel, col] = np.random.normal(loc=max_thr2, scale=std[col], size=s2)
        print('\t', col, 'MIN:', s1, 'MAX:', s2)
        if SAVE_IMAGE:
            plt.subplot(122)
            plt.hist(data[col], bins=30, edgecolor='k')
            plt.title('MIN:' + str(s1) + ',MAX:' + str(s2) + ' Adjust Data')
            plt.savefig(os.path.join(data_path, 'Features_stc', col+'.png'))
            plt.close()

    # p = (description.loc['mean'] - description.loc['50%']) / description.loc['50%']
    # for col, value in zip(p.index, p.values):
    #     if value > 2.5:
    #         print('大异常：', col)
    #         t = np.percentile(data[col], 99.9)
    #         data.loc[data[col] > t, col] = t
    #     elif value < -2.5:
    #         print('小异常：', col)
    #         t = np.percentile(data[col], 0.1)
    #         data.loc[data[col] < t, col] = t
    # description = data.describe()
    # print(description)
    # # print(data)

    x = data.drop(labels='label', axis=1)
    x = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)
    y = data['label']
    sel = y == -1
    x_train = x[~sel]
    y_train = y[~sel]
    x_test = x[sel]
    # y_test = y[sel]
    # print('x_train = \n', x_train)
    # print('y_train = \n', y_train)
    # print('x_test = \n', x_test)
    # print('y_test = \n', y_test)

    # # 特征选择
    col_sel = data_train.columns
    feature_path = os.path.join(data_path, 'feature_selector.npy')
    if os.path.exists(feature_path):
        print('加载“特征重要度”数据')
        col_sel = np.load(feature_path, allow_pickle=True)
        print('\t重要特征共%d个，分别是：' % len(col_sel), col_sel)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=100, class_weight={0: 1, 1: 10, 2: 3, 3: 1})
        print('训练“特征重要度”模型...')
        t_start = time.time()
        model.fit(x_train, y_train)
        t_end = time.time()
        print('\t耗时：%.3f秒。' % (t_end - t_start))
        print('\t特征重要度：', model.feature_importances_)
        fi = pd.Series(data=model.feature_importances_, index=list(x_train.columns), name='特征重要度')
        fi.sort_values(ascending=False, inplace=True)
        fi.to_excel(os.path.join(data_path, '特征重要度.xls'))
        col_sel = fi[fi.values > 1/(2*205)].index
        print('\t重要特征共%d个，分别是：' % len(col_sel), col_sel)
        np.save(feature_path, np.array(col_sel))

    # 建模
    x_train = x_train[col_sel]
    x_test = x_test[col_sel]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=88)
    model_path = os.path.join(data_path, 'knn_kd.model')
    if os.path.exists(model_path):
        print('加载模型...')
        model = joblib.load(model_path)
    else:
        # lr = LogisticRegression(multi_class='ovr', class_weight={0: 2, 1: 5, 2: 3, 3: 3})
        # model = LogisticRegression(multi_class='multinomial', C=10, penalty='elasticnet', l1_ratio=0.5, solver='saga', class_weight={0: 2, 1: 6, 2: 3, 3: 3})
        # model = BaggingClassifier(sm, n_estimators=10, max_samples=1.0, max_features=1.0)
        # cw_list = create_class_weights()
        # print(cw_list)
        # dt = DecisionTreeClassifier(max_depth=9, class_weight={0: 2, 1: 5, 2: 3, 3: 3})  # 0: 3, 1: 2, 2: 4, 3: 3
        # rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight={0: 2, 1: 5, 2: 3, 3: 3})
        # model = GridSearchCV(rf, cv=3, param_grid={
        #     # 'class_weight': cw_list   # [{0: 1, 1: 5, 2: 3, 3: 3}, {0: 2, 1: 5, 2: 3, 3: 3}]
        #     'max_depth': np.arange(12, 16)
        # })
        # svc = SVC(kernel='rbf', gamma=1)
        # model = GridSearchCV(svc, cv=3, param_grid={
        #     'degree': np.arange(2, 6)
        # })
        # model = MLPClassifier(hidden_layer_sizes=(256, 64, 16))
        # model = GaussianNB()
        model = KNeighborsClassifier(n_neighbors=11, algorithm='ball_tree')
        print(time.asctime(), '训练模型...')   # time.localtime()
        t_start = time.time()
        model.fit(x_train, y_train)
        t_end = time.time()
        print('\t耗时：%.3f秒。' % (t_end - t_start))
        # print('最优参数：', model.best_params_)
        # model = model.best_estimator_
        joblib.dump(model, model_path)
        # print(model.estimators_)
    # print('截距：\n', model.intercept_)
    # print(model.intercept_.shape)
    # print('系数：\n', model.coef_)
    # print(model.coef_.shape)
    # print(model.coef_.dtype)
    print('开始样本预测...')
    y_train_pred = model.predict(x_train)
    print('训练集混淆矩阵：\n', confusion_matrix(y_train, y_train_pred))
    print('\t训练集正确率：', accuracy_score(y_train, y_train_pred))
    print('\t训练集精确度：', precision_score(y_train, y_train_pred, average='macro'))
    print('\t训练集召回率：', recall_score(y_train, y_train_pred, average='macro'))
    print('\t训练集F1 值：', f1_score(y_train, y_train_pred, average='macro'))

    y_val_pred = model.predict(x_val)
    print('验证集混淆矩阵：\n', confusion_matrix(y_val, y_val_pred))
    print('\t验证集正确率：', accuracy_score(y_val, y_val_pred))
    print('\t验证集精确度：', precision_score(y_val, y_val_pred, average='macro'))
    print('\t验证集召回率：', recall_score(y_val, y_val_pred, average='macro'))
    print('\t验证集F1 值：', f1_score(y_val, y_val_pred, average='macro'))

    y_test_pred = model.predict(x_test)
    result_path = os.path.join(data_path, 'result.csv')
    pd.Series(data=y_test_pred, name='label').to_csv(result_path, index=False, header=True)

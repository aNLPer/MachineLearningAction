import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter

# 数据集加载和预处理
iris = datasets.load_iris()
X, y = shuffle(iris.data, iris.target, random_state=3)
X = X.astype(np.float32)
offset = int(X.shape[0]*0.7)
x_train, y_train = X[:offset], y[:offset]
x_test, y_test = X[offset:], y[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

#定义距离度量函数
def L2_Distance(x, x_train):
    """
    计算L2距离
    :param x: 测试数据
    :param x_train: 训练数据
    :return: 测试数据与训练数据的L2距离矩阵
    """
    num_x = x.shape[0]
    num_train = x_train.shape[0]
    dists = np.zeros((num_x, num_train))

    M = np.dot(x, x_train.T)
    te = np.square(x).sum(axis=1)
    tr = np.square(x_train).sum(axis=1)
    dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
    return dists

#计算测试集和训练集的距离
dists = L2_Distance(x_test, x_train)
plt.imshow(dists)
plt.show()

#预测测试集
def pre_label(y_train, dists, k = 1):
    """
    预测测试集
    :param y_train: 训练数据的标签
    :param dists: 测试数据和训练数据的距离
    :param k: 超参数
    :return: 预测标签
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        t = np.argsort(dists[i, :])
        label = y_train[t].flatten()
        closest_k = label[0:k]
        c = Counter(closest_k)
        y_pred[i] = c.most_common(1)[0][0]
    return y_pred

# 计算acc
y_test_pred = pre_label(y_train, dists, k=1)
y_test_pred = y_test_pred.reshape((-1,1))
num_correct = np.sum(y_test_pred==y_test)
acc = num_correct/y_test.shape[0]
print("acc:%f"%(acc))







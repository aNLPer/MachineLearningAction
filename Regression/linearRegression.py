"""
线性回归
"""
import numpy as np
def linearReg(X, y, w, b):
    """
    线性回归模型主体
    :param X: 训练数据
    :param y: 标签
    :param w: 权重参数
    :param b: 偏置
    :return:
    """
    num_sample = X.shape[0]
    num_feature = X.shape[1]

    # 计算损失函数
    y_hat = np.dot(X, w)+b
    linear_loss = np.sum((y_hat-y)**2)/num_sample

    #求参数偏导数
    dw = 2*np.dot(X.T, (y_hat-y))/num_sample
    db = 2*np.sum(y_hat-y)/num_sample

    return linear_loss, dw, db

def paramter_init(dims):
    """
    参数初始化
    :param dims:数据维度
    :return: 标准正太分布初始化的参数
    """
    param = np.random.normal(size=(dims+1, 1))

    w = param[:len(param)-1]
    b = param[-1]
    return w, b

# 训练模型
def training(X, y, lr, epochs):
    """
    梯度下降算法训练线性模型
    :param X: 数据集
    :param y: 标签
    :param lr: 学习率
    :param epochs: 训练次数
    :return: 误差、w、b参数列表
    """
    w, b = paramter_init(X.shape[1])
    loss_list = []
    w_list = []
    b_list = []

    w_list.append(w)
    b_list.append(b)

    for epoch in range(epochs):
        # 计算损失函数和梯度
        loss, dw, db = linearReg(X,y,w,b)
        loss_list.append(loss)
        #更新参数
        w -= lr*dw
        b -= lr*db
        # 记录参数
        w_list.append(w)
        b_list.append(b)

        if (epoch+1)%100 == 0:
            print("loss:", loss)

    return loss_list, w_list, b_list


def predict(X, params_w, params_b):
    w = params_w[-1]
    b = params_b[-1]
    pred_y = np.dot(X, w)+b
    return pred_y

# 例子
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 加载数据集
diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target

# 打乱数据
X, y = shuffle(data, target, random_state=13)
X = X.astype(np.float32)

#划分数据集
offset = int(X.shape[0]*0.9)
x_train, y_train = X[:offset], y[:offset]
x_test, y_test = X[offset:], y[offset:]
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

#训练模型
loss_list, w_list, b_list = training(x_train, y_train, 0.001, 100000)

#预测标签
y_pred = predict(x_test,w_list, b_list)

# 误差下降
plt.figure()
plt.plot(range(len(loss_list)), loss_list)

# 线性模型对数据的拟合不足
plt.figure()
plt.scatter(range(x_test.shape[0]), y_test)
plt.plot(range(x_test.shape[0]), y_pred)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


















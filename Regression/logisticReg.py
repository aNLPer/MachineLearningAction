"""
逻辑回归算法
"""
import numpy as np

#逻辑回归函数
def sigmoid(x):
    """
    逻辑回归函数
    :param x:
    :return:
    """
    z = 1/(1+np.exp(-x))
    return z

#参数初始化
def init_parameters(dim):
    """

    :param dim:数据维度
    :return: 初始化参数
    """
    w = np.zeros((dim,1))
    b = 0
    return w, b

#逻辑回归函数的主体
def logistic(x_train, x_label, w, b):
    """

    :param x_train: 训练数据
    :param x_label: 训练数据标签
    :param w: 参数
    :param b: 偏置
    :return: 损失值、梯度
    """

    num_train = x_train.shape[0]
    num_feature = x_train.shape[1]

    # 计算模型输出
    out = sigmoid(np.dot(x_train, w)+b)
    #计算损失
    cost = - 1/num_train * np.sum(x_label*np.log(out)+(1-x_label)*np.log(1-out))
    #计算梯度
    dw = -1/num_train * np.sum(np.dot(x_train.T, (out - x_label)))
    db = -1/num_train * np.sum(out-x_label)
    return cost, dw, db

#训练模型
def logistic_train(X, y, lr=0.01, epochs=1000):
    """

    :param X: 训练数据集
    :param y: 标签
    :param lr: 学习率
    :param epochs: 轮数
    :return: cost、训练后的参数
    """
    #获取初始化参数
    w, b = init_parameters(X.shape[1])
    cost_list = []
    #迭代训练
    for i in range(epochs):
        #计算损失和梯度
        cost, dw, db = logistic(X, y, w, b)
        cost_list.append(cost)
        #更新参数
        w = w-lr*dw
        b = w-lr*db

    #保存参数
    parameters={"w":w, "b":b}
    return cost_list, parameters

#预测
def predict(x,parameters):
    """

    :param x: 测试集
    :param parameters: 训练阶段产生的参数
    :return: 预测值
    """
    pred_y = sigmoid(np.dot(x, parameters["w"])+parameters["b"])
    for i in range(len(pred_y)):
        if pred_y[i]>=0.5:
            pred_y[i] = 1
        else:
            pred_y[i] = 0
    return pred_y













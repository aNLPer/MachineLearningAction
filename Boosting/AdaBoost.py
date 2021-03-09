import numpy as np
import math

class DecisionStump():
    def __init__(self):
        # 基于划分阈值决定样本分类为1还是-1
        self.polarity = 1
        # 特征索引
        self.feature_index = None
        # 特征划分阈值
        self.threshold = None
        # 指示分类准确率的值
        self.alpha = None

class Adaboost():
    # 若分类器个数
    def __init__(self, n_estimator=5):
        self.n_estimators = n_estimator
        self.estimators = []
    # AdaBoost拟合算法
    def fit(self, X, Y):
        # 数据数量， 特征维度
        n_examples, n_features = X.shape
        # 初始权重分布
        w = np.full((n_examples,), 1/n_examples)

        for _ in range(self.n_estimators):
            # 训练一个若分类器
            clf = DecisionStump()
            # 设置最小的泛化误差
            min_error = float("inf")

            # 在每个特征上找到一个分类规则
            for feature_i in range(n_features):
                # 获取特征值
                feature_values = np.expand_dims(X[:,feature_i], axis=1)
                unique_values = np.unique(feature_values)

                #尝试将每一个特征值作为分类的阈值
                for threshold in unique_values:
                    p = 1
                    # 初始化预测值
                    predictions = np.ones_like(Y)
                    predictions[X[:, feature_i]>threshold] = -1
                    # 计算误差(期望误差)
                    error = np.sum(w[predictions != Y])

                    # 如果误差大于0.5，那么分类规则反转
                    if error>0.5:
                        error = 1-error
                        p = -1
                    #选择在该特征上的选择最佳的分类器
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # 计算该分类器的权重
            clf.alpha = math.log((1-min_error)/(min_error+1e-10),10)

            predictions = np.ones(np.shape(Y))
            negative_idx = (clf.polarity*X[:,clf.feature_index] < clf.polarity*clf.threshold)
            predictions[negative_idx] = -1

            # 更新样本权重
            w *= np.exp(-clf.alpha*Y*predictions)
            w /= np.sum(w)

            #保存该若分类器
            self.estimators.append(clf)








'''
@Project:MachineLearningAction

@Author:lincoln

@File:grammer_test

@Time:2020-03-10 11:24:44

@Description:
'''
import numpy as np
a = np.array([[1,2,3,4], [1,2,3,4]])
print(a.shape)
print(np.ndim(a))
b = np.array([[1,1],[2,2],[3,3],[4,4]])
print(a.dot(b))
x = np.array([1,2,3,4])
print(x.reshape((4,1)))